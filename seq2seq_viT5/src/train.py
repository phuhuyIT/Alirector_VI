#!/usr/bin/env python
"""
Stage-1 Alirector – fine-tune ViT5 for Vietnamese Grammar Error Correction.

Major differences from the BARTpho implementation:
1. Instruction prefix – every source sentence is prepended with the string "gec: ".
2. FlashAttention – can be enabled via the --flash_attn flag for much faster
   training if you have PyTorch ≥2.1 and the `flash_attn` library installed.
3. No word segmentation needed by default (ViT5 is syllable based).  A
   --word_segment flag is still provided for completeness / experimentation.

Example (A100 GPU, bf16 + FlashAttention):

python -m src.train \
       --dataset_name bmd1905/vi-error-correction-v2 \
       --output_dir runs/vit5-gec \
       --flash_attn --isbf16 \
       --wandb_project vietgec
"""

import os, argparse, wandb
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn.functional as F
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.utils.logging import disable_progress_bar

disable_progress_bar()

# ---------------------------------------------------------------------------
# Optional VNCoreNLP word segmentation (mostly unused for ViT5)
# ---------------------------------------------------------------------------
try:
    from py_vncorenlp import VnCoreNLP  # noqa: E402
except ImportError:
    VnCoreNLP = None  # will raise later if user requests word segmentation


@lru_cache(maxsize=1)
def get_segmenter(word_segment_save_dir: str):
    if VnCoreNLP is None:
        raise RuntimeError("py_vncorenlp not installed – pip install py_vncorenlp")
    return VnCoreNLP(save_dir=word_segment_save_dir, annotators=["wseg"])


def segment_batch(texts: List[str], args):
    """Optionally segment a list of raw sentences with VNCoreNLP."""
    if not args.word_segment:
        return texts
    seg = get_segmenter(args.word_segment_save_dir)
    res = []
    for t in texts:
        seg_words = seg.word_segment(t)[0]
        res.append(" ".join(seg_words))
    return res


# ---------------------------------------------------------------------------
# Custom Trainer – edit-weighted CE + optional R-Drop
# ---------------------------------------------------------------------------
class EditWeightedCrossEntropyTrainer(Seq2SeqTrainer):
    def __init__(
            self,
            *args,
            edit_weight: float = 1.0,
            label_smoother: Optional[LabelSmoother] = None,
            **kwargs,
        ) -> None:
            super().__init__(*args, label_smoother=label_smoother, **kwargs)
            self.edit_weight = float(edit_weight)
            if self.edit_weight <= 0:
                raise ValueError("edit_weight must be positive.")

    # ---------------------------------------------------------------------
    # Core logic -----------------------------------------------------------
    # ---------------------------------------------------------------------
    def compute_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[Any, ...]]]:
        """Override HF Trainer's loss computation to insert per‑token weights."""

        labels = inputs.pop("labels")  # (bs, tgt_len)
        # Forward pass -----------------------------------------------------
        outputs = model(**inputs, labels=labels)
        logits = outputs.logits  # (bs, tgt_len, vocab)

        # Shift so that tokens <t> predict t+1 --------------------------------
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_src    = inputs["input_ids"][:, 1:].contiguous()  # align with tgt
        vocab_size   = shift_logits.size(-1)

        # ------------------------------------------------------------------
        #  Edit mask: tokens that are different *and* not padding (-100)
        # ------------------------------------------------------------------
        non_pad     = shift_labels != -100
        edit_mask   = (shift_labels != shift_src) & non_pad

        # Per‑token weights -------------------------------------------------
        weights = torch.ones_like(shift_labels, dtype=shift_logits.dtype)
        if self.edit_weight != 1.0:
            weights[edit_mask] = self.edit_weight

        # Cross‑entropy over vocabulary ------------------------------------
        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        loss_flat = loss_fct(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
        ).view_as(shift_labels)

        # Apply weights, ignore padding ------------------------------------
        loss = (loss_flat * weights)[non_pad].mean()

        # Optional label‑smoothing after weighting -------------------------
        if self.label_smoother is not None and self.label_smoother.smoothing > 0:
            smooth_loss = self.label_smoother(outputs, shift_labels)
            loss = loss + smooth_loss

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser()
    # data & model ----------------------------------------------------------
    p.add_argument("--dataset_name", type=str, default="bmd1905/vi-error-correction-v2")
    p.add_argument("--model_name_or_path", type=str, default="VietAI/vit5-base")
    p.add_argument("--max_source_len", type=int, default=384)
    p.add_argument("--max_target_len", type=int, default=384)
    p.add_argument("--output_dir", type=str, required=True)
    # optimisation ---------------------------------------------------------
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_train_epochs", type=int, default=6)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    # misc flags -----------------------------------------------------------
    p.add_argument("--flash_attn", action="store_true",
                   help="Enable FlashAttention v2 during model loading (requires torch >=2.1)")
    p.add_argument("--isbf16", action="store_true",
                   help="Train with bfloat16. If not set, fp16 autocast will be used")
    p.add_argument("--word_segment", action="store_true",
                   help="Apply VNCoreNLP word segmentation before tokenisation")
    p.add_argument("--word_segment_save_dir", type=str, default="")
    # regularisation -------------------------------------------------------
    p.add_argument("--edit_weight", type=float, default=1.0,
                   help="Gamma for edit-weighted CE loss")
    # WandB ---------------------------------------------------------------
    p.add_argument("--wandb_project", type=str, default="vietgec")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_api_key", type=str, default=None)
    return p


# ---------------------------------------------------------------------------
# Utility – load model with optional FlashAttention
# ---------------------------------------------------------------------------

def load_vit5_with_flashattention(name: str, dtype, flash: bool):
    if not flash:
        return AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=dtype)
    try:
        # transformers >=4.38 – native arg
        return AutoModelForSeq2SeqLM.from_pretrained(
            name, torch_dtype=dtype, use_flash_attention_2=True
        )
    except TypeError:
        # Older transformers – toggle via config field
        cfg = AutoConfig.from_pretrained(name)
        if hasattr(cfg, "use_flash_attention_2"):
            cfg.use_flash_attention_2 = True
        elif hasattr(cfg, "use_flash_attention"):
            cfg.use_flash_attention = True
        return AutoModelForSeq2SeqLM.from_pretrained(name, config=cfg, torch_dtype=dtype)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main():
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ----------- Weights & Biases ----------------------------------------
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    if args.wandb_project:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=os.path.basename(args.output_dir), config=vars(args))

    # ----------- Tokeniser & model ---------------------------------------
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    dtype = torch.bfloat16 if args.isbf16 else (torch.float16 if torch.cuda.is_available() else None)
    model = load_vit5_with_flashattention(args.model_name_or_path, dtype, args.flash_attn)

    # ----------- Dataset --------------------------------------------------
    raw_ds = load_dataset(args.dataset_name)
    dataset = DatasetDict(train=raw_ds["train"], validation=raw_ds["validation"], test=raw_ds["test"])

    # ----------- Pre-processing ------------------------------------------
    prefix = "gec: "

    def preprocess(batch):
        inputs = [prefix + s for s in batch["incorrect_text"]]
        inputs = segment_batch(inputs, args)
        model_inputs = tok(inputs, max_length=args.max_source_len, truncation=True)
        with tok.as_target_tokenizer():
            labels = tok(batch["correct_text"], max_length=args.max_target_len, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenised = dataset.map(preprocess, batched=True,
                            remove_columns=["incorrect_text", "correct_text"])

    # ----------- Data collator -------------------------------------------
    collator = DataCollatorForSeq2Seq(tok, model=model)

    # ----------- TrainingArguments ---------------------------------------
    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        fp16=not args.isbf16,
        bf16=args.isbf16,
        report_to=["wandb"] if args.wandb_project else [],
        logging_steps=100,
        dataloader_num_workers=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )

    # ----------- Trainer --------------------------------------------------
    trainer = EditWeightedCrossEntropyTrainer(
        model=model,
        args=targs,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["validation"],
        tokenizer=tok,
        data_collator=collator,
        edit_weight=args.edit_weight
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
