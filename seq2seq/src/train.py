#!/usr/bin/env python
"""
Fine-tune BARTpho-syllable for Vietnamese GEC (Stage-1 Alirector).

Usage (Colab GPU):
  python -m src.train_bartpho \
      --dataset_name bmd1905/vi-error-correction-v2 \
      --output_dir runs/bartpho-gec \
      --wandb_project vietgec --wandb_entity myteam --wandb_api_key $KEY
"""

import os, argparse, wandb
import torch
import torch.nn.functional as F
from datasets import load_dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)
from functools import lru_cache
try:
    from py_vncorenlp import VnCoreNLP   # noqa: E402
except ImportError:
    VnCoreNLP = None  # will raise later if user requests word segmentation
from typing import List
from transformers.utils.logging import disable_progress_bar
disable_progress_bar()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    # data & model
    p.add_argument("--dataset_name", type=str, default="bmd1905/vi-error-correction-v2")
    p.add_argument("--model_name_or_path", type=str, default="vinai/bartpho-syllable")
    p.add_argument("--max_source_len", type=int, default=384)
    p.add_argument("--max_target_len", type=int, default=384)
    p.add_argument("--output_dir", type=str, required=True)
    # train hyper-params
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_train_epochs", type=int, default=6)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    # wandb
    p.add_argument("--wandb_project", type=str, default="Vi_Alirector_syllable_base")
    p.add_argument("--wandb_entity", type=str, default="phuhuy02003-university-of-transport-and-communications")
    p.add_argument("--wandb_api_key", type=str, default=None)
    p.add_argument("--word_segment", action="store_true",
                   help="Run VNCoreNLP word segmentation before tokenisation "
                        "(required for bartpho-word checkpoints)")
    p.add_argument("--isbf16", type=bool, default=False,
                   help="Use bf16 instead of fp16")
    p.add_argument("--word_segment_save_dir", type=str, default="")
    # Edit-weighted CE
    p.add_argument("--edit_weight", type=float, default=1.0,
                   help="Weight for edited tokens in CE loss (gamma). 1.0 = no weighting, >1.0 = focus on edits")
    # R-Drop regularization
    p.add_argument("--rdrop_weight", type=float, default=0.0,
                   help="Weight for R-Drop regularization. 0.0 = disabled, 0.1-0.5 = typical values")
    return p

# ------------- DEFINE helper ---------------------------------------------------

@lru_cache(maxsize=1)
def get_segmenter(word_segment_save_dir: str):
    """Lazy-load VNCoreNLP only once (fork-safe)."""
    
    return VnCoreNLP(save_dir=word_segment_save_dir,
                annotators=["wseg"])


def segment_batch(texts, args):
    """Segment a list of raw sentences -> list of 'word1_word2' strings."""
    
    # Validate inputs
    if not isinstance(texts, list):
        return texts
    
    # Clean and validate text inputs
    clean_texts = []
    for text in texts:
        if text is None:
            clean_texts.append("")
        elif not isinstance(text, str):
            clean_texts.append(str(text))
        else:
            clean_texts.append(text)
    
    # VnCoreNLP segmentation - process each text individually
    seg = get_segmenter(args.word_segment_save_dir)
    segmented_results = []
    for text in clean_texts:
        # word_segment expects a single string, returns List[List[str]]
        # We take the first (and only) result [0]
        segmented_words = seg.word_segment(text)[0]
        segmented_results.append("".join(segmented_words))
    
    return segmented_results


class EditWeightedCrossEntropyTrainer(Seq2SeqTrainer):
    """Custom trainer that applies higher weights to edited tokens in CE loss and R-Drop regularization."""
    
    def __init__(self, edit_weight=1.0, rdrop_weight=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edit_weight = edit_weight
        self.rdrop_weight = rdrop_weight
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute edit-weighted cross entropy loss with optional R-Drop regularization."""
        labels = inputs.get("labels")
        
        if labels is None:
            outputs = model(**inputs)
            return outputs.loss if return_outputs else (outputs.loss, outputs)
        
        # First forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Compute edit-weighted CE loss
        if self.edit_weight == 1.0:
            loss_ce = outputs.loss  # Standard CE loss
        else:
            # Compute edit-weighted CE loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Create weight tensor: edit_weight for real tokens, 1.0 for padding
            weights = torch.ones_like(shift_labels, dtype=torch.float)
            mask = shift_labels != -100  # non-padding tokens
            weights[mask] = self.edit_weight
            
            # Compute CE loss with weights
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                             shift_labels.view(-1))
            losses = losses.view(shift_labels.shape)
            
            # Apply weights and mask
            weighted_losses = losses * weights
            loss_ce = weighted_losses[mask].mean()
        
        # R-Drop regularization
        if self.rdrop_weight > 0.0 and model.training:
            # Second forward pass with different dropout
            outputs2 = model(**inputs)
            logits2 = outputs2.logits
            
            # Compute KL divergence between two outputs
            # Shift logits for autoregressive models
            shift_logits1 = logits[..., :-1, :].contiguous()
            shift_logits2 = logits2[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Create mask for non-padding tokens
            mask = shift_labels != -100
            
            # Compute log probabilities
            log_p1 = F.log_softmax(shift_logits1, dim=-1)
            log_p2 = F.log_softmax(shift_logits2, dim=-1)
            p1 = F.softmax(shift_logits1, dim=-1)
            p2 = F.softmax(shift_logits2, dim=-1)
            
            # Symmetric KL divergence
            kl_loss1 = F.kl_div(log_p1, p2, reduction='none').sum(-1)  # [B, T]
            kl_loss2 = F.kl_div(log_p2, p1, reduction='none').sum(-1)  # [B, T]
            
            # Apply mask and average
            rdrop_loss = ((kl_loss1 + kl_loss2) * mask.float()).sum() / (2 * mask.sum())
            
            # Add R-Drop loss
            loss = loss_ce + self.rdrop_weight * rdrop_loss
        else:
            loss = loss_ce
        
        return (loss, outputs) if return_outputs else loss


def main():
    args = build_argparser().parse_args()

    # ---- Weights & Biases login ------------------------------------------
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               name=os.path.basename(args.output_dir),
               config=vars(args))

    # ---- Load model & tokenizer ------------------------------------------
    auto_word = "word" in args.model_name_or_path.lower()
    do_segment = args.word_segment or auto_word

    if do_segment and VnCoreNLP is None:
        print("WARNING: py_vncorenlp is not installed but word segmentation was requested.")
        print("Disabling word segmentation and continuing with syllable-level processing.")
        do_segment = False
    elif do_segment:
        # Test if VnCoreNLP actually works before proceeding
        try:
            test_seg = get_segmenter(args.word_segment_save_dir)
            test_result = test_seg.word_segment("Test")
            if not test_result or len(test_result) == 0 or len(test_result[0]) == 0:
                raise RuntimeError("VnCoreNLP test failed - no output")
            print("VnCoreNLP word segmentation is working correctly.")
        except Exception as e:
            print(f"WARNING: VnCoreNLP failed to initialize: {e}")
            print("Disabling word segmentation and continuing with syllable-level processing.")
            do_segment = False

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    # ---- Load HF dataset --------------------------------------------------
    ds = load_dataset(args.dataset_name)
    dataset = DatasetDict(train=ds["train"], validation=ds["validation"],
                          test=ds["test"])

    # ---- Tokenisation -----------------------------------------------------
    def preprocess(examples):
        inputs = examples["incorrect_text"]
        if do_segment:
            inputs = segment_batch(inputs, args)

        model_inputs = tok(
            inputs,
            max_length=args.max_source_len,
            truncation=True
        )

        with tok.as_target_tokenizer():
            labels = tok(
                examples["correct_text"],
                max_length=args.max_target_len,
                truncation=True
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenised = dataset.map(preprocess, batched=True,
                            remove_columns=["incorrect_text", "correct_text"])

    # ---- Data collator ----------------------------------------------------
    collator = DataCollatorForSeq2Seq(tok, model=model)

    # ---- TrainingArguments ------------------------------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        predict_with_generate=True,
        fp16=not args.isbf16,                        # bf16=True if A100
        bf16=args.isbf16,
        report_to=["wandb"],
        logging_steps=20,
        dataloader_num_workers=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )

    # ---- Trainer ----------------------------------------------------------
    trainer = EditWeightedCrossEntropyTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["validation"],
        tokenizer=tok,
        data_collator=collator,
        edit_weight=args.edit_weight,
        rdrop_weight=args.rdrop_weight
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
