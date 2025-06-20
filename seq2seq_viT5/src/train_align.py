#!/usr/bin/env python
"""
Stage-2 Alirector – train forward / reverse alignment models for ViT5.
Flows:
  1. The Stage-2 dataset is produced by predict.py and contains columns
     [incorrect_text, pred, target].
  2. For a *forward* model we feed:  "gec: X <extra_id_0> Ŷ" → Y (gold correct)
     For a *reverse* model we feed:  "gec: Ŷ <extra_id_0> X" → Y
  3. There is **no** word-segmentation by default because ViT5 is syllable-
     based.  The --word_segment flag is provided for experimentation.

Key differences vs BARTpho version:
  • Instruction prefix "gec: "
  • Separator token is the sentinel <extra_id_0>
  • Optional FlashAttention flag (shared with Stage-1)
"""

import os, argparse, wandb, torch
from functools import lru_cache
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# ───────────────────────────  VNCoreNLP  ────────────────────────────────
try:
    from py_vncorenlp import VnCoreNLP
except ImportError:
    VnCoreNLP = None

@lru_cache(maxsize=1)
def get_segmenter(word_segment_save_dir: str):
    if VnCoreNLP is None:
        raise RuntimeError("Install py_vncorenlp for word segmentation")
    return VnCoreNLP(save_dir=word_segment_save_dir, annotators=["wseg"])

def maybe_segment(texts, needed, args):
    if not needed:
        return texts
    seg = get_segmenter(args.word_segment_save_dir)
    res = []
    for t in texts:
        res.append(" ".join(seg.word_segment(t)[0]))
    return res

# ───────────────────────────  Args  ─────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser()
    # data & checkpoints ---------------------------------------------------
    p.add_argument("--dataset_dir", required=True,
                   help="Folder created by predict.py – contains Arrow files")
    p.add_argument("--model_name_or_path", default="VietAI/vit5-base")
    p.add_argument("--direction", choices=["forward", "reverse"], default="forward")
    p.add_argument("--output_dir", required=True)
    # training hparams -----------------------------------------------------
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--num_train_epochs", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_source_len", type=int, default=384)
    p.add_argument("--max_target_len", type=int, default=384)
    p.add_argument("--beam_size", type=int, default=5)
    # misc -----------------------------------------------------------------
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--flash_attn", action="store_true",
                   help="Enable FlashAttention 2 whilst loading model")
    # word-seg -------------------------------------------------------------
    p.add_argument("--word_segment", action="store_true")
    p.add_argument("--word_segment_save_dir", type=str, default="")
    # W&B ------------------------------------------------------------------
    p.add_argument("--wandb_project", type=str, default="vietgec")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_api_key", type=str, default=None)
    return p

# ───────────────────  Helper: load with FlashAttention  ─────────────────

def load_model(name, dtype, flash):
    if not flash:
        return AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=dtype)
    try:
        return AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=dtype, use_flash_attention_2=True)
    except TypeError:
        cfg = AutoConfig.from_pretrained(name)
        if hasattr(cfg, "use_flash_attention_2"):
            cfg.use_flash_attention_2 = True
        elif hasattr(cfg, "use_flash_attention"):
            cfg.use_flash_attention = True
        return AutoModelForSeq2SeqLM.from_pretrained(name, config=cfg, torch_dtype=dtype)

# ───────────────────────────  Main  ─────────────────────────────────────

def main():
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # wandb ---------------------------------------------------------------
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    if args.wandb_project:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=os.path.basename(args.output_dir), config=vars(args))

    # tokenizer & model ----------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    dtype = torch.float16 if args.fp16 else (torch.float32)
    model = load_model(args.model_name_or_path, dtype, args.flash_attn)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # dataset --------------------------------------------------------------
    ds = load_from_disk(args.dataset_dir)
    if isinstance(ds, DatasetDict) and "validation" in ds:
        train_ds = ds["train"]
        dev_ds = ds["validation"]
    else:
        split = ds.train_test_split(test_size=0.2, seed=42)
        train_ds, dev_ds = split["train"], split["test"]

    # build alignment inputs ----------------------------------------------
    prefix = "gec: "
    sep_tok = "<extra_id_0>"
    order_src_first = args.direction == "forward"

    def concat_examples(batch):
        src = batch["incorrect_text"]
        hyp = batch["pred"]
        seg_needed = args.word_segment  # ViT5 is syllable‐based; only if user requests
        if seg_needed:
            src = maybe_segment(src, True, args)
        merged = []
        for s, h in zip(src, hyp):
            if order_src_first:
                merged.append(f"{prefix}{s} {sep_tok} {h}")
            else:
                merged.append(f"{prefix}{h} {sep_tok} {s}")
        batch["source"] = merged
        return batch

    columns_to_remove = [c for c in train_ds.column_names if c not in ("source", "target")]
    train_ds = train_ds.map(concat_examples, batched=True, batch_size=1024,
                            remove_columns=columns_to_remove)
    dev_ds = dev_ds.map(concat_examples, batched=True, batch_size=1024,
                        remove_columns=columns_to_remove)

    # tokenisation ---------------------------------------------------------
    def tok_fn(batch):
        model_inputs = tokenizer(batch["source"], truncation=True, max_length=args.max_source_len)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(batch["target"], truncation=True, max_length=args.max_target_len)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["source", "target"])
    dev_ds = dev_ds.map(tok_fn, batched=True, remove_columns=["source", "target"])

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        predict_with_generate=True,
        generation_num_beams=args.beam_size,
        fp16=args.fp16,
        report_to=["wandb"] if args.wandb_project else [],
        logging_steps=200,
        metric_for_best_model="loss",
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
