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
    p.add_argument("--max_source_len", type=int, default=192)
    p.add_argument("--max_target_len", type=int, default=192)
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
    return p

# ------------- DEFINE helper ---------------------------------------------------

@lru_cache(maxsize=1)
def get_segmenter():
    """Lazy-load VNCoreNLP only once (fork-safe)."""
    return VnCoreNLP(save_dir= "/content/vncorenlp",
                annotators=["wseg"])


def segment_batch(texts):
    """Segment a list of raw sentences -> list of 'word1_word2' strings."""
    
    seg = get_segmenter()
    # seg.tokenize expects List[str] and returns List[List[str]]
    out = seg.word_segment(texts)
    return " ".join(out)

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
        raise RuntimeError("py_vncorenlp is not installed but --word_segment "
                           "was set or bartpho-word detected.")

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    # ---- Load HF dataset --------------------------------------------------
    ds = load_dataset(args.dataset_name)
    # create 10k dev split
    dev_size = 100000
    ds = ds["train"].train_test_split(test_size=dev_size, seed=42)
    dataset = DatasetDict(train=ds["train"], validation=ds["test"],
                          test=load_dataset(args.dataset_name)["test"])

    # ---- Tokenisation -----------------------------------------------------
    def preprocess(examples):
        inputs = examples["input"]
        if do_segment:
            inputs = segment_batch(inputs)

        model_inputs = tok(
            inputs,
            max_length=args.max_source_len,
            truncation=True
        )

        with tok.as_target_tokenizer():
            labels = tok(
                examples["output"],
                max_length=args.max_target_len,
                truncation=True
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenised = dataset.map(preprocess, batched=True,
                            remove_columns=["input", "output"])

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
        logging_steps=500,
        dataloader_num_workers=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )

    # ---- Trainer ----------------------------------------------------------
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["validation"],
        tokenizer=tok,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
