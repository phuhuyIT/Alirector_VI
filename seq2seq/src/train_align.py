#!/usr/bin/env python
"""
Stage-2 Alirector – train a forward or reverse ALIGNMENT model.

* expects a dataset saved by predict.py with columns: input, pred, target
* builds source sentence:  X </s> Ŷ   (direction=forward)
                           Ŷ </s> X   (direction=reverse)
* works with bartpho-syllable or bartpho-word (auto word-segmentation)
* logs to Weights & Biases
"""

import os, argparse, wandb, torch
from functools import lru_cache
from datasets import load_from_disk, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

# ---------- optional Vietnamese word-segmentation ---------------------------
try:
    from py_vncorenlp import VnCoreNLP      # pip install py_vncorenlp
except ImportError:
    VnCoreNLP = None


# ---------------------------------------------------------------------------
# helpers for VNCoreNLP
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_segmenter(word_segment_save_dir: str):
    if VnCoreNLP is None:
        raise RuntimeError("py_vncorenlp not installed – "
                           "pip install py_vncorenlp and ensure Java ≥8")
    return VnCoreNLP(save_dir=word_segment_save_dir, annotators=["wseg"])


def wseg_sentence(sent: str, word_segment_save_dir: str) -> str:
    """Return sentence with multi-syllable words joined by underscores."""
    seg = get_segmenter(word_segment_save_dir)
    return " ".join(seg.word_segment(sent)[0])


def maybe_segment(texts, seg_needed, args):
    """Segment a list[str] if seg_needed else return original list."""
    if not seg_needed:
        return texts
    seg = get_segmenter(args.word_segment_save_dir)
    # word_segment expects a single string, returns List[List[str]]
    # Process each text individually
    segmented_results = []
    for text in texts:
        segmented_words = seg.word_segment(text)[0]  # Take first (and only) result
        segmented_results.append("".join(segmented_words))
    return segmented_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser()
    # data & checkpoints
    p.add_argument("--dataset_dir", required=True,
                   help="Folder produced by predict.py (contains Arrow files)")
    p.add_argument("--model_name_or_path", default="vinai/bartpho-syllable")
    p.add_argument("--direction", choices=["forward", "reverse"],
                   default="forward",
                   help="Input order: forward = X ⟂ Ŷ, reverse = Ŷ ⟂ X")
    p.add_argument("--output_dir", required=True)
    # training hyper-params
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--num_train_epochs", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_source_len", type=int, default=384)
    p.add_argument("--max_target_len", type=int, default=384)
    # misc
    p.add_argument("--beam_size", type=int, default=5)
    p.add_argument("--fp16", action="store_true")

    # word-seg flag
    p.add_argument("--word_segment", action="store_true",
                   help="Force VNCoreNLP segmentation")
    p.add_argument("--word_segment_save_dir", type=str, default="")
    # W&B
    p.add_argument("--wandb_project", type=str, default="Vi_Alirector_syllable_base")
    p.add_argument("--wandb_entity", type=str, default="phuhuy02003-university-of-transport-and-communications")
    p.add_argument("--wandb_api_key", type=str, default=None)
    return p


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = build_argparser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- W&B ---------------------------------------------------------
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    wandb.init(project=args.wandb_project,
               entity=args.wandb_entity,
               name=os.path.basename(args.output_dir),
               config=vars(args))

    # ---------- detect if BARTpho-word → need segmentation ------------------
    auto_word = "word" in args.model_name_or_path.lower()
    seg_needed = auto_word or args.word_segment

    # ---------- tokenizer & model ------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if args.fp16 else None
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # ---------- load Stage-2 dataset ---------------------------------------
    ds = load_from_disk(args.dataset_dir)
    if isinstance(ds, DatasetDict) and "validation" in ds:
        dataset = ds["train"]
        dev = ds["validation"]
    else:
        # --- NEW 80/20 split ------------------------------------------------
        print(f"No validation split found; creating {int(0.2*100)}% "
              f"hold-out from the loaded data")
        split = ds.train_test_split(test_size=0.2, seed=42)
        dataset, dev = split["train"], split["test"]

    # ---------- build alignment input --------------------------------------
    sep_tok = tokenizer.eos_token  # "</s>" for BARTpho
    order_src_first = args.direction == "forward"

    def concat_examples(batch):
        src = batch["incorrect_text"]
        hyp = batch["pred"]
        if seg_needed:
            src = maybe_segment(src, True, args)   # only segment X
        merged = []
        for s, h in zip(src, hyp):
            if order_src_first:
                merged.append(f"{s} {sep_tok} {h}")
            else:
                merged.append(f"{h} {sep_tok} {s}")
        batch["source"] = merged
        return batch

    dataset = dataset.map(concat_examples,
                          batched=True, batch_size=1024,
                          remove_columns=[c for c in dataset.column_names
                                          if c not in ("source", "target")])

    if dev:
        dev = dev.map(concat_examples,
                      batched=True, batch_size=1024,
                      remove_columns=[c for c in dev.column_names
                                      if c not in ("source", "target")])

    # ---------- tokenisation ----------------------------------------------
    def tok_fn(batch):
        model_inputs = tokenizer(batch["source"],
                                 truncation=True,
                                 max_length=args.max_source_len)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(batch["target"],
                               truncation=True,
                               max_length=args.max_target_len)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = dataset.map(tok_fn, batched=True,
                          remove_columns=["source", "target"])
    if dev:
        dev = dev.map(tok_fn, batched=True,
                      remove_columns=["source", "target"])

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # ---------- training args ---------------------------------------------
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

        report_to=["wandb"],
        logging_steps=200,
        metric_for_best_model="loss",
        load_best_model_at_end=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=dataset,
        eval_dataset=dev if dev else None,
        tokenizer=tokenizer,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
