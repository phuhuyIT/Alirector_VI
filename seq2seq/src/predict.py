#!/usr/bin/env python
"""
Generate draft corrections Ŷ for every X in a dataset (Stage-2 Alirector).

Example (Colab):

    python -m src.predict \
        --model_path runs/bartpho-syl-gec \
        --dataset_name bmd1905/vi-error-correction-v2 \
        --split train \
        --output_dir data/stage2/train_pred

    # bartpho-word, with auto word-segmentation
    python -m src.predict \
        --model_path runs/bartpho-word-gec \
        --dataset_name bmd1905/vi-error-correction-v2 \
        --split train \
        --output_dir data/stage2/train_pred
"""
import argparse, os, torch, json, time
import wandb
from functools import lru_cache
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    from py_vncorenlp import VnCoreNLP   # optional
except ImportError:
    VnCoreNLP = None


# --------------------------------------------------------------------------
# Word-segmentation helpers (only needed for bartpho-word checkpoints)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_segmenter():
    if VnCoreNLP is None:
        raise RuntimeError("py_vncorenlp not installed — "
                           "pip install py_vncorenlp && ensure Java ≥8")
    return VnCoreNLP(save_dir="vncorenlp", annotators=["wseg"])


def segment_sentences(batch):
    seg = get_segmenter()
    tok_lists = seg.word_segment(batch["input"])
    batch["input"] = [" ".join(tok_lists)]
    return batch


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="Path or HF id of the fine-tuned Stage-1 model")
    p.add_argument("--dataset_name", type=str,
                   help="HF dataset id or path (if previously saved)")
    p.add_argument("--dataset_path", type=str,
                   help="Local path created by save_to_disk(); "
                        "takes precedence over --dataset_name")
    p.add_argument("--split", type=str, default="train",
                   help="Dataset split to run prediction on")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--beam_size", type=int, default=5)
    p.add_argument("--max_len", type=int, default=192)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--wandb_project", type=str, default="Vi_Alirector_syllable_base")
    p.add_argument("--wandb_entity", type=str, default="phuhuy02003-university-of-transport-and-communications")
    p.add_argument("--wandb_api_key", type=str, default=None)
    p.add_argument("--word_segment", type=bool, default=False,
                   help="Force VNCoreNLP segmentation regardless of checkpoint")
    p.add_argument("--fp16", type=bool, default=False,
                   help="Generate in FP16 (recommended on L4/A100)")
    return p


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    args = build_argparser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Weights & Biases login ------------------------------------------
    use_wandb = bool(args.wandb_project)
    if use_wandb:
        if args.wandb_api_key:
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.init(project=args.wandb_project,
                   entity=args.wandb_entity,
                   name=f"predict-{os.path.basename(args.model_path)}",
                   config=vars(args))
    # Detect whether we need word segmentation
    auto_word = "word" in os.path.basename(args.model_path).lower()
    do_segment = args.word_segment or auto_word

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path,
                                                  torch_dtype=torch.float16
                                                  if args.fp16 else None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Load dataset
    if args.dataset_path:
        dataset = load_from_disk(args.dataset_path)[args.split]
    else:
        dataset = load_dataset(args.dataset_name, split=args.split,
                               streaming=False)

    # Optional segmentation
    if do_segment:
        dataset = dataset.map(segment_sentences,
                              batched=True, batch_size=1024,
                              num_proc=1)

    # Generation loop -------------------------------------------------------
    seen = 0
    t0 = time.time()
    def generate_batch(batch):
        with torch.no_grad():
            inputs = tokenizer(batch["incorrect_text"],
                               padding=True,
                               truncation=True,
                               max_length=args.max_len,
                               return_tensors="pt").to(device)
            gen_ids = model.generate(
                **inputs,
                num_beams=args.beam_size,
                max_length=args.max_len,
                early_stopping=True
            )
            preds = tokenizer.batch_decode(gen_ids,
                                           skip_special_tokens=True,
                                           clean_up_tokenization_spaces=True)
            batch["pred"] = preds
            if use_wandb:
                nonlocal seen
                seen += len(batch["incorrect_text"])
                elapsed = time.time() - t0
                wandb.log({
                    "sentences_processed": seen,
                    "sentences_per_sec": seen / elapsed,
                    "avg_latency_ms": 1000 * elapsed / seen
                })
        return batch

    dataset = dataset.map(generate_batch,
                          batched=True,
                          batch_size=args.batch_size,
                          remove_columns=[c for c in dataset.column_names
                                          if c not in ("incorrect_text", "correct_text")])

    # Rename gold column -> target for clarity
    dataset = dataset.rename_column("correct_text", "target")

    # Save to disk for Stage-2 training
    dataset.save_to_disk(args.output_dir)
    # -------------- log preview table to W&B ------------------------------
    if use_wandb:
        import pandas as pd
        preview_df = pd.DataFrame(dataset.select(range(min(20, len(dataset)))))
        wandb.log({"preview": wandb.Table(dataframe=preview_df)})
        wandb.finish()
    # Also dump a small JSONL preview for sanity-check
    preview_path = os.path.join(args.output_dir, "head10.jsonl")
    with open(preview_path, "w", encoding="utf8") as fo:
        for ex in dataset.select(range(10)):
            fo.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"✓ Saved Stage-2 data to {args.output_dir}")
    print(f"  Preview → {preview_path}")


if __name__ == "__main__":
    main()
