#!/usr/bin/env python
"""
Evaluate a trained Vietnamese GEC model.

Metrics: F0.5 (token diff), GLEU, chrF++  (optionally BLEU)
Usage (Colab):
  python -m src.evaluate_gec \
      --model_path runs/distilled_syl \
      --dataset_name bmd1905/vi-error-correction-v2 \
      --split test \
      --output_dir eval_out \
      --wandb_project vietgec
"""
import os, argparse, json, torch
from functools import lru_cache
from difflib import SequenceMatcher
from tqdm import tqdm
import wandb

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Fix sacrebleu imports with better error handling
try:
    import sacrebleu
    from sacrebleu.metrics import CHRF, BLEU
    SACREBLEU_AVAILABLE = True
    print("sacrebleu available - BLEU and chrF++ metrics enabled")
except ImportError:
    print("ERROR: sacrebleu not installed. Install with: pip install sacrebleu")
    SACREBLEU_AVAILABLE = False

# GLEU is available in the evaluate library from Hugging Face
try:
    import evaluate
    GLEU_AVAILABLE = True
    print("evaluate library available - GLEU metric enabled")
except ImportError:
    print("WARNING: evaluate library not installed. Install with: pip install evaluate")
    print("GLEU metric will not be available")
    GLEU_AVAILABLE = False

# ───────────── VNCoreNLP (for bartpho-word) ─────────────
try:
    from py_vncorenlp import VnCoreNLP        # noqa: E402
except ImportError:
    VnCoreNLP = None

@lru_cache(maxsize=1)
def get_segmenter():
    if VnCoreNLP is None:
        raise RuntimeError("Install py_vncorenlp and Java ≥ 8 for word segmentation.")
    return VnCoreNLP(save_dir="vncorenlp", annotators=["wseg"])

def maybe_segment(sentences, needed):
    """Segment sentences if needed, with proper error handling."""
    if not needed:
        return sentences
    
    try:
        seg = get_segmenter()
        # Fix: VnCoreNLP.word_segment() expects individual strings, not a list
        # Process each sentence individually
        segmented_results = []
        for sentence in sentences:
            if sentence is None or sentence.strip() == "":
                segmented_results.append("")
                continue
            try:
                # word_segment returns List[List[str]] for each sentence
                segmented_words = seg.word_segment(sentence)[0]
                segmented_results.append(" ".join(segmented_words))
            except Exception as e:
                print(f"Warning: Failed to segment sentence '{sentence[:50]}...': {e}")
                segmented_results.append(sentence)  # Fallback to original
        return segmented_results
    except Exception as e:
        print(f"Warning: Segmentation failed: {e}. Using original sentences.")
        return sentences

# ───────────── F-beta (token diff) ─────────────
def extract_edit_positions(src_tokens, tgt_tokens):
    """Return a set of source indices that need changing."""
    edits = set()
    sm = SequenceMatcher(None, src_tokens, tgt_tokens)
    for tag, i1, i2, _, _ in sm.get_opcodes():
        if tag != "equal":
            edits.update(range(i1, i2))
    return edits

def corpus_fbeta(orig_list, pred_list, gold_list, beta=0.5):
    TP = FP = FN = 0
    for src, sys, ref in zip(orig_list, pred_list, gold_list):
        s_tok, p_tok, g_tok = src.split(), sys.split(), ref.split()
        sys_edits  = extract_edit_positions(s_tok, p_tok)
        gold_edits = extract_edit_positions(s_tok, g_tok)
        TP += len(sys_edits & gold_edits)
        FP += len(sys_edits - gold_edits)
        FN += len(gold_edits - sys_edits)
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec  = TP / (TP + FN) if TP + FN else 0.0
    beta2 = beta * beta
    if prec + rec == 0:
        f = 0.0
    else:
        f = (1 + beta2) * prec * rec / (beta2 * prec + rec)
    return prec, rec, f

# ───────────── Argument parser ─────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--dataset_name", type=str,
                   help="HF dataset id (e.g. bmd1905/vi-error-correction-v2)")
    p.add_argument("--dataset_dir", type=str,
                   help="Local dataset folder from save_to_disk (takes priority)")
    p.add_argument("--split", default="validation")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--max_len", type=int, default=384)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--calc_bleu", action="store_true",
                   help="Also compute BLEU (slower)")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--word_segment", action="store_true")
    # W&B
    p.add_argument("--wandb_project", type=str, default="Vi_Alirector_syllable_base")
    p.add_argument("--wandb_entity", type=str, default="phuhuy02003-university-of-transport-and-communications")
    p.add_argument("--wandb_api_key", type=str, default=None)
    p.add_argument("--log_predictions", action="store_true",
                   help="Log first 100 preds as W&B table")
    p.add_argument("--subset_ratio", type=float, default=0.05)
    return p.parse_args()

# ───────────── Main ─────────────
def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # W&B
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               name=f"eval-{os.path.basename(args.model_path)}",
               config=vars(args))

    # detect segmentation need
    auto_seg = "word" in args.model_path.lower()
    seg_need = auto_seg or args.word_segment

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if args.fp16 else None).eval().to("cuda")

    # dataset
    ds = (load_from_disk(args.dataset_dir)[args.split]
          if args.dataset_dir else
          load_dataset(args.dataset_name, split=args.split))
    # use 5% of ds
    ds = ds.train_test_split(test_size=args.subset_ratio, seed=42)["test"]
    src_texts   = ds["incorrect_text"]
    gold_texts  = ds["correct_text"] if "correct_text" in ds.column_names else ds["target"]
    
    # Apply segmentation if needed
    if seg_need:
        print("Applying word segmentation...")
        src_texts   = maybe_segment(src_texts, seg_need)
        gold_texts  = maybe_segment(gold_texts, seg_need)

    # generation
    sys_texts = []
    for i in tqdm(range(0, len(src_texts), args.batch_size),
                  desc="Generating"):
        batch_sent = src_texts[i:i + args.batch_size]
        enc = tok(batch_sent, return_tensors="pt",
                  padding=True, truncation=True,
                  max_length=args.max_len).to("cuda")
        with torch.no_grad():
            out_ids = model.generate(**enc,
                                     num_beams=args.beam,
                                     max_length=args.max_len,
                                     early_stopping=True)
        sys_texts.extend(tok.batch_decode(out_ids,
                                          skip_special_tokens=True))

    # save hypotheses
    pred_file = os.path.join(args.output_dir, "pred.txt")
    with open(pred_file, "w", encoding="utf8") as fo:
        fo.write("\n".join(sys_texts))
    print(f"✓ Saved predictions → {pred_file}")

    # ───────────── Metrics ─────────────
    # F0.5 (token diff)
    prec, rec, f05 = corpus_fbeta(src_texts, sys_texts, gold_texts, beta=0.5)

    # sacreBLEU metrics with better error handling
    gleu, chrf, bleu = None, None, None
    
    if not SACREBLEU_AVAILABLE:
        print("ERROR: Cannot compute GLEU/BLEU/chrF++ - sacrebleu not available")
        print("Install with: pip install sacrebleu")
    else:
        try:
            # chrF++
            chrf_metric = CHRF(word_order=2)
            chrf = chrf_metric.corpus_score(sys_texts, [gold_texts]).score
            
            # BLEU (optional)
            if args.calc_bleu:
                bleu_metric = BLEU()
                bleu = bleu_metric.corpus_score(sys_texts, [gold_texts]).score
                
        except Exception as e:
            print(f"ERROR computing sacrebleu metrics: {e}")
            print("This might be due to:")
            print("1. Empty predictions/references")
            print("2. Incompatible sacrebleu version")
            print("3. Encoding issues")

    if GLEU_AVAILABLE:
        try:
            # GLEU using evaluate library
            gleu_metric = evaluate.load("gleu")
            # GLEU expects references as list of lists for each prediction
            references_formatted = [[ref] for ref in gold_texts]
            gleu_result = gleu_metric.compute(predictions=sys_texts, references=references_formatted)
            gleu = gleu_result["gleu"] * 100  # Scale to 0-100 range like other metrics
        except Exception as e:
            print(f"ERROR computing GLEU metric: {e}")
            print("This might be due to:")
            print("1. Empty predictions/references")
            print("2. Incompatible evaluate version")
            print("3. Encoding issues")
            gleu = None

    # log / print
    print(f"F0.5  : {f05*100:6.2f} (P={prec*100:.2f}, R={rec*100:.2f})")
    if gleu is not None:
        print(f"GLEU  : {gleu:6.2f}")
    if chrf is not None:
        print(f"chrF++: {chrf:6.2f}")
    if bleu is not None:
        print(f"BLEU  : {bleu:6.2f}")

    # Log to wandb with safe values
    wandb.log({
        "F0.5": f05, "Precision": prec, "Recall": rec,
        "GLEU": gleu/100.0 if gleu is not None else None,
        "chrF++": chrf/100.0 if chrf is not None else None,
        "BLEU": bleu/100.0 if bleu is not None else None
    })
    
    # optional prediction table
    if args.log_predictions:
        table = wandb.Table(columns=["src", "pred", "gold"])
        for s, p, g in zip(src_texts[:100], sys_texts[:100], gold_texts[:100]):
            table.add_data(s, p, g)
        wandb.log({"samples": table})

    wandb.finish()

if __name__ == "__main__":
    main()