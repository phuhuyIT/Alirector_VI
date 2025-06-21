#!/usr/bin/env python
"""
Evaluate a ViT5-based Vietnamese GEC model (all three stages).

Differences from the BARTpho version:
• Adds the instruction prefix "gec: " to every input sentence before tokenisation.
• Uses <extra_id_…> tokens internally, but generation / metrics stay unchanged.
• No automatic word-segmentation – enabled only if --word_segment is passed.

Example:
    python -m src.evaluate_gec \
        --model_path runs/distilled_vit5 \
        --dataset_name bmd1905/vi-error-correction-v2 \
        --split test \
        --output_dir eval_vit5
"""
import os, argparse, json, torch
from functools import lru_cache
from difflib import SequenceMatcher
from tqdm import tqdm
import wandb

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# —— sacrebleu & evaluate (optional) ————————————
try:
    import sacrebleu
    from sacrebleu.metrics import CHRF, BLEU
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False

try:
    import evaluate
    GLEU_AVAILABLE = True
except ImportError:
    GLEU_AVAILABLE = False

# —— VNCoreNLP (optional word segmentation) ———————
try:
    from py_vncorenlp import VnCoreNLP
except ImportError:
    VnCoreNLP = None

@lru_cache(maxsize=1)
def get_segmenter(word_segment_save_dir: str):
    if VnCoreNLP is None:
        raise RuntimeError("Install py_vncorenlp for word segmentation")
    return VnCoreNLP(save_dir=word_segment_save_dir or "vncorenlp", annotators=["wseg"])

def maybe_segment(texts, needed, args):
    if not needed:
        return texts
    seg = get_segmenter(args.word_segment_save_dir)
    out = []
    for t in texts:
        try:
            words = seg.word_segment(t)[0]
            out.append(" ".join(words))
        except Exception:
            out.append(t)
    return out

# ——  F0.5 (token-level diff) ————————————————————
def extract_edit_positions(src_tokens, tgt_tokens):
    edits = set()
    sm = SequenceMatcher(None, src_tokens, tgt_tokens)
    for tag, i1, i2, _, _ in sm.get_opcodes():
        if tag != "equal":
            edits.update(range(i1, i2))
    return edits

def corpus_fbeta(src_list, pred_list, gold_list, beta=0.5):
    TP = FP = FN = 0
    for s, p, g in zip(src_list, pred_list, gold_list):
        s_tok, p_tok, g_tok = s.split(), p.split(), g.split()
        sys_edits = extract_edit_positions(s_tok, p_tok)
        gold_edits = extract_edit_positions(s_tok, g_tok)
        TP += len(sys_edits & gold_edits)
        FP += len(sys_edits - gold_edits)
        FN += len(gold_edits - sys_edits)
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec  = TP / (TP + FN) if TP + FN else 0.0
    beta2 = beta * beta
    f = 0.0 if prec + rec == 0 else (1 + beta2) * prec * rec / (beta2 * prec + rec)
    return prec, rec, f

# ——  Argparser ———————————————————————————————

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    # dataset (HF hub or load_from_disk)
    p.add_argument("--dataset_name", type=str)
    p.add_argument("--dataset_dir", type=str)
    p.add_argument("--split", default="validation")
    # generation params
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--max_len", type=int, default=384)
    # misc
    p.add_argument("--output_dir", required=True)
    p.add_argument("--subset_ratio", type=float, default=0.05)
    p.add_argument("--calc_bleu", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--word_segment", action="store_true")
    p.add_argument("--word_segment_save_dir", type=str, default="")
    # WandB
    p.add_argument("--wandb_project", type=str, default="vietgec")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_api_key", type=str, default=None)
    p.add_argument("--log_predictions", action="store_true")
    return p

# ——  Main ——————————————————————————————————————

def main():
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               name=f"eval-{os.path.basename(args.model_path)}", config=vars(args))

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16 if args.fp16 else None
    ).eval().to("cuda")

    # dataset -------------------------------------------------------------
    if args.dataset_dir:
        ds = load_from_disk(args.dataset_dir)[args.split]
    else:
        ds = load_dataset(args.dataset_name, split=args.split)

    if 0 < args.subset_ratio < 1.0:
        ds = ds.train_test_split(test_size=args.subset_ratio, seed=42)["test"]

    raw_src   = ds["incorrect_text"]
    gold_text = ds["correct_text"] if "correct_text" in ds.column_names else ds["target"]

    # segmentation (optional) --------------------------------------------
    seg_needed = args.word_segment
    if seg_needed:
        raw_src   = maybe_segment(raw_src, seg_needed, args)
        gold_text = maybe_segment(gold_text, seg_needed, args)

    prefix = "gec: "
    src_texts = [prefix + s for s in raw_src]

    # generation ----------------------------------------------------------
    sys_texts = []
    for idx in tqdm(range(0, len(src_texts), args.batch_size), desc="Generating"):
        batch = src_texts[idx: idx + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len).to("cuda")
        with torch.no_grad():
            out_ids = model.generate(**enc, num_beams=args.beam, max_length=args.max_len, early_stopping=True)
        sys_texts.extend(tok.batch_decode(out_ids, skip_special_tokens=True))

    # save preds ----------------------------------------------------------
    pred_path = os.path.join(args.output_dir, "pred.txt")
    with open(pred_path, "w", encoding="utf8") as fo:
        fo.write("\n".join(sys_texts))
    print(f"✓ Predictions saved → {pred_path}")

    # metrics -------------------------------------------------------------
    prec, rec, f05 = corpus_fbeta(raw_src, sys_texts, gold_text)
    gleu = chrf = bleu = None

    if SACREBLEU_AVAILABLE:
        try:
            chrf = CHRF(word_order=2).corpus_score(sys_texts, [gold_text]).score
            if args.calc_bleu:
                bleu = BLEU().corpus_score(sys_texts, [gold_text]).score
        except Exception as e:
            print("sacrebleu error:", e)

    if GLEU_AVAILABLE:
        try:
            gleu_metric = evaluate.load("gleu")
            refs_formatted = [[g] for g in gold_text]
            gleu = gleu_metric.compute(predictions=sys_texts, references=refs_formatted)["gleu"] * 100
        except Exception as e:
            print("GLEU error:", e)

    # print / log ---------------------------------------------------------
    print(f"F0.5  : {f05*100:6.2f} (P={prec*100:.2f}, R={rec*100:.2f})")
    if gleu is not None: print(f"GLEU  : {gleu:6.2f}")
    if chrf is not None: print(f"chrF++: {chrf:6.2f}")
    if bleu is not None: print(f"BLEU  : {bleu:6.2f}")

    wandb.log({
        "F0.5": f05,
        "Precision": prec,
        "Recall": rec,
        "GLEU": gleu/100 if gleu is not None else None,
        "chrF++": chrf/100 if chrf is not None else None,
        "BLEU": bleu/100 if bleu is not None else None,
    })

    if args.log_predictions:
        tbl = wandb.Table(columns=["src", "pred", "gold"])
        for s, p, g in zip(raw_src[:100], sys_texts[:100], gold_text[:100]):
            tbl.add_data(s, p, g)
        wandb.log({"samples": tbl})

    wandb.finish()


if __name__ == "__main__":
    main()
