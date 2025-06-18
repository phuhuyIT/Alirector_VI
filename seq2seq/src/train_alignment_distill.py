#!/usr/bin/env python
"""
Stage-3 Alirector — distil alignment teachers (A, A-rev) into the student
corrector C.  Works with bartpho-syllable or bartpho-word.

Example:
python -m src.train_align_distill \
        --dataset_dir data/stage2/train_pred \
        --student_path runs/bartpho-syl-gec \
        --teacher_fwd_path runs/align_fwd_syl \
        --teacher_rev_path runs/align_rev_syl \
        --output_dir runs/distilled_syl \
        --wandb_project vietgec
"""
import os, argparse, wandb, torch, torch.nn.functional as F
from functools import lru_cache
from datasets import load_from_disk
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq)
import torch
from torch.nn.functional import pad
from dataclasses import dataclass
from typing import List, Dict, Any
@dataclass
class DataCollatorWithTeachers:
    tokenizer: Any
    model: Any = None                      # keeps Trainer happy
    pad_to_multiple_of: int | None = None  # optional hf arg

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # split teacher vs student keys -----------------------------
        teacher_keys = ["fwd_ids", "fwd_mask", "rev_ids", "rev_mask"]
        teacher_batches = {k: [] for k in teacher_keys}
        student_feats   = []

        for feat in features:
            for k in teacher_keys:
                teacher_batches[k].append(torch.tensor(feat.pop(k)))
            student_feats.append(feat)

        # let HF collator pad the *student* part --------------------
        base = DataCollatorForSeq2Seq(self.tokenizer, model=self.model,
                                      pad_to_multiple_of=self.pad_to_multiple_of)(
                                            student_feats)

        # pad teacher sequences to same max_len ---------------------
        for ids_key, mask_key in [("fwd_ids", "fwd_mask"),
                                  ("rev_ids", "rev_mask")]:
            ids_list  = teacher_batches[ids_key]
            mask_list = teacher_batches[mask_key]
            max_len = max(len(x) for x in ids_list)

            padded_ids  = [pad(x, (0, max_len-len(x)),
                               value=self.tokenizer.pad_token_id)
                           for x in ids_list]
            padded_mask = [pad(x, (0, max_len-len(x)), value=0)
                           for x in mask_list]

            base[ids_key]  = torch.stack(padded_ids)
            base[mask_key] = torch.stack(padded_mask)

        return base

# ─────────────────────────  VNCoreNLP helper  ────────────────────────────
try:
    from py_vncorenlp import VnCoreNLP       # :contentReference[oaicite:1]{index=1}
except ImportError:
    VnCoreNLP = None

@lru_cache(maxsize=1)
def get_segmenter(word_segment_save_dir: str):
    if VnCoreNLP is None:
        raise RuntimeError("Install py_vncorenlp for word segmentation")
    return VnCoreNLP(save_dir=word_segment_save_dir, annotators=["wseg"])

def maybe_segment(texts, needed: bool, args):
    if not needed:
        return texts
    seg = get_segmenter(args.word_segment_save_dir)
    # word_segment expects a single string, returns List[List[str]]
    # Process each text individually
    segmented_results = []
    for text in texts:
        segmented_words = seg.word_segment(text)[0]  # Take first (and only) result
        segmented_results.append(" ".join(segmented_words))
    return segmented_results

# ────────────────────────────  Args  ─────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--student_path", required=True)
    p.add_argument("--teacher_fwd_path", required=True)
    p.add_argument("--teacher_rev_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--alpha", type=float, default=0.9,
                   help="weight for forward-KL")      # :contentReference[oaicite:2]{index=2}
    p.add_argument("--beta", type=float, default=0.5,
                   help="weight for reverse-KL")
    p.add_argument("--tau", type=float, default=1.0,
                   help="temperature for KD")
    # training
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=64)
    p.add_argument("--fp16", action="store_true")
    # word-seg
    p.add_argument("--word_segment", action="store_true")
    p.add_argument("--word_segment_save_dir", type=str, default="")
    # wandb
    p.add_argument("--wandb_project", type=str, default="Vi_Alirector_syllable_base")
    p.add_argument("--wandb_entity", type=str, default="phuhuy02003-university-of-transport-and-communications")
    p.add_argument("--wandb_api_key", type=str, default=None)
    return p

# ────────────────────────────  Trainer  ──────────────────────────────────
class DistilTrainer(Seq2SeqTrainer):
    def __init__(self, *args, teacher_fwd=None, teacher_rev=None,
                 alpha=1.0, beta=0.5, tau=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_fwd = teacher_fwd.eval()
        self.teacher_rev = teacher_rev.eval()
        for p in self.teacher_fwd.parameters():
            p.requires_grad_(False)
        for p in self.teacher_rev.parameters():
            p.requires_grad_(False)
        self.alpha, self.beta, self.tau = alpha, beta, tau

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        student_args = {
            k: inputs[k]
            for k in ("input_ids", "attention_mask", "labels")
            if k in inputs
        }
        outputs_s = model(**student_args)      
        loss_ce = outputs_s.loss                                     # NLL
        # ----- teacher logits (no grad) -----------------------------
        with torch.no_grad():
            tf_logits = self.teacher_fwd(
                input_ids=inputs["fwd_ids"],
                attention_mask=inputs["fwd_mask"],
                labels=labels).logits / self.tau
            tr_logits = self.teacher_rev(
                input_ids=inputs["rev_ids"],
                attention_mask=inputs["rev_mask"],
                labels=labels).logits / self.tau
        st_logits = outputs_s.logits / self.tau
        tf_prob  = F.softmax(tf_logits, dim=-1)
        tr_prob  = F.softmax(tr_logits, dim=-1)
        st_logp = F.log_softmax(st_logits, dim=-1)
        # ----- KL loss with proper masking -------------------------------
        mask_token = (labels != -100)            # [B, T]
        # full matrices
        kl_fwd_all = F.kl_div(st_logp, tf_prob, reduction="none")   # [B,T,V]
        kl_rev_all = F.kl_div(st_logp, tr_prob, reduction="none")

        # sum over vocabulary dimension,  keep token dimension
        kl_fwd_tok = kl_fwd_all.sum(-1)          # [B, T]
        kl_rev_tok = kl_rev_all.sum(-1)          # [B, T]

        # apply mask and normalise
        denom = mask_token.sum()                 # number of real tokens
        kld_fwd = (kl_fwd_tok * mask_token).sum() / denom
        kld_rev = (kl_rev_tok * mask_token).sum() / denom

        loss = loss_ce + self.alpha * kld_fwd + self.beta * kld_rev

        return (loss, outputs_s) if return_outputs else loss

# ─────────────────────────────  Main  ────────────────────────────────────
def main():
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               name=os.path.basename(args.output_dir), config=vars(args))

    # detect segmentation need
    auto_word = "word" in args.student_path.lower()
    seg_needed = auto_word or args.word_segment

    tok = AutoTokenizer.from_pretrained(args.student_path, use_fast=True)
    student = AutoModelForSeq2SeqLM.from_pretrained(
        args.student_path,
        torch_dtype=torch.float16 if args.fp16 else None)

    teacher_fwd = AutoModelForSeq2SeqLM.from_pretrained(
        args.teacher_fwd_path,
        torch_dtype=torch.float16 if args.fp16 else None)
    teacher_rev = AutoModelForSeq2SeqLM.from_pretrained(
        args.teacher_rev_path,
        torch_dtype=torch.float16 if args.fp16 else None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    student.to(device); teacher_fwd.to(device); teacher_rev.to(device)

    # ---------- data ------------------------------------------------------
    ds = load_from_disk(args.dataset_dir)
    if "validation" not in ds:
        ds = ds.train_test_split(test_size=0.20, seed=42)             # 80/20
    sep_tok = tok.eos_token
    def build_src(batch):
        x   = maybe_segment(batch["incorrect_text"], seg_needed, args)
        y0  = maybe_segment(batch["pred"], seg_needed, args)
        fwd = [f"{a} {sep_tok} {b}" for a, b in zip(x, y0)]
        rev = [f"{b} {sep_tok} {a}" for a, b in zip(x, y0)]
        batch.update({"src_student": x,
                      "src_fwd": fwd,
                      "src_rev": rev,
                      "labels_text": batch["target"]})
        return batch
    ds = ds.map(build_src, batched=True, batch_size=1024)

    def encode(b):
        stu = tok(b["src_student"], truncation=True, max_length=256)
        fwd = tok(b["src_fwd"],     truncation=True, max_length=256)
        rev = tok(b["src_rev"],     truncation=True, max_length=256)
        with tok.as_target_tokenizer():
            lbl = tok(b["labels_text"], truncation=True, max_length=192)
        out = {
            # student
            "input_ids":        stu["input_ids"],
            "attention_mask":   stu["attention_mask"],
            # teachers
            "fwd_ids":          fwd["input_ids"],
            "fwd_mask":         fwd["attention_mask"],
            "rev_ids":          rev["input_ids"],
            "rev_mask":         rev["attention_mask"],
            # gold
            "labels":           lbl["input_ids"]
        }
        return out
    keep = {"input_ids", "attention_mask", "labels",
            "fwd_ids", "fwd_mask", "rev_ids", "rev_mask"}
    ds = ds.map(encode, batched=True,
                remove_columns=[c for c in ds["train"].column_names
                                if c not in keep])

    collator = DataCollatorWithTeachers(tokenizer=tok, model=student)

    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        remove_unused_columns=False,
        report_to=["wandb"],
        fp16=args.fp16,
        predict_with_generate=True,
        generation_num_beams=5,
        save_total_limit=3,
        logging_steps=20
    )

    trainer = DistilTrainer(
        model=student,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tok,
        data_collator=collator,
        teacher_fwd=teacher_fwd,
        teacher_rev=teacher_rev,
        alpha=args.alpha, beta=args.beta, tau=args.tau
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    wandb.finish()

if __name__ == "__main__":
    main()
