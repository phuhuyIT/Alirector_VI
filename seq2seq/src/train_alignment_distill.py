#!/usr/bin/env python
"""
Stage-3 Alirector – distil the forward & reverse alignment teachers
into the correction model with CE + KL loss.

Requirements
------------
* teachers were trained with train_align.py
* dataset_dir is the Stage-2 Arrow folder (input, pred, target)
"""

import os, argparse, wandb, torch
import torch.nn.functional as F
from functools import lru_cache
from datasets import load_from_disk, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer)

# -------------------------------------------------------------------------
# VNCoreNLP helpers (only for bartpho-word)
# -------------------------------------------------------------------------
try:
    from py_vncorenlp import VnCoreNLP
except ImportError:
    VnCoreNLP = None

@lru_cache(maxsize=1)
def get_segmenter():
    if VnCoreNLP is None:
        raise RuntimeError("Install py_vncorenlp & Java ≥8 for word seg.")
    return VnCoreNLP(save_dir="vncorenlp", annotators=["wseg"])

def maybe_segment(texts, seg_flag):
    if not seg_flag:
        return texts
    tok_lists = get_segmenter().word_segment(texts)
    return " ".join(tok_lists)

# -------------------------------------------------------------------------
# Custom collator – pads three separate encoder inputs
# -------------------------------------------------------------------------
class DistilCollator:
    def __init__(self, tok):
        self.tok = tok
        self.pad = tok.pad_token_id

    def __call__(self, batch):
        student_batch = self.tok.pad(
            {k: [b[k] for b in batch] for k in ("input_ids", "attention_mask")},
            return_tensors="pt")
        fwd_batch = self.tok.pad(
            {k: [b[k] for b in batch] for k in ("input_ids_fwd", "attention_mask_fwd")},
            return_tensors="pt")
        rev_batch = self.tok.pad(
            {k: [b[k] for b in batch] for k in ("input_ids_rev", "attention_mask_rev")},
            return_tensors="pt")
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(b["labels"]) for b in batch],
            batch_first=True, padding_value=-100)
        return {**student_batch, **{
            "input_ids_fwd": fwd_batch["input_ids"],
            "attention_mask_fwd": fwd_batch["attention_mask"],
            "input_ids_rev": rev_batch["input_ids"],
            "attention_mask_rev": rev_batch["attention_mask"],
            "labels": labels
        }}

# -------------------------------------------------------------------------
# KD-aware Trainer
# -------------------------------------------------------------------------
class DistilTrainer(Seq2SeqTrainer):
    def __init__(self, teacher_fwd, teacher_rev,
                 alpha, beta, temperature, **kwargs):
        super().__init__(**kwargs)
        self.teacher_fwd = teacher_fwd.eval()
        self.teacher_rev = teacher_rev.eval()
        for p in list(self.teacher_fwd.parameters())+list(self.teacher_rev.parameters()):
            p.requires_grad = False
        self.alpha, self.beta, self.tau = alpha, beta, temperature
        self.kl = torch.nn.KLDivLoss(reduction="batchmean")

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # student forward
        stu_out = model(**{k: inputs[k] for k in ("input_ids","attention_mask")})
        stu_logits = stu_out.logits

        # forward teacher
        with torch.no_grad():
            t_fwd = self.teacher_fwd(input_ids=inputs["input_ids_fwd"],
                                     attention_mask=inputs["attention_mask_fwd"]).logits
            t_rev = self.teacher_rev(input_ids=inputs["input_ids_rev"],
                                     attention_mask=inputs["attention_mask_rev"]).logits

        # ----- losses -------------------------------------------------------
        ce = F.cross_entropy(stu_logits.view(-1, stu_logits.size(-1)),
                             labels.view(-1), ignore_index=-100)

        tau = self.tau
        p_fwd = F.softmax(t_fwd / tau, dim=-1)
        p_rev = F.softmax(t_rev / tau, dim=-1)
        p_teacher = (self.alpha * p_fwd + self.beta * p_rev) / (self.alpha + self.beta)

        log_p_s = F.log_softmax(stu_logits / tau, dim=-1)
        kd = self.kl(log_p_s, p_teacher) * (tau ** 2)

        loss = ce + kd
        return (loss, stu_out) if return_outputs else loss

# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--student_path", required=True)
    p.add_argument("--teacher_fwd_path", required=True)
    p.add_argument("--teacher_rev_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--grad_accum", type=int, default=64)
    # word-seg + val split
    p.add_argument("--word_segment", action="store_true")
    p.add_argument("--val_ratio", type=float, default=0.2)
    # wandb
    p.add_argument("--wandb_project", type=str, default="Vi_Alirector_syllable_base")
    p.add_argument("--wandb_entity", type=str, default="phuhuy02003-university-of-transport-and-communications")
    p.add_argument("--wandb_api_key", type=str, default=None)
    return p

# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------
def main():
    args = build_argparser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    wandb.init(project=args.wandb_project,
               entity=args.wandb_entity,
               name=os.path.basename(args.output_dir),
               config=vars(args))

    auto_word = "word" in args.student_path.lower()
    seg_flag = auto_word or args.word_segment

    tok = AutoTokenizer.from_pretrained(args.student_path, use_fast=True)

    # ---------- load models -------------------------------------------------
    student = AutoModelForSeq2SeqLM.from_pretrained(args.student_path, torch_dtype=torch.float16)
    teacher_fwd = AutoModelForSeq2SeqLM.from_pretrained(args.teacher_fwd_path, torch_dtype=torch.float16)
    teacher_rev = AutoModelForSeq2SeqLM.from_pretrained(args.teacher_rev_path, torch_dtype=torch.float16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    student.to(device); teacher_fwd.to(device); teacher_rev.to(device)

    # ---------- load data ---------------------------------------------------
    ds = load_from_disk(args.dataset_dir)
    if isinstance(ds, DatasetDict) and "validation" in ds:
        train_ds, val_ds = ds["train"], ds["validation"]
    else:
        split = ds.train_test_split(seed=42, test_size=args.val_ratio)
        train_ds, val_ds = split["train"], split["test"]

    sep_tok = tok.eos_token

    def preprocess(batch):
        X = maybe_segment(batch["input"], seg_flag)
        Y_hat = maybe_segment(batch["pred"], seg_flag)
        # student source = X
        stu_enc = tok(X, truncation=True, max_length=256)
        # teachers
        fwd_src = [f"{x} {sep_tok} {y}" for x, y in zip(X, Y_hat)]
        rev_src = [f"{y} {sep_tok} {x}" for x, y in zip(X, Y_hat)]
        fwd_enc = tok(fwd_src, truncation=True, max_length=256)
        rev_enc = tok(rev_src, truncation=True, max_length=256)
        with tok.as_target_tokenizer():
            lab_enc = tok(maybe_segment(batch["target"], seg_flag),
                          truncation=True, max_length=192)
        return {
            "input_ids": stu_enc["input_ids"],
            "attention_mask": stu_enc["attention_mask"],
            "input_ids_fwd": fwd_enc["input_ids"],
            "attention_mask_fwd": fwd_enc["attention_mask"],
            "input_ids_rev": rev_enc["input_ids"],
            "attention_mask_rev": rev_enc["attention_mask"],
            "labels": lab_enc["input_ids"]
        }

    train_ds = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

    collator = DistilCollator(tok)

    # ---------- training args ----------------------------------------------
    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
        predict_with_generate=False,
        report_to=["wandb"],
        logging_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )

    trainer = DistilTrainer(
        teacher_fwd=teacher_fwd,
        teacher_rev=teacher_rev,
        alpha=args.alpha,
        beta=args.beta,
        temperature=args.temperature,
        model=student,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
