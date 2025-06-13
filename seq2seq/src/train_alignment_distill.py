#!/usr/bin/env python
"""
Stage-3 Bidirectional Alignment Distillation for Alirector (Vietnamese).

– student  : Stage-1 correction model (BARTpho-syl | -word)
– teachers : Stage-2 forward & reverse alignment models
"""

import os, argparse, wandb, torch
from functools import lru_cache
from datasets import load_from_disk, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer)
import torch.nn.functional as F

# ---------- VNCoreNLP (optional) -------------------------------------------
try:
    from py_vncorenlp import VnCoreNLP
except ImportError:
    VnCoreNLP = None


@lru_cache(maxsize=1)
def get_segmenter():
    if VnCoreNLP is None:
        raise RuntimeError(
            "py_vncorenlp not installed – pip install py_vncorenlp & Java≥8"
        )
    return VnCoreNLP(save_dir="vncorenlp", annotators=["wseg"])


def segment_batch(texts):
    seg = get_segmenter()
    tok_lists = seg.tokenize(texts)
    return [" ".join(toks) for toks in tok_lists]


# ---------------- CLI -------------------------------------------------------
def build_args():
    p = argparse.ArgumentParser()
    # paths
    p.add_argument("--dataset_dir", required=True,
                   help="Folder from predict.py with columns input/pred/target")
    p.add_argument("--student_path", required=True,
                   help="Stage-1 correction checkpoint")
    p.add_argument("--teacher_fwd_path", required=True,
                   help="Stage-2 forward aligner")
    p.add_argument("--teacher_rev_path", required=True,
                   help="Stage-2 reverse aligner")
    p.add_argument("--output_dir", required=True)
    # training
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--num_train_epochs", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--max_source_len", type=int, default=256)
    p.add_argument("--max_target_len", type=int, default=192)
    # KD hyper-params
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--temp", type=float, default=1.0)
    # misc
    p.add_argument("--val_ratio", type=float, default=0.20)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--word_segment", action="store_true",
                   help="Force VNCoreNLP segmentation")
    # W&B
    p.add_argument("--wandb_project", type=str, default="Vi_Alirector_syllable_base")
    p.add_argument("--wandb_entity", type=str, default="phuhuy02003-university-of-transport-and-communications")
    p.add_argument("--wandb_api_key", type=str, default=None)
    return p.parse_args()


# ---------------- custom Trainer -------------------------------------------
class DistilTrainer(Seq2SeqTrainer):

    def __init__(self, teacher_fwd, teacher_rev, tau, alpha, beta, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_fwd, self.t_rev = teacher_fwd, teacher_rev
        self.tau, self.alpha, self.beta = tau, alpha, beta
        for p in self.t_fwd.parameters():
            p.requires_grad = False
        for p in self.t_rev.parameters():
            p.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        # -------- unpack inputs --------------------------------------------
        s_inp = {"input_ids": inputs.pop("stu_ids"),
                 "attention_mask": inputs.pop("stu_att")}
        f_inp = {"input_ids": inputs.pop("fwd_ids"),
                 "attention_mask": inputs.pop("fwd_att")}
        r_inp = {"input_ids": inputs.pop("rev_ids"),
                 "attention_mask": inputs.pop("rev_att")}

        # -------- student forward ------------------------------------------
        out_s = model(**s_inp, labels=labels)
        loss_ce = out_s.loss
        logit_s = out_s.logits / self.tau
        log_probs_s = F.log_softmax(logit_s, dim=-1)

        # -------- teachers --------------------------------------------------
        with torch.no_grad():
            logit_f = self.t_fwd(**f_inp, labels=labels).logits / self.tau
            logit_r = self.t_rev(**r_inp, labels=labels).logits / self.tau
            prob_f = F.softmax(logit_f, dim=-1)
            prob_r = F.softmax(logit_r, dim=-1)

        # KL divergence (batch-mean)
        kl_f = F.kl_div(log_probs_s, prob_f, reduction="batchmean")
        kl_r = F.kl_div(log_probs_s, prob_r, reduction="batchmean")
        loss_kd = self.alpha * kl_f + (1.0 - self.alpha) * kl_r

        loss = loss_ce + self.beta * loss_kd
        return (loss, out_s) if return_outputs else loss


# ---------------- main -----------------------------------------------------
def main():
    args = build_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- W&B -------------------------------------------------------------
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    wandb.init(project=args.wandb_project,
               entity=args.wandb_entity,
               name=os.path.basename(args.output_dir),
               config=vars(args))

    # ---- detect segmentation --------------------------------------------
    auto_word = "word" in args.student_path.lower()
    seg_needed = auto_word or args.word_segment

    # ---- tokenizer -------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.student_path, use_fast=True)

    # ---- models ----------------------------------------------------------
    dtype = torch.float16 if args.fp16 else None
    student = AutoModelForSeq2SeqLM.from_pretrained(args.student_path,
                                                    torch_dtype=dtype)
    teacher_fwd = AutoModelForSeq2SeqLM.from_pretrained(args.teacher_fwd_path,
                                                        torch_dtype=dtype)
    teacher_rev = AutoModelForSeq2SeqLM.from_pretrained(args.teacher_rev_path,
                                                        torch_dtype=dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    student.to(device); teacher_fwd.to(device); teacher_rev.to(device)
    teacher_fwd.eval(); teacher_rev.eval()

    # ---- load Stage-2 data ----------------------------------------------
    ds = load_from_disk(args.dataset_dir)
    if isinstance(ds, DatasetDict) and "validation" in ds:
        train_set, val_set = ds["train"], ds["validation"]
    else:
        split = ds.train_test_split(seed=42, test_size=args.val_ratio)
        train_set, val_set = split["train"], split["test"]

    sep_tok = tok.eos_token

    # ---- preprocessing ---------------------------------------------------
    def preprocess(batch):
        X, Y_hat, Y = batch["input"], batch["pred"], batch["target"]
        if seg_needed:
            X = segment_batch(X)
            Y_hat = segment_batch(Y_hat)
        # sources
        stu_src = X
        fwd_src = [f"{x} {sep_tok} {yhat}" for x, yhat in zip(X, Y_hat)]
        rev_src = [f"{yhat} {sep_tok} {x}" for x, yhat in zip(X, Y_hat)]

        tok_stu = tok(stu_src, max_length=args.max_source_len,
                      truncation=True, padding="max_length")
        tok_fwd = tok(fwd_src, max_length=args.max_source_len,
                      truncation=True, padding="max_length")
        tok_rev = tok(rev_src, max_length=args.max_source_len,
                      truncation=True, padding="max_length")
        with tok.as_target_tokenizer():
            labels = tok(Y, max_length=args.max_target_len,
                         truncation=True, padding="max_length").input_ids

        return {
            "stu_ids": tok_stu.input_ids,
            "stu_att": tok_stu.attention_mask,
            "fwd_ids": tok_fwd.input_ids,
            "fwd_att": tok_fwd.attention_mask,
            "rev_ids": tok_rev.input_ids,
            "rev_att": tok_rev.attention_mask,
            "labels": labels
        }

    train_set = train_set.map(preprocess, batched=True,
                              remove_columns=train_set.column_names)
    val_set = val_set.map(preprocess, batched=True,
                          remove_columns=val_set.column_names)

    # ---- training arguments ---------------------------------------------
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
        fp16=args.fp16,
        logging_steps=500,
        report_to=["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )

    trainer = DistilTrainer(
        teacher_fwd=teacher_fwd,
        teacher_rev=teacher_rev,
        tau=args.temp,
        alpha=args.alpha,
        beta=args.beta,
        model=student,
        args=targs,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=tok,
        # DataCollator not needed – everything is same length after padding
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
