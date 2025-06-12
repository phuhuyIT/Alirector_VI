#!/usr/bin/env python
"""
Stage-3 of Alirector — distil forward & reverse alignment teachers
into the original correction model (student).

◦ student  : BARTpho-(syllable|word) checkpoint from Stage-1
◦ teachers : forward & reverse alignment checkpoints from Stage-2
◦ data     : folder created by predict.py    (columns: input, pred, target)
"""

import os, argparse, torch, wandb
from functools import lru_cache
from datasets import load_from_disk
import torch.nn.functional as F
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer)

# ───── optional word-segmentation ────────────────────────────────────────────
try:
    from py_vncorenlp import VnCoreNLP          # :contentReference[oaicite:1]{index=1}
except ImportError:
    VnCoreNLP = None

def segment_batch(texts):
    seg = get_segmenter()
    return [" ".join(ws) for ws in seg.tokenize(texts)]

@lru_cache(maxsize=1)
def get_segmenter():
    if VnCoreNLP is None:
        raise RuntimeError("pip install py_vncorenlp and install Java ≥8")
    return VnCoreNLP(save_dir="vncorenlp", annotators=["wseg"])

# ───── argument parser ───────────────────────────────────────────────────────
def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--student_path", required=True)
    p.add_argument("--teacher_fwd_path", required=True)
    p.add_argument("--teacher_rev_path", required=True)
    p.add_argument("--output_dir", required=True)

    # train & KD hyper-params
    p.add_argument("--alpha", type=float, default=0.5)   # weigh fwd teacher
    p.add_argument("--beta",  type=float, default=1.5)   # KD vs NLL
    p.add_argument("--tau",   type=float, default=1.0)
    p.add_argument("--lr",    type=float, default=3e-5)
    p.add_argument("--batch", type=int,   default=4)
    p.add_argument("--epochs", type=int,  default=3)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--val_ratio", type=float, default=0.2)

    # W&B
    p.add_argument("--wandb_project", type=str, default="Vi_Alirector_syllable_base")
    p.add_argument("--wandb_entity", type=str, default="phuhuy02003-university-of-transport-and-communications")
    p.add_argument("--wandb_api_key", type=str, default=None)
    p.add_argument("--word_segment",  action="store_true")
    return p

# ───── custom Trainer to add KD loss ─────────────────────────────────────────
class DistilTrainer(Seq2SeqTrainer):
    def __init__(self, teacher_fwd, teacher_rev, alpha, beta, tau, *args, **kw):
        super().__init__(*args, **kw)
        self.teacher_fwd, self.teacher_rev = teacher_fwd, teacher_rev
        self.alpha, self.beta, self.tau = alpha, beta, tau
        for t in (self.teacher_fwd, self.teacher_rev):
            t.eval(); t.requires_grad_(False)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        # ─ student forward pass ─
        outputs_stu = model(**{k: v for k, v in inputs.items()
                               if k != "labels"})
        logits_stu = outputs_stu.logits
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss_nll = loss_fct(logits_stu.view(-1, logits_stu.size(-1)),
                            labels.view(-1))

        # ─ teacher logits (no_grad) ─
        with torch.no_grad():
            logits_fwd = self.teacher_fwd(**inputs).logits
            logits_rev = self.teacher_rev(**inputs).logits

        # ─ KL distillation ─
        t = self.tau
        log_p_stu = F.log_softmax(logits_stu / t, dim=-1)
        p_fwd = F.softmax(logits_fwd / t, dim=-1)
        p_rev = F.softmax(logits_rev / t, dim=-1)

        kl_fwd = F.kl_div(log_p_stu, p_fwd, reduction="batchmean") * (t**2)
        kl_rev = F.kl_div(log_p_stu, p_rev, reduction="batchmean") * (t**2)
        loss_kd = self.alpha * kl_fwd + (1 - self.alpha) * kl_rev

        loss = loss_nll + self.beta * loss_kd
        return (loss, outputs_stu) if return_outputs else loss

# ───── main ─────────────────────────────────────────────────────────────────
def main():
    args = build_argparser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               name=os.path.basename(args.output_dir), config=vars(args))

    auto_word = "word" in args.student_path.lower()
    seg_needed = auto_word or args.word_segment

    tok = AutoTokenizer.from_pretrained(args.student_path, use_fast=True)
    student = AutoModelForSeq2SeqLM.from_pretrained(
        args.student_path,
        torch_dtype=torch.float16 if args.fp16 else None)
    teacher_fwd = AutoModelForSeq2SeqLM.from_pretrained(
        args.teacher_fwd_path, torch_dtype=torch.float16 if args.fp16 else None)
    teacher_rev = AutoModelForSeq2SeqLM.from_pretrained(
        args.teacher_rev_path, torch_dtype=torch.float16 if args.fp16 else None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    student.to(device); teacher_fwd.to(device); teacher_rev.to(device)

    ds = load_from_disk(args.dataset_dir)
    if "validation" in ds:
        train_ds, val_ds = ds["train"], ds["validation"]
    else:
        splitter = ds.train_test_split(test_size=args.val_ratio, seed=42)
        train_ds, val_ds = splitter["train"], splitter["test"]

    sep_tok = tok.eos_token

    def build_inputs(batch):
        src = batch["input"]; hyp = batch["pred"]
        if seg_needed:
            src = segment_batch(src); hyp = segment_batch(hyp)
        batch["source"] = src
        batch["labels_text"] = batch["target"]
        return batch

    train_ds = train_ds.map(build_inputs, batched=True, batch_size=1024)
    val_ds   = val_ds.map(build_inputs,   batched=True, batch_size=1024)

    # tokenise once for all three models (same tokenizer/vocab)
    def tok_fn(batch):
        model_in = tok(batch["source"], truncation=True, max_length=192)
        with tok.as_target_tokenizer():
            labels = tok(batch["labels_text"], truncation=True, max_length=192)
        model_in["labels"] = labels["input_ids"]
        return model_in

    train_tok = train_ds.map(tok_fn, batched=True,
                             remove_columns=train_ds.column_names)
    val_tok   = val_ds.map(tok_fn,   batched=True,
                           remove_columns=val_ds.column_names)

    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=args.fp16,
        logging_steps=200,
        report_to=["wandb"],
        metric_for_best_model="loss",
        load_best_model_at_end=True
    )

    trainer = DistilTrainer(
        model=student,
        teacher_fwd=teacher_fwd,
        teacher_rev=teacher_rev,
        alpha=args.alpha, beta=args.beta, tau=args.tau,
        args=train_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tok,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    wandb.finish()

if __name__ == "__main__":
    main()
