#!/usr/bin/env python
"""
Stage-3 Alirector – Knowledge-Distillation of alignment teachers into the
ViT5 student.

Differences vs the BARTpho implementation:
  • Instruction prefix "gec: " for every student / teacher source.
  • Separator token is the sentinel <extra_id_0>.
  • Optional FlashAttention when loading all three models.
  • Edit-weighted CE + τ-scaled KD with α, β, τ hyper-params.
"""

import os, argparse, wandb, torch, torch.nn.functional as F
from functools import lru_cache
from dataclasses import dataclass
from typing import List, Dict, Any
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

# ───────────────────────────  VNCoreNLP util  ────────────────────────────
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
    return [" ".join(seg.word_segment(t)[0]) for t in texts]

# ───────────────────────────  Data collator w/ teachers  ─────────────────
@dataclass
class DataCollatorWithTeachers:
    tokenizer: Any
    model: Any = None
    pad_to_multiple_of: int | None = None

    def __call__(self, features: List[Dict[str, Any]]):
        teacher_keys = ["fwd_ids", "fwd_mask", "rev_ids", "rev_mask"]
        batches = {k: [] for k in teacher_keys}
        student_feats = []
        for feat in features:
            for k in teacher_keys:
                batches[k].append(torch.tensor(feat.pop(k)))
            student_feats.append(feat)
        base = DataCollatorForSeq2Seq(self.tokenizer, model=self.model,
                                      pad_to_multiple_of=self.pad_to_multiple_of)(student_feats)
        for ids_key, mask_key in [("fwd_ids", "fwd_mask"), ("rev_ids", "rev_mask")]:
            ids_list, mask_list = batches[ids_key], batches[mask_key]
            max_len = max(len(x) for x in ids_list)
            pad_id = self.tokenizer.pad_token_id
            base[ids_key] = torch.stack([torch.nn.functional.pad(x, (0, max_len - len(x)), value=pad_id) for x in ids_list])
            base[mask_key] = torch.stack([torch.nn.functional.pad(x, (0, max_len - len(x)), value=0) for x in mask_list])
        return base

# ───────────────────────────  Args  ─────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--student_path", required=True)
    p.add_argument("--teacher_fwd_path", required=True)
    p.add_argument("--teacher_rev_path", required=True)
    p.add_argument("--output_dir", required=True)
    # KD params -----------------------------------------------------------
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta",  type=float, default=0.3)
    p.add_argument("--tau",   type=float, default=4.0)
    # edit-weighted CE ----------------------------------------------------
    p.add_argument("--edit_weight", type=float, default=1.0)
    # training ------------------------------------------------------------
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=64)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--flash_attn", action="store_true")
    # word-seg ------------------------------------------------------------
    p.add_argument("--word_segment", action="store_true")
    p.add_argument("--word_segment_save_dir", type=str, default="")
    # wandb ---------------------------------------------------------------
    p.add_argument("--wandb_project", type=str, default="vietgec")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_api_key", type=str, default=None)
    return p

# ───────────────────  Helper – load with FlashAttention  ────────────────

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

# ───────────────────────────  Trainer  ──────────────────────────────────
class DistilTrainer(Seq2SeqTrainer):
    def __init__(self, *args, teacher_fwd=None, teacher_rev=None, alpha=1.0, beta=0.5, tau=1.0, edit_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_fwd = teacher_fwd.eval()
        self.teacher_rev = teacher_rev.eval()
        for p in self.teacher_fwd.parameters():
            p.requires_grad_(False)
        for p in self.teacher_rev.parameters():
            p.requires_grad_(False)
        self.alpha, self.beta, self.tau, self.edit_weight = alpha, beta, tau, edit_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        student_args = {k: inputs[k] for k in ("input_ids", "attention_mask", "labels") if k in inputs}
        outputs_s = model(**student_args)

        # ----- Edit-weighted CE ------------------------------------------
        if self.edit_weight == 1.0:
            loss_ce = outputs_s.loss
        else:
            logits = outputs_s.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            weights = torch.ones_like(shift_labels, dtype=torch.float)
            mask = shift_labels != -100
            weights[mask] = self.edit_weight
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view_as(shift_labels)
            loss_ce = (token_losses * weights)[mask].mean()

        # ----- Teachers (no grad) ----------------------------------------
        with torch.no_grad():
            tf_logits = self.teacher_fwd(input_ids=inputs["fwd_ids"], attention_mask=inputs["fwd_mask"], labels=labels).logits / self.tau
            tr_logits = self.teacher_rev(input_ids=inputs["rev_ids"], attention_mask=inputs["rev_mask"], labels=labels).logits / self.tau
        st_logits = outputs_s.logits / self.tau

        tf_prob = F.softmax(tf_logits, dim=-1)
        tr_prob = F.softmax(tr_logits, dim=-1)
        st_logp = F.log_softmax(st_logits, dim=-1)

        mask_tok = labels != -100
        kl_fwd = F.kl_div(st_logp, tf_prob, reduction="none").sum(-1)
        kl_rev = F.kl_div(st_logp, tr_prob, reduction="none").sum(-1)
        denom = mask_tok.sum()
        kld_fwd = (kl_fwd * mask_tok).sum() / denom
        kld_rev = (kl_rev * mask_tok).sum() / denom

        kld_fwd *= self.tau ** 2
        kld_rev *= self.tau ** 2
        loss = loss_ce + self.alpha * kld_fwd + self.beta * kld_rev
        return (loss, outputs_s) if return_outputs else loss

# ───────────────────────────  Main  ─────────────────────────────────────

def main():
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    if args.wandb_project:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=os.path.basename(args.output_dir), config=vars(args))

    dtype = torch.float16 if args.fp16 else None
    tok = AutoTokenizer.from_pretrained(args.student_path, use_fast=True)

    student     = load_model(args.student_path, dtype, args.flash_attn).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    teacher_fwd = load_model(args.teacher_fwd_path, dtype, args.flash_attn).to(student.device)
    teacher_rev = load_model(args.teacher_rev_path, dtype, args.flash_attn).to(student.device)

    # data -----------------------------------------------------------------
    ds = load_from_disk(args.dataset_dir)
    if "validation" not in ds:
        ds = ds.train_test_split(test_size=0.20, seed=42)
    prefix = "gec: "
    sep_tok = "<extra_id_0>"

    def build_src(batch):
        x   = maybe_segment(batch["incorrect_text"], args.word_segment, args)
        y0  = maybe_segment(batch["pred"], args.word_segment, args)
        fwd = [f"{prefix}{a} {sep_tok} {b}" for a, b in zip(x, y0)]
        rev = [f"{prefix}{b} {sep_tok} {a}" for a, b in zip(x, y0)]
        batch.update({"src_student": x, "src_fwd": fwd, "src_rev": rev, "labels_text": batch["target"]})
        return batch

    ds = ds.map(build_src, batched=True, batch_size=1024)

    def encode(b):
        stu = tok(b["src_student"], truncation=True, max_length=384)
        fwd = tok(b["src_fwd"],     truncation=True, max_length=384)
        rev = tok(b["src_rev"],     truncation=True, max_length=384)
        with tok.as_target_tokenizer():
            lbl = tok(b["labels_text"], truncation=True, max_length=384)
        return {
            "input_ids": stu["input_ids"], "attention_mask": stu["attention_mask"],
            "fwd_ids": fwd["input_ids"],  "fwd_mask": fwd["attention_mask"],
            "rev_ids": rev["input_ids"],  "rev_mask": rev["attention_mask"],
            "labels": lbl["input_ids"],
        }

    keep_keys = {"input_ids", "attention_mask", "labels", "fwd_ids", "fwd_mask", "rev_ids", "rev_mask"}
    ds = ds.map(encode, batched=True, remove_columns=[c for c in ds["train"].column_names if c not in keep_keys])

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
        report_to=["wandb"] if args.wandb_project else [],
        fp16=args.fp16,
        predict_with_generate=True,
        generation_num_beams=5,
        save_total_limit=3,
        logging_steps=20,
    )

    trainer = DistilTrainer(
        model=student,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"] if "validation" in ds else ds["test"],
        tokenizer=tok,
        data_collator=collator,
        teacher_fwd=teacher_fwd,
        teacher_rev=teacher_rev,
        alpha=args.alpha,
        beta=args.beta,
        tau=args.tau,
        edit_weight=args.edit_weight,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
