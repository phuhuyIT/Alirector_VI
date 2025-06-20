# ------------------------------------------------------------
#  Unified ViT5 Alirector Training Pipeline
#  -----------------------------------------------------------
#  This single file exposes **four** CLI entry‑points that mirror the
#  original Chinese Alirector stages but adapted for ViT5‑base on
#  Vietnamese GEC.  Each stage is invoked via the "task" flag:
#     python vit5_alirector_pipeline.py --task stage1_train [...]
#     python vit5_alirector_pipeline.py --task stage2_predict [...]
#     python vit5_alirector_pipeline.py --task stage2_train_align [...]
#     python vit5_alirector_pipeline.py --task stage3_distill [...]
#
#  Key changes vs. BARTpho version:
#  ✦ Prefix "gec: " automatically added to *all* sources.
#  ✦ Separator token between incorrect / draft = "<extra_id_0>".
#  ✦ Default max_source_len bumped to 384 (stage2 + stage3).
#  ✦ ViT5 defaults: AdaFactor, LR 5e‑5, bf16 True, decoder_start_token_id=0.
#  ✦ Optional edit‑weighted CE implemented (γ_edit=3.0).
# ------------------------------------------------------------

import argparse, os, json, math, itertools, random
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn.functional as F
from torch import nn

from datasets import load_from_disk, load_metric, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    set_seed,
    Adafactor,
)


# -----------------------------
# Helper: edit‑weighted loss
# -----------------------------
class EditWeightedLossTrainer(Seq2SeqTrainer):
    """Override compute_loss to weight edited tokens heavier."""

    def __init__(self, gamma_edit: float = 3.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma_edit = gamma_edit

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        logits = outputs.logits  # (B, L, V)
        loss_mask = labels != -100
        # token‑level CE
        ce = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
        # mask out padding (‑100) first
        ce = ce * loss_mask
        # find tokens that truly changed (gold != source)
        changed = (labels != inputs["input_ids"]) & loss_mask
        weights = torch.where(changed, torch.full_like(ce, self.gamma_edit), torch.ones_like(ce))
        loss = (ce * weights).sum() / weights.sum()
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------
# Argument templates for each sub‑task
# ---------------------------------------
COMMON_ARGS = dict(
    model_name_or_path="VietAI/vit5-base",
    prefix="gec: ",
    max_source_len=256,
    max_target_len=256,
    learning_rate=5e-5,
    gamma_edit=3.0,
)


# utility: build HF Seq2SeqTrainingArguments quickly

def build_training_args(output_dir: str, args: argparse.Namespace, **extra):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        evaluation_strategy="steps" if args.eval_steps else "no",
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        fp16=not args.bf16,
        bf16=args.bf16,
        logging_steps=50,
        gradient_accumulation_steps=args.grad_acc,
        label_smoothing_factor=args.label_smoothing,
        predict_with_generate=True,
        generation_max_length=args.max_target_len,
        generation_num_beams=args.beam_size,
        decoder_start_token_id=0,
        **extra,
    )
    return hf_args


# ---------------------------------------
# Stage‑1: baseline corrector training
# ---------------------------------------

def stage1_train(args: argparse.Namespace):
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    ds = load_from_disk(args.dataset_dir)
    # Assume columns: incorrect_text, correct_text
    def preprocess(batch):
        src = [args.prefix + x for x in batch["incorrect_text"]]
        tgt = batch["correct_text"]
        model_inputs = tok(
            src,
            truncation=True,
            max_length=args.max_source_len,
            padding="max_length",
        )
        with tok.as_target_tokenizer():
            labels = tok(
                tgt,
                truncation=True,
                max_length=args.max_target_len,
                padding="max_length",
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    ds = ds.map(preprocess, batched=True, desc="Tokenising")

    data_collator = DataCollatorForSeq2Seq(tok, model=model, pad_to_multiple_of=8)

    training_args = build_training_args("stage1_ckpt", args)

    trainer_cls = EditWeightedLossTrainer if args.use_edit_weight else Seq2SeqTrainer

    optim = Adafactor(model.parameters(), lr=args.learning_rate, scale_parameter=False, relative_step=False)

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        tokenizer=tok,
        data_collator=data_collator,
        optimizers=(optim, None),
        gamma_edit=args.gamma_edit,
    )

    trainer.train()
    trainer.save_model()
    tok.save_pretrained("stage1_ckpt")


# ---------------------------------------
# Stage‑2‑A: prediction with Stage‑1 model
# ---------------------------------------

def stage2_predict(args: argparse.Namespace):
    tok = AutoTokenizer.from_pretrained(args.stage1_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.stage1_path).to(args.device)
    model.eval()

    ds = load_from_disk(args.dataset_dir)

    def generate(batch):
        src = [args.prefix + x for x in batch["incorrect_text"]]
        inputs = tok(
            src,
            truncation=True,
            max_length=args.max_source_len,
            padding=True,
            return_tensors="pt",
        ).to(args.device)
        with torch.cuda.amp.autocast(enabled=args.bf16):
            outs = model.generate(
                **inputs,
                num_beams=args.beam_size,
                length_penalty=0.8,
                early_stopping=True,
                max_length=args.max_target_len,
            )
        batch["pred"] = tok.batch_decode(outs, skip_special_tokens=True)
        return batch

    ds = ds.map(generate, batched=True, batch_size=args.gen_bs, desc="Generating", remove_columns=[])
    out_dir = Path(args.pred_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))


# ---------------------------------------
# Stage‑2‑B: train alignment teachers (A & A‑rev)
# ---------------------------------------

def stage2_train_align(args: argparse.Namespace):
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    sep_tok = "<extra_id_0>"

    ds = load_from_disk(args.pred_dir)

    # keep only rows where pred != correct
    ds = ds.filter(lambda ex: ex["pred"].strip() != ex["correct_text"].strip(), num_proc=4)

    def build_pair(batch, reverse=False):
        src = []
        tgt = []
        for inc, draft, cor in zip(batch["incorrect_text"], batch["pred"], batch["correct_text"]):
            if not reverse:
                src.append(f"{args.prefix}{inc} {sep_tok} {draft}")
                tgt.append(cor)
            else:
                src.append(f"{args.prefix}{cor} {sep_tok} {draft}")
                tgt.append(inc)
        res = tok(src, truncation=True, max_length=args.max_source_len, padding="max_length")
        with tok.as_target_tokenizer():
            lab = tok(tgt, truncation=True, max_length=args.max_target_len, padding="max_length")
        res["labels"] = lab["input_ids"]
        return res

    # split 80/20 once, no stratification needed
    ds_train = ds.shuffle(seed=42).select(range(int(len(ds)*0.8)))
    ds_val   = ds.select(range(int(len(ds)*0.8), len(ds)))

    data_collator = DataCollatorForSeq2Seq(tok, pad_to_multiple_of=8)

    # Train A (inc→cor)
    train_ds_a = ds_train.map(build_pair, batched=True)
    val_ds_a   = ds_val.map(build_pair, batched=True)
    model_a = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    training_args_a = build_training_args("stage2_teacher_A", args)
    trainer_a = Seq2SeqTrainer(
        model=model_a,
        args=training_args_a,
        train_dataset=train_ds_a,
        eval_dataset=val_ds_a,
        tokenizer=tok,
        data_collator=data_collator,
    )
    trainer_a.train()
    trainer_a.save_model()

    # Train A‑rev (cor→inc)
    train_ds_b = ds_train.map(lambda batch: build_pair(batch, reverse=True), batched=True)
    val_ds_b   = ds_val.map(lambda batch: build_pair(batch, reverse=True), batched=True)
    model_b = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    training_args_b = build_training_args("stage2_teacher_Rev", args)
    trainer_b = Seq2SeqTrainer(
        model=model_b,
        args=training_args_b,
        train_dataset=train_ds_b,
        eval_dataset=val_ds_b,
        tokenizer=tok,
        data_collator=data_collator,
    )
    trainer_b.train()
    trainer_b.save_model()


# ---------------------------------------
# Stage‑3: distil teachers into student
# ---------------------------------------

def stage3_distill(args: argparse.Namespace):
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    sep_tok = "<extra_id_0>"

    # Load datasets with pred column again
    ds = load_from_disk(args.pred_dir)
    ds = ds.filter(lambda ex: ex["pred"].strip() != ex["correct_text"].strip(), num_proc=4)

    teacher_a = AutoModelForSeq2SeqLM.from_pretrained(args.teacher_A_path).eval().to(args.device)
    teacher_b = AutoModelForSeq2SeqLM.from_pretrained(args.teacher_B_path).eval().to(args.device)

    student = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    def build_input(batch):
        src = [f"{args.prefix}{x} {sep_tok} {y}" for x, y in zip(batch["incorrect_text"], batch["pred"])]
        tgt = batch["correct_text"]
        model_inputs = tok(src, truncation=True, max_length=args.max_source_len, padding="max_length")
        with tok.as_target_tokenizer():
            lab = tok(tgt, truncation=True, max_length=args.max_target_len, padding="max_length")
        model_inputs["labels"] = lab["input_ids"]
        return model_inputs

    ds = ds.map(build_input, batched=True, desc="preprocess distill data")

    data_collator = DataCollatorForSeq2Seq(tok, pad_to_multiple_of=8)

    training_args = build_training_args("stage3_student", args)

    class DistilTrainer(EditWeightedLossTrainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            # Student forward
            outputs = model(**inputs, labels=labels)
            student_logits = outputs.logits
            # Teacher ensembled soft targets
            with torch.no_grad():
                t_out_a = teacher_a(**inputs)
                t_out_b = teacher_b(**inputs)
                teacher_logits = (t_out_a.logits + t_out_b.logits) / 2
            loss_ce = super().compute_loss(model, {**inputs, "labels": labels})
            # KL distillation
            kl = F.kl_div(
                F.log_softmax(student_logits / args.kd_temp, dim=-1),
                F.softmax(teacher_logits / args.kd_temp, dim=-1),
                reduction="batchmean",
            ) * (args.kd_temp ** 2)
            loss = args.kd_alpha * kl + (1 - args.kd_alpha) * loss_ce
            return (loss, outputs) if return_outputs else loss

    optim = Adafactor(student.parameters(), lr=args.learning_rate, scale_parameter=False, relative_step=False)

    trainer = DistilTrainer(
        model=student,
        args=training_args,
        train_dataset=ds,
        tokenizer=tok,
        data_collator=data_collator,
        optimizers=(optim, None),
        gamma_edit=args.gamma_edit,
    )

    trainer.train()
    trainer.save_model()


# ---------------------------------------
#  CLI parsing
# ---------------------------------------

def main():
    parser = argparse.ArgumentParser("ViT5 Alirector")
    parser.add_argument("--task", required=True, choices=[
        "stage1_train", "stage2_predict", "stage2_train_align", "stage3_distill",
    ])
    # shared params
    parser.add_argument("--dataset_dir")
    parser.add_argument("--model_name_or_path", default=COMMON_ARGS["model_name_or_path"])
    parser.add_argument("--prefix", default=COMMON_ARGS["prefix"])
    parser.add_argument("--max_source_len", type=int, default=COMMON_ARGS["max_source_len"])
    parser.add_argument("--max_target_len", type=int, default=COMMON_ARGS["max_target_len"])
    parser.add_argument("--learning_rate", type=float, default=COMMON_ARGS["learning_rate"])
    parser.add_argument("--gamma_edit", type=float, default=COMMON_ARGS["gamma_edit"])
    parser.add_argument("--train_bs", type=int, default=4)
    parser.add_argument("--eval_bs", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--grad_acc", type=int, default=1)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use_edit_weight", action="store_true")

    # stage‑specific
    parser.add_argument("--stage1_path", default="stage1_ckpt")
    parser.add_argument("--pred_dir", default="stage2_pred")
    parser.add_argument("--teacher_A_path", default="stage2_teacher_A")
    parser.add_argument("--teacher_B_path", default="stage2_teacher_Rev")
    parser.add_argument("--gen_bs", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kd_alpha", type=float, default=0.4)
    parser.add_argument("--kd_temp", type=float, default=2.0)

    args = parser.parse_args()

    set_seed(42)

    task_fn = {
        "stage1_train": stage1_train,
        "stage2_predict": stage2_predict,
        "stage2_train_align": stage2_train_align,
        "stage3_distill": stage3_distill,
    }[args.task]

    task_fn(args)


if __name__ == "__main__":
    main()
