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

# ─────────────────────────  VNCoreNLP helper  ────────────────────────────
try:
    from py_vncorenlp import VnCoreNLP       # :contentReference[oaicite:1]{index=1}
except ImportError:
    VnCoreNLP = None

@lru_cache(maxsize=1)
def get_segmenter():
    if VnCoreNLP is None:
        raise RuntimeError("Install py_vncorenlp for word segmentation")
    return VnCoreNLP(save_dir="vncorenlp", annotators=["wseg"])

def maybe_segment(texts, needed: bool):
    if not needed:
        return texts
    seg = get_segmenter()
    return [" ".join(ws) for ws in seg.tokenize(texts)]

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
        outputs_s = model(**inputs)
        loss_ce = outputs_s.loss                                     # NLL
        # ----- teacher logits (no grad) -----------------------------
        with torch.no_grad():
            tf_logits = self.teacher_fwd(**inputs).logits / self.tau
            tr_logits = self.teacher_rev(**inputs).logits / self.tau
        st_logits = outputs_s.logits / self.tau
        st_logp  = F.log_softmax(st_logits,  dim=-1)
        tf_prob  = F.softmax(tf_logits, dim=-1)
        tr_prob  = F.softmax(tr_logits, dim=-1)
        kld_fwd = F.kl_div(st_logp, tf_prob, reduction="batchmean")   # :contentReference[oaicite:3]{index=3}
        kld_rev = F.kl_div(st_logp, tr_prob, reduction="batchmean")
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
    def build_src(batch):
        src = maybe_segment(batch["input"], seg_needed)
        hyp = batch["pred"]                                           # already gen-seg
        batch["source"] = src
        batch["labels_text"] = batch["target"]
        return batch
    ds = ds.map(build_src, batched=True, batch_size=1024)

    def encode(b):
        enc = tok(b["source"], truncation=True, max_length=256)
        with tok.as_target_tokenizer():
            lbl = tok(b["labels_text"], truncation=True, max_length=192)
        enc["labels"] = lbl["input_ids"]
        return enc
    ds = ds.map(encode, batched=True,
                remove_columns=[c for c in ds["train"].column_names
                                if c not in ("labels", "input_ids",
                                             "attention_mask")])

    collator = DataCollatorForSeq2Seq(tok, model=student)

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
        report_to=["wandb"],
        fp16=args.fp16,
        predict_with_generate=True,
        generation_num_beams=5,
        save_total_limit=3,
        logging_steps=500
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
