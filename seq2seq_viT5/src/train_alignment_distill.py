#!/usr/bin/env python
"""
Stage-3 Alirector — distil alignment teachers (A, A-rev) into the student
corrector C for ViT5 with LoRA.

Example:
python -m src.train_alignment_distill \
        --dataset_dir data/stage2/train_pred \
        --student_path runs/vit5-gec \
        --teacher_fwd_path runs/vit5_align_fwd \
        --teacher_rev_path runs/vit5_align_rev \
        --output_dir runs/vit5_distilled \
        --wandb_project vietgec
"""
import os, argparse, wandb, torch, torch.nn.functional as F
from functools import lru_cache
from datasets import load_from_disk
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
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
            if not ids_list:
                continue
            
            max_len = max(t.size(0) for t in ids_list)
            padded_ids, padded_masks = [], []
            for ids_t, mask_t in zip(ids_list, mask_list):
                pad_len = max_len - ids_t.size(0)
                ids_padded  = pad(ids_t,  (0, pad_len), value=self.tokenizer.pad_token_id)
                mask_padded = pad(mask_t, (0, pad_len), value=0)
                padded_ids.append(ids_padded)
                padded_masks.append(mask_padded)
            
            base[ids_key]  = torch.stack(padded_ids)
            base[mask_key] = torch.stack(padded_masks)

        return base

# ─────────────────────────  VNCoreNLP helper  ────────────────────────────
try:
    from py_vncorenlp import VnCoreNLP       
except ImportError:
    VnCoreNLP = None

@lru_cache(maxsize=1)
def get_segmenter(word_segment_save_dir: str):
    if VnCoreNLP is None:
        raise RuntimeError("py_vncorenlp not installed")
    return VnCoreNLP(save_dir=word_segment_save_dir, annotators=["wseg"])

def maybe_segment(texts, needed: bool, args):
    if not needed:
        return texts
    seg = get_segmenter(args.word_segment_save_dir)
    results = []
    for text in texts:
        segmented_words = seg.word_segment(text)[0]
        results.append(" ".join(segmented_words))  # Fixed: add spaces
    return results

# ────────────────────────────  Args  ─────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--student_path", type=str, required=True, 
                   help="Path to stage-1 ViT5 model")
    p.add_argument("--teacher_fwd_path", type=str, required=True)
    p.add_argument("--teacher_rev_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    
    # LoRA parameters for student
    p.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha") 
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # distillation hyperparams
    p.add_argument("--alpha", type=float, default=0.5, help="Forward KLD weight")
    p.add_argument("--beta", type=float, default=0.3, help="Reverse KLD weight") 
    p.add_argument("--tau", type=float, default=4.0, help="Temperature for KLD")
    p.add_argument("--edit_weight", type=float, default=1.0, help="Edit-weighted CE")
    
    # training
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-5)  # Lower LR for distillation
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_train_epochs", type=int, default=6)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    
    # optional
    p.add_argument("--word_segment", action="store_true")
    p.add_argument("--word_segment_save_dir", type=str, default="./vncorenlp")
    p.add_argument("--wandb_project", type=str, default="Vi_Alirector_vit5_distill")
    p.add_argument("--wandb_entity", type=str, default="phuhuy02003-university-of-transport-and-communications")
    p.add_argument("--wandb_api_key", type=str, default=None)
    return p

# ────────────────────────────  Trainer  ──────────────────────────────────
class DistilTrainer(Seq2SeqTrainer):
    def __init__(self, *args, teacher_fwd=None, teacher_rev=None,
                 alpha=1.0, beta=0.5, tau=1.0, edit_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_fwd = teacher_fwd.eval()
        self.teacher_rev = teacher_rev.eval()
        for p in self.teacher_fwd.parameters():
            p.requires_grad_(False)
        for p in self.teacher_rev.parameters():
            p.requires_grad_(False)
        self.alpha, self.beta, self.tau, self.edit_weight = alpha, beta, tau, edit_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract teacher inputs
        fwd_ids  = inputs.pop("fwd_ids", None)   
        fwd_mask = inputs.pop("fwd_mask", None)
        rev_ids  = inputs.pop("rev_ids", None)   
        rev_mask = inputs.pop("rev_mask", None)

        # Student forward pass
        student_out = model(**inputs)
        student_logits = student_out.logits
        
        # Standard cross-entropy loss with edit weighting
        labels = inputs.get("labels")
        if labels is not None:
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # Apply edit weighting
            if self.edit_weight != 1.0:
                mask = (shift_labels != -100)
                ce_losses = F.cross_entropy(shift_logits, shift_labels, reduction='none')
                weighted_losses = torch.where(mask, 
                                             ce_losses * self.edit_weight, 
                                             ce_losses)
                loss_ce = weighted_losses[mask].mean()
            else:
                loss_ce = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        else:
            loss_ce = torch.tensor(0.0, device=student_logits.device)

        # Teacher forward passes
        kld_fwd = torch.tensor(0.0, device=student_logits.device)
        kld_rev = torch.tensor(0.0, device=student_logits.device)
        
        if fwd_ids is not None and fwd_mask is not None:
            with torch.no_grad():
                teacher_fwd_out = self.teacher_fwd(input_ids=fwd_ids, 
                                                   attention_mask=fwd_mask)
                teacher_fwd_logits = teacher_fwd_out.logits
            
            # Align dimensions and compute KL divergence
            min_len = min(student_logits.size(1), teacher_fwd_logits.size(1))
            student_logits_fwd = student_logits[:, :min_len, :]
            teacher_fwd_logits = teacher_fwd_logits[:, :min_len, :]
            
            # Apply temperature scaling
            student_probs = F.log_softmax(student_logits_fwd / self.tau, dim=-1)
            teacher_probs = F.softmax(teacher_fwd_logits / self.tau, dim=-1)
            
            # Create mask for valid positions
            if labels is not None:
                labels_fwd = labels[:, :min_len]
                mask_fwd = (labels_fwd != -100)
                
                # Compute KL divergence only on valid positions
                kld_fwd = F.kl_div(student_probs, teacher_probs, reduction='none')
                kld_fwd = kld_fwd.sum(dim=-1)  # Sum over vocab
                kld_fwd = kld_fwd[mask_fwd].mean()  # Average over valid positions
                kld_fwd *= self.tau ** 2  # Restore gradient scale (classic KD fix)
        
        if rev_ids is not None and rev_mask is not None:
            with torch.no_grad():
                teacher_rev_out = self.teacher_rev(input_ids=rev_ids, 
                                                   attention_mask=rev_mask)
                teacher_rev_logits = teacher_rev_out.logits
            
            # Align dimensions and compute KL divergence
            min_len = min(student_logits.size(1), teacher_rev_logits.size(1))
            student_logits_rev = student_logits[:, :min_len, :]
            teacher_rev_logits = teacher_rev_logits[:, :min_len, :]
            
            # Apply temperature scaling
            student_probs = F.log_softmax(student_logits_rev / self.tau, dim=-1)
            teacher_probs = F.softmax(teacher_rev_logits / self.tau, dim=-1)
            
            # Create mask for valid positions
            if labels is not None:
                labels_rev = labels[:, :min_len]
                mask_rev = (labels_rev != -100)
                
                # Compute KL divergence only on valid positions
                kld_rev = F.kl_div(student_probs, teacher_probs, reduction='none')
                kld_rev = kld_rev.sum(dim=-1)  # Sum over vocab
                kld_rev = kld_rev[mask_rev].mean()  # Average over valid positions
                kld_rev *= self.tau ** 2  # Restore gradient scale (classic KD fix)

        # Total loss
        loss = loss_ce + self.alpha * kld_fwd + self.beta * kld_rev
        
        # Log components
        self.log({
            "train/loss_ce": loss_ce.item(),
            "train/loss_kld_fwd": kld_fwd.item(),
            "train/loss_kld_rev": kld_rev.item(),
            "train/loss_total": loss.item()
        })
        
        return (loss, student_out) if return_outputs else loss

# ─────────────────────────────  Main  ────────────────────────────────────
def main():
    import time
    args = build_parser().parse_args()
    
    # wandb
    if args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     config=vars(args))
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_dir}...")
    dataset = load_from_disk(args.dataset_dir)
    
    # Load tokenizer (should be same for all models)
    tokenizer = AutoTokenizer.from_pretrained(args.student_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load student model and add LoRA
    print(f"Loading student model from {args.student_path}...")
    if os.path.exists(os.path.join(args.student_path, "adapter_config.json")):
        # Student already has LoRA adapters
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.student_path.replace("/checkpoint-*", "").split("/")[0] if "checkpoint" in args.student_path else "VietAI/vit5-base",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        student_model = PeftModel.from_pretrained(base_model, args.student_path)
    else:
        # Add LoRA to pretrained model
        student_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.student_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q", "v", "k", "o", "wi_0", "wi_1", "wo"]
        )
        student_model = get_peft_model(student_model, lora_config)
    
    # Load teacher models
    print(f"Loading forward teacher from {args.teacher_fwd_path}...")
    if os.path.exists(os.path.join(args.teacher_fwd_path, "adapter_config.json")):
        base_model_fwd = AutoModelForSeq2SeqLM.from_pretrained(
            "VietAI/vit5-base",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        teacher_fwd = PeftModel.from_pretrained(base_model_fwd, args.teacher_fwd_path)
    else:
        teacher_fwd = AutoModelForSeq2SeqLM.from_pretrained(
            args.teacher_fwd_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    print(f"Loading reverse teacher from {args.teacher_rev_path}...")
    if os.path.exists(os.path.join(args.teacher_rev_path, "adapter_config.json")):
        base_model_rev = AutoModelForSeq2SeqLM.from_pretrained(
            "VietAI/vit5-base", 
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        teacher_rev = PeftModel.from_pretrained(base_model_rev, args.teacher_rev_path)
    else:
        teacher_rev = AutoModelForSeq2SeqLM.from_pretrained(
            args.teacher_rev_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    print("Models loaded successfully!")
    student_model.print_trainable_parameters()
    
    # Check word segmentation
    seg_needed = args.word_segment
    print(f"Word segmentation: {'enabled' if seg_needed else 'disabled'}")
    
    # Preprocessing function
    def preprocess_fn(batch):
        inputs = batch['input']
        preds = batch['pred']
        targets = batch['target']
        
        # Apply segmentation if needed
        if seg_needed:
            inputs = maybe_segment(inputs, True, args)
            preds = maybe_segment(preds, True, args)
            targets = maybe_segment(targets, True, args)
        
        # Student data: gec: X -> Y
        student_sources = [f"gec: {inp}" for inp in inputs]
        
        # Teacher forward data: gec: X <extra_id_0> Ŷ
        fwd_sources = [f"gec: {inp} <extra_id_0> {pred}" for inp, pred in zip(inputs, preds)]
        
        # Teacher reverse data: gec: Ŷ <extra_id_0> X  
        rev_sources = [f"gec: {pred} <extra_id_0> {inp}" for inp, pred in zip(inputs, preds)]
        
        # Tokenize student inputs
        student_inputs = tokenizer(
            student_sources,
            max_length=384,
            truncation=True,
            padding=False
        )
        
        # Tokenize student targets
        with tokenizer.as_target_tokenizer():
            student_labels = tokenizer(
                targets,
                max_length=384,
                truncation=True,
                padding=False
            )
        
        # Tokenize teacher inputs
        fwd_inputs = tokenizer(fwd_sources, max_length=384, truncation=True, padding=False)
        rev_inputs = tokenizer(rev_sources, max_length=384, truncation=True, padding=False)
        
        # Prepare labels for student (replace pad_token_id with -100)
        student_labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in student_labels["input_ids"]
        ]
        
        # Combine everything
        result = {
            "input_ids": student_inputs["input_ids"],
            "attention_mask": student_inputs["attention_mask"],
            "labels": student_labels["input_ids"],
            "fwd_ids": fwd_inputs["input_ids"],
            "fwd_mask": fwd_inputs["attention_mask"],
            "rev_ids": rev_inputs["input_ids"],
            "rev_mask": rev_inputs["attention_mask"],
        }
        
        return result
    
    # Split dataset if needed
    if hasattr(dataset, 'train'):
        train_ds = dataset.train
        eval_ds = dataset.get('validation', dataset.get('test', None))
        if eval_ds is None:
            split_ds = train_ds.train_test_split(test_size=0.1, seed=42)
            train_ds, eval_ds = split_ds["train"], split_ds["test"]
    else:
        split_ds = dataset.train_test_split(test_size=0.1, seed=42)
        train_ds, eval_ds = split_ds["train"], split_ds["test"]
    
    print(f"Train size: {len(train_ds)}, Eval size: {len(eval_ds)}")
    
    # Apply preprocessing
    train_ds = train_ds.map(preprocess_fn, batched=True, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(preprocess_fn, batched=True, remove_columns=eval_ds.column_names)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=50,
        eval_steps=500,
        save_steps=1000,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name=f"vit5-distill-{time.strftime('%Y%m%d-%H%M%S')}",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    
    # Data collator
    data_collator = DataCollatorWithTeachers(tokenizer=tokenizer, model=student_model)
    
    # Trainer
    trainer = DistilTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        teacher_fwd=teacher_fwd,
        teacher_rev=teacher_rev,
        alpha=args.alpha,
        beta=args.beta,
        tau=args.tau,
        edit_weight=args.edit_weight,
    )
    
    # Train
    print("Starting distillation training...")
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    wandb.finish()
    print(f"Distillation training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
