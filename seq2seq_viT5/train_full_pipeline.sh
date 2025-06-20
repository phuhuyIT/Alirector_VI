#!/bin/bash

# Vietnamese GEC Training Pipeline with ViT5 and LoRA
# Make sure you have set up the environment and installed requirements

set -e  # Exit on any error

echo "Starting Vietnamese GEC Training Pipeline with ViT5..."

# Configuration
DATASET_NAME="bmd1905/vi-error-correction-v2"
MODEL_NAME="VietAI/vit5-base"
WANDB_PROJECT="vietgec_vit5"
WANDB_ENTITY="phuhuy02003-university-of-transport-and-communications"

# Directories
BASE_DIR="runs"
STAGE1_DIR="${BASE_DIR}/vit5-gec"
STAGE2_DIR="data/stage2/train_pred" 
ALIGN_FWD_DIR="${BASE_DIR}/vit5_align_fwd"
ALIGN_REV_DIR="${BASE_DIR}/vit5_align_rev"
DISTILL_DIR="${BASE_DIR}/vit5_distilled"

# Create directories
mkdir -p data/stage2 results

echo "=== Stage 1: Base GEC Model Training ==="
python -m src.train \
    --dataset_name ${DATASET_NAME} \
    --model_name_or_path ${MODEL_NAME} \
    --output_dir ${STAGE1_DIR} \
    --learning_rate 5e-5 \
    --num_train_epochs 6 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --edit_weight 2.0 \
    --rdrop_weight 0.15 \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_entity ${WANDB_ENTITY}

echo "=== Stage 2: Generate Predictions ==="
python -m src.predict \
    --model_path ${STAGE1_DIR} \
    --dataset_name ${DATASET_NAME} \
    --split train \
    --output_dir ${STAGE2_DIR} \
    --batch_size 32 \
    --use_amp \
    --torch_compile

echo "=== Stage 3a: Forward Alignment Training ==="
python -m src.train_align \
    --dataset_dir ${STAGE2_DIR} \
    --direction forward \
    --model_name_or_path ${MODEL_NAME} \
    --output_dir ${ALIGN_FWD_DIR} \
    --learning_rate 5e-5 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --wandb_project ${WANDB_PROJECT}_align \
    --wandb_entity ${WANDB_ENTITY}

echo "=== Stage 3b: Reverse Alignment Training ==="
python -m src.train_align \
    --dataset_dir ${STAGE2_DIR} \
    --direction reverse \
    --model_name_or_path ${MODEL_NAME} \
    --output_dir ${ALIGN_REV_DIR} \
    --learning_rate 5e-5 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --wandb_project ${WANDB_PROJECT}_align \
    --wandb_entity ${WANDB_ENTITY}

echo "=== Stage 4: Knowledge Distillation ==="
python -m src.train_alignment_distill \
    --dataset_dir ${STAGE2_DIR} \
    --student_path ${STAGE1_DIR} \
    --teacher_fwd_path ${ALIGN_FWD_DIR} \
    --teacher_rev_path ${ALIGN_REV_DIR} \
    --output_dir ${DISTILL_DIR} \
    --learning_rate 1e-5 \
    --alpha 0.5 \
    --beta 0.3 \
    --tau 4.0 \
    --edit_weight 1.5 \
    --wandb_project ${WANDB_PROJECT}_distill \
    --wandb_entity ${WANDB_ENTITY}

echo "=== Evaluation ==="
python -m src.evaluate_gec \
    --model_path ${DISTILL_DIR} \
    --dataset_name ${DATASET_NAME} \
    --split test \
    --output_file results/vit5_evaluation.json \
    --compute_all_metrics \
    --batch_size 32

echo "=== Training Pipeline Complete! ==="
echo "Final model saved to: ${DISTILL_DIR}"
echo "Evaluation results saved to: results/vit5_evaluation.json"

# Print model sizes
echo ""
echo "=== Model Sizes ==="
du -sh ${STAGE1_DIR} 2>/dev/null || echo "Stage 1 model: N/A"
du -sh ${ALIGN_FWD_DIR} 2>/dev/null || echo "Forward alignment: N/A"  
du -sh ${ALIGN_REV_DIR} 2>/dev/null || echo "Reverse alignment: N/A"
du -sh ${DISTILL_DIR} 2>/dev/null || echo "Final distilled model: N/A"
