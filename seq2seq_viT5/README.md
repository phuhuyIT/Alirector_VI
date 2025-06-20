# Vietnamese Grammar Error Correction with ViT5 and LoRA

This directory contains the ViT5-based implementation of the Alirector Vietnamese Grammar Error Correction system with LoRA (Low-Rank Adaptation) for efficient training.

## Key Differences from BARTpho Implementation

1. **LoRA Training**: Uses Parameter-Efficient Fine-tuning to reduce memory requirements
2. **Instruction Prefix**: Adds "gec: " prefix to all source texts
3. **Separator Token**: Uses `<extra_id_0>` instead of `</s>` for T5-style models
4. **BF16 Precision**: Uses bfloat16 for better stability on modern GPUs
5. **Higher Learning Rate**: Uses 5e-5 (vs 3e-5) optimized for LoRA training

## Installation

```bash
cd seq2seq_viT5
pip install -r requirements.txt

# Install VnCoreNLP if using word segmentation
# Requires Java 8+
```

## Training Pipeline

### Stage 1: Base GEC Model Training

Train the base ViT5 model for Vietnamese GEC:

```bash
python -m src.train \
    --dataset_name bmd1905/vi-error-correction-v2 \
    --model_name_or_path VietAI/vit5-base \
    --output_dir runs/vit5-gec \
    --learning_rate 5e-5 \
    --num_train_epochs 6 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --edit_weight 2.0 \
    --rdrop_weight 0.15 \
    --wandb_project vietgec_vit5
```

**Key Parameters:**
- `--lora_rank`: LoRA rank (16 is a good balance)
- `--lora_alpha`: LoRA scaling parameter (32 = 2x rank)
- `--edit_weight`: Apply higher weight to edited tokens (1.5-2.0 recommended)
- `--rdrop_weight`: R-Drop regularization (0.1-0.15 recommended)

### Stage 2: Generate Predictions

Generate draft corrections for alignment training:

```bash
python -m src.predict \
    --model_path runs/vit5-gec \
    --dataset_name bmd1905/vi-error-correction-v2 \
    --split train \
    --output_dir data/stage2/train_pred \
    --batch_size 32 \
    --use_amp \
    --torch_compile
```

### Stage 3: Alignment Model Training

#### Forward Alignment (X → Ŷ → Y)

```bash
python -m src.train_align \
    --dataset_dir data/stage2/train_pred \
    --direction forward \
    --model_name_or_path VietAI/vit5-base \
    --output_dir runs/vit5_align_fwd \
    --learning_rate 5e-5 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --wandb_project vietgec_vit5_align
```

#### Reverse Alignment (Ŷ → X → Y)

```bash
python -m src.train_align \
    --dataset_dir data/stage2/train_pred \
    --direction reverse \
    --model_name_or_path VietAI/vit5-base \
    --output_dir runs/vit5_align_rev \
    --learning_rate 5e-5 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --wandb_project vietgec_vit5_align
```

### Stage 4: Knowledge Distillation

Distill alignment knowledge into the student model:

```bash
python -m src.train_alignment_distill \
    --dataset_dir data/stage2/train_pred \
    --student_path runs/vit5-gec \
    --teacher_fwd_path runs/vit5_align_fwd \
    --teacher_rev_path runs/vit5_align_rev \
    --output_dir runs/vit5_distilled \
    --learning_rate 1e-5 \
    --alpha 0.5 \
    --beta 0.3 \
    --tau 4.0 \
    --edit_weight 1.5 \
    --wandb_project vietgec_vit5_distill
```

**Distillation Parameters:**
- `--alpha`: Forward teacher weight (0.5 recommended)
- `--beta`: Reverse teacher weight (0.3 recommended)
- `--tau`: Temperature for knowledge distillation (4.0 recommended)
- Lower learning rate (1e-5) for stable distillation

## Model Usage

### Inference Example

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Load model
tokenizer = AutoTokenizer.from_pretrained("runs/vit5_distilled")
base_model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")
model = PeftModel.from_pretrained(base_model, "runs/vit5_distilled")

# Correct text
input_text = "Tôi đi học ở trường đại học bách khoa hà nội"
input_with_prefix = f"gec: {input_text}"

inputs = tokenizer(input_with_prefix, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=192, num_beams=4)
corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Original: {input_text}")
print(f"Corrected: {corrected}")
```

### Batch Processing

```python
def correct_batch(texts, model, tokenizer, batch_size=16):
    predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        prefixed_batch = [f"gec: {text}" for text in batch]
        
        inputs = tokenizer(prefixed_batch, return_tensors="pt", 
                          padding=True, truncation=True)
        outputs = model.generate(**inputs, max_new_tokens=192, num_beams=4)
        batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        predictions.extend(batch_predictions)
    
    return predictions
```

## Evaluation

Evaluate model performance:

```bash
python -m src.evaluate_gec \
    --model_path runs/vit5_distilled \
    --dataset_name bmd1905/vi-error-correction-v2 \
    --split test \
    --output_file results/vit5_evaluation.json \
    --compute_all_metrics \
    --batch_size 32
```

## Hardware Requirements

### Minimum Requirements
- **GPU**: 8GB VRAM (RTX 3070, V100)
- **RAM**: 16GB system RAM
- **Storage**: 10GB for models and data

### Recommended Requirements  
- **GPU**: 16GB+ VRAM (RTX 4080, A100)
- **RAM**: 32GB system RAM
- **Storage**: 50GB for full pipeline

### LoRA Benefits
- **Memory Reduction**: ~75% less VRAM usage
- **Faster Training**: 2-3x faster convergence
- **Easy Merging**: Combine adapters easily

## Performance Optimization

### Training Optimizations
```bash
# Use automatic mixed precision
--use_amp

# Enable torch.compile (PyTorch 2.0+)
--torch_compile

# Dynamic batch sizing
--dynamic_batching

# Gradient checkpointing for memory
--gradient_checkpointing
```

### Inference Optimizations
```bash
# Use FP16 for faster inference
--fp16

# Increase batch size if memory allows
--batch_size 64

# Use beam search for better quality
--num_beams 4
```

## Expected Results

### Stage 1 (Base Model)
- **BLEU**: ~45-50
- **Exact Match**: ~25-30%
- **Training Time**: 2-3 hours on single V100

### Stage 4 (After Distillation)
- **BLEU**: ~52-58
- **Exact Match**: ~35-40%
- **Training Time**: 1-2 hours additional

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   --per_device_train_batch_size 2
   --gradient_accumulation_steps 8
   
   # Lower LoRA rank
   --lora_rank 8
   ```

2. **Slow Training**
   ```bash
   # Enable optimizations
   --use_amp --torch_compile --dynamic_batching
   ```

3. **Poor Quality**
   ```bash
   # Increase edit weighting
   --edit_weight 2.5
   
   # Add R-Drop regularization
   --rdrop_weight 0.2
   ```

### Model Loading Issues
```python
# If adapter loading fails, load base model first
base_model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")
model = PeftModel.from_pretrained(base_model, "path/to/adapters")
```

## File Structure

```
seq2seq_viT5/
├── src/
│   ├── train.py                    # Stage 1: Base model training
│   ├── predict.py                  # Stage 2: Generate predictions  
│   ├── train_align.py              # Stage 3: Alignment training
│   ├── train_alignment_distill.py  # Stage 4: Knowledge distillation
│   └── evaluate_gec.py             # Model evaluation
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## Citation

If you use this implementation, please cite:

```bibtex
@inproceedings{alirector2023,
  title={Alirector: Vietnamese Grammar Error Correction with Multi-Stage Training},
  author={Your Name},
  booktitle={Proceedings of Conference},
  year={2023}
}
```
