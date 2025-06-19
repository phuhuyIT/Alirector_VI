#!/usr/bin/env python
"""
Generate draft corrections Ŷ for every X in a dataset (Stage-2 Alirector).
OPTIMIZED VERSION for faster GPU inference on L4/A100 GPUs.

Example (Colab):

    python -m src.predict \
        --model_path runs/bartpho-syl-gec \
        --dataset_name bmd1905/vi-error-correction-v2 \
        --split train \
        --output_dir data/stage2/train_pred \
        --fp16 \
        --batch_size 32 \
        --use_amp \
        --torch_compile

    # bartpho-word, with auto word-segmentation
    python -m src.predict \
        --model_path runs/bartpho-word-gec \
        --dataset_name bmd1905/vi-error-correction-v2 \
        --split train \
        --output_dir data/stage2/train_pred \
        --fp16 \
        --batch_size 32 \
        --use_amp \
        --torch_compile
"""
import argparse, os, torch, json, time
import wandb
from functools import lru_cache
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gc
from torch.cuda.amp import autocast
import logging

# Set up logging for performance monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from py_vncorenlp import VnCoreNLP   # optional
except ImportError:
    VnCoreNLP = None


# --------------------------------------------------------------------------
# GPU Optimization Settings
# --------------------------------------------------------------------------
def optimize_gpu_settings():
    """Optimize GPU settings for better performance"""
    if torch.cuda.is_available():
        # Enable TensorFloat-32 for faster training on Ampere GPUs (L4, A100)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN benchmarking for consistent input sizes
        torch.backends.cudnn.benchmark = True
        
        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        logger.info(f"GPU optimization enabled. Device: {torch.cuda.get_device_name()}")
        logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def clear_gpu_cache():
    """Clear GPU cache to free memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


# --------------------------------------------------------------------------
# Dynamic Batching for Better GPU Utilization
# --------------------------------------------------------------------------
def get_optimal_batch_size(tokenizer, model, device, base_batch_size=16, max_length=192):
    """Dynamically determine optimal batch size based on available GPU memory"""
    if not torch.cuda.is_available():
        return base_batch_size
    
    # Test with dummy data to find optimal batch size
    dummy_text = ["This is a test sentence for batch size optimization."] * base_batch_size
    
    for batch_size in [base_batch_size * 2, base_batch_size * 3, base_batch_size * 4]:
        try:
            test_batch = dummy_text[:batch_size]
            with torch.no_grad():
                inputs = tokenizer(test_batch, 
                                 padding=True, 
                                 truncation=True,
                                 max_length=max_length,
                                 return_tensors="pt").to(device)
                
                # Test generation
                _ = model.generate(**inputs, max_length=max_length, num_beams=1, do_sample=False)
                clear_gpu_cache()
                logger.info(f"Batch size {batch_size} works successfully")
                
        except torch.cuda.OutOfMemoryError:
            clear_gpu_cache()
            logger.info(f"Batch size {batch_size} causes OOM, using {batch_size // 2}")
            return max(batch_size // 2, base_batch_size)
    
    return base_batch_size * 4  # If all tests pass, use 4x base batch size


# --------------------------------------------------------------------------
# Word-segmentation helpers (only needed for bartpho-word checkpoints)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_segmenter(word_segment_save_dir: str):
    if VnCoreNLP is None:
        raise RuntimeError("py_vncorenlp not installed — "
                           "pip install py_vncorenlp && ensure Java ≥8")
    return VnCoreNLP(save_dir=word_segment_save_dir, annotators=["wseg"])


def segment_sentences(batch, args):
    seg = get_segmenter(args.word_segment_save_dir)
    # word_segment expects a single string, returns List[List[str]]
    # Process each sentence individually
    segmented_inputs = []
    for text in batch["input"]:
        segmented_words = seg.word_segment(text)[0]  # Take first (and only) result
        segmented_inputs.append("".join(segmented_words))
    
    batch["input"] = segmented_inputs
    return batch


# --------------------------------------------------------------------------
# CLI with additional optimization options
# --------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="Path or HF id of the fine-tuned Stage-1 model")
    p.add_argument("--dataset_name", type=str,
                   help="HF dataset id or path (if previously saved)")
    p.add_argument("--dataset_path", type=str,
                   help="Local path created by save_to_disk(); "
                        "takes precedence over --dataset_name")
    p.add_argument("--split", type=str, default="train",
                   help="Dataset split to run prediction on")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--beam_size", type=int, default=5)
    p.add_argument("--max_len", type=int, default=192)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--wandb_project", type=str, default="Vi_Alirector_syllable_base")
    p.add_argument("--wandb_entity", type=str, default="phuhuy02003-university-of-transport-and-communications")
    p.add_argument("--wandb_api_key", type=str, default=None)
    p.add_argument("--word_segment", type=bool, default=False,
                   help="Force VNCoreNLP segmentation regardless of checkpoint")
    p.add_argument("--fp16", action="store_true",
                   help="Generate in FP16 (recommended on L4/A100)")
    p.add_argument("--use_amp", action="store_true",
                   help="Use Automatic Mixed Precision for faster inference")
    p.add_argument("--torch_compile", action="store_true",
                   help="Use torch.compile for faster inference (PyTorch 2.0+)")
    p.add_argument("--dynamic_batching", action="store_true", default=True,
                   help="Automatically determine optimal batch size")
    p.add_argument("--num_workers", type=int, default=4,
                   help="Number of workers for data loading")
    p.add_argument("--word_segment_save_dir", type=str, default="")
    p.add_argument("--subset_ratio", type=float, default=1.0,
                   help="Subset ratio for dataset")
    return p


# --------------------------------------------------------------------------
# Optimized Generation Function
# --------------------------------------------------------------------------
def create_optimized_generate_function(model, tokenizer, device, args, use_amp=False):
    """Create an optimized generation function with various performance improvements"""
    
    def generate_batch(batch):
        with torch.no_grad():
            # Use AMP if enabled
            context_manager = autocast() if use_amp else torch.no_grad()
            
            with context_manager:
                # Tokenize input
                inputs = tokenizer(batch["incorrect_text"],
                                 padding=True,
                                 truncation=True,
                                 max_length=args.max_len,
                                 return_tensors="pt").to(device, non_blocking=True)
                
                # Generate with optimized settings
                gen_ids = model.generate(
                    **inputs,
                    num_beams=args.beam_size,
                    max_length=args.max_len,
                    early_stopping=True,
                    do_sample=False,  # Deterministic for consistency
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,  # Enable KV caching for faster beam search
                )
                
                # Decode predictions
                preds = tokenizer.batch_decode(gen_ids,
                                             skip_special_tokens=True,
                                             clean_up_tokenization_spaces=True)
                batch["pred"] = preds
                
        return batch
    
    return generate_batch


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    args = build_argparser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Optimize GPU settings
    optimize_gpu_settings()
    
    # ---- Weights & Biases login ------------------------------------------
    use_wandb = bool(args.wandb_project)
    if use_wandb:
        if args.wandb_api_key:
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.init(project=args.wandb_project,
                   entity=args.wandb_entity,
                   name=f"predict-{os.path.basename(args.model_path)}-optimized",
                   config=vars(args))
    
    # Detect whether we need word segmentation
    auto_word = "word" in os.path.basename(args.model_path).lower()
    do_segment = args.word_segment or auto_word

    # Validate subset_ratio parameter
    if not (0.0 < args.subset_ratio <= 1.0):
        raise ValueError(f"subset_ratio must be between 0.0 and 1.0, got {args.subset_ratio}")

    # Load model + tokenizer with optimizations
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    
    # Load model with appropriate dtype
    model_dtype = torch.float16 if args.fp16 else torch.float32
    
    # Load model without device_map to avoid tensor parallel issues
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_path,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True
        )
    except Exception as e:
        logger.warning(f"Failed to load model with torch_dtype={model_dtype}, trying without dtype specification: {e}")
        # Fallback: load without dtype specification
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_path,
            low_cpu_mem_usage=True
        )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Move model to device explicitly
    if device == "cuda":
        model.to(device)
    
    model.eval()
    
    # Enable torch.compile if requested (PyTorch 2.0+)
    if args.torch_compile and hasattr(torch, 'compile'):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")
    
    # Determine optimal batch size
    if args.dynamic_batching:
        optimal_batch_size = get_optimal_batch_size(tokenizer, model, device, args.batch_size, args.max_len)
        logger.info(f"Using optimal batch size: {optimal_batch_size}")
        args.batch_size = optimal_batch_size

    # Load dataset
    logger.info("Loading dataset...")
    if args.dataset_path:
        dataset = load_from_disk(args.dataset_path)[args.split]
    else:
        dataset = load_dataset(args.dataset_name, split=args.split, streaming=False)

    # Optional segmentation
    if do_segment:
        logger.info("Applying word segmentation...")
        dataset = dataset.map(lambda batch: segment_sentences(batch, args),
                              batched=True, 
                              batch_size=1024,
                              num_proc=args.num_workers)

    # Create optimized generation function
    generate_batch = create_optimized_generate_function(
        model, tokenizer, device, args, use_amp=args.use_amp
    )

    # Generation loop with progress tracking
    logger.info(f"Starting generation with batch size {args.batch_size}...")
    seen = 0
    t0 = time.time()
    
    def track_progress_and_generate(batch):
        nonlocal seen
        result = generate_batch(batch)
        
        # Update progress tracking
        seen += len(batch["incorrect_text"])
        elapsed = time.time() - t0
        
        if use_wandb and seen % (args.batch_size * 10) == 0:  # Log every 10 batches
            wandb.log({
                "sentences_processed": seen,
                "sentences_per_sec": seen / elapsed,
                "avg_latency_ms": 1000 * elapsed / seen,
                "gpu_memory_used": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            })
        
        # Periodic memory cleanup
        if seen % (args.batch_size * 50) == 0:  # Clean every 50 batches
            clear_gpu_cache()
            
        return result

    # Apply generation with progress bar
    dataset = dataset.map(
        track_progress_and_generate,
        batched=True,
        batch_size=args.batch_size,
        remove_columns=[c for c in dataset.column_names if c not in ("incorrect_text", "correct_text")],
        desc="Generating predictions"
    )

    # Final cleanup
    clear_gpu_cache()
    
    # Rename gold column -> target for clarity
    dataset = dataset.rename_column("correct_text", "target")

    # Save to disk for Stage-2 training
    logger.info(f"Saving results to {args.output_dir}...")
    dataset.save_to_disk(args.output_dir)
    
    # Final performance stats
    total_time = time.time() - t0
    logger.info(f"✓ Processed {seen} sentences in {total_time:.2f}s ({seen/total_time:.2f} sentences/sec)")
    
    # -------------- log preview table to W&B ------------------------------
    if use_wandb:
        import pandas as pd
        preview_df = pd.DataFrame(dataset.select(range(min(20, len(dataset)))))
        wandb.log({
            "preview": wandb.Table(dataframe=preview_df),
            "final_sentences_per_sec": seen / total_time,
            "total_processing_time": total_time
        })
        wandb.finish()
    
    # Also dump a small JSONL preview for sanity-check
    preview_path = os.path.join(args.output_dir, "head10.jsonl")
    with open(preview_path, "w", encoding="utf8") as fo:
        for ex in dataset.select(range(10)):
            fo.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    print(f"✓ Saved Stage-2 data to {args.output_dir}")
    print(f"Performance: {seen/total_time:.2f} sentences/sec")


if __name__ == "__main__":
    main()
