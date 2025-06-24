"""
train_dpo.py – Alignment-tune Dolly-v2-3b with Direct Preference Optimization (DPO)
using a configurable subset of the Anthropic HH-RLHF preference dataset.

Run locally or on a single‑GPU cloud instance (e.g. RunPod A100). Example:

    python train_dpo.py \
        --model_name databricks/dolly-v2-3b \
        --subset_size 20000 \
        --subset_strategy random \
        --output_dir ./dpo_dolly_20k \
        --batch_size 32 \
        --gradient_accumulation_steps 16 \
        --epochs 3 \
        --learning_rate 2e-5 \
        --beta 0.1

Dependencies (see requirements.txt):
  • python>=3.11   • torch>=2.3   • transformers>=4.43   • datasets>=2.20
  • trl[torch]>=0.9   • accelerate>=0.28   • bitsandbytes   • evaluate   • wandb (optional)
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Literal

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from trl import DPOTrainer, DPOConfig
from accelerate import Accelerator

################################################################################
# Helper functions
################################################################################

def sample_hh_dataset(
    size: int,
    strategy: Literal["random", "stratified"] = "random",
    seed: int = 42,
):
    """Return a DatasetDict with `prompt`, `chosen`, `rejected` columns limited to
    *size* examples using the selected *strategy*.

    Supported strategies:
      • random – uniform random sampling.
      • stratified – equal halves of harmless/helpful categories.
    """
    hh = load_dataset("Anthropic/hh-rlhf", split="train")

    if strategy == "random":
        hh = hh.shuffle(seed=seed).select(range(size))
    elif strategy == "stratified":
        # Split by helpful vs harmless tags.
        helpful = hh.filter(lambda ex: ex["helpfulness"] == "helpful")
        harmless = hh.filter(lambda ex: ex["harmlessness"] == "harmless")
        half = size // 2
        helpful = helpful.shuffle(seed=seed).select(range(half))
        harmless = harmless.shuffle(seed=seed).select(range(size - half))
        hh = helpful.concatenate(harmless)
        hh = hh.shuffle(seed=seed)
    else:
        raise ValueError("Unsupported subset_strategy. Choose 'random' or 'stratified'.")

    # Keep only the columns DPO expects.
    hh = hh.remove_columns([col for col in hh.column_names if col not in {"prompt", "chosen", "rejected"}])
    return hh

################################################################################
# Argument parsing
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="DPO fine‑tune Dolly with HH‑RLHF subset")

    # Model & data
    parser.add_argument("--model_name", default="databricks/dolly-v2-3b", help="HF model repo or path")
    parser.add_argument("--subset_size", type=int, default=20000, help="Number of HH‑RLHF pairs to train on")
    parser.add_argument("--subset_strategy", choices=["random", "stratified"], default="random")
    parser.add_argument("--subset_seed", type=int, default=42, help="Deterministic seed for subset sampling")

    # Training hyper‑parameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32, help="Reduced default batch size for memory efficiency")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Increased to maintain effective batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.1, help="β coefficient in DPO objective")
    parser.add_argument("--max_len", type=int, default=512)

    # Memory optimization options
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing to save memory")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization for even more memory savings")
    parser.add_argument("--share_reference_model", action="store_true", default=True, help="Share reference model to save memory")

    # Misc
    parser.add_argument("--output_dir", default="./dpo_output")
    parser.add_argument("--wandb_project", default=None, help="Enable Weights & Biases logging with this project name")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

################################################################################
# Main routine
################################################################################

def main():
    args = parse_args()

    # Set environment variable for memory fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    accelerator = Accelerator(log_with="wandb" if args.wandb_project else None)
    if accelerator.is_main_process:
        print("Loading base & reference models…")
        print(f"Memory optimization settings:")
        print(f"  - Batch size: {args.batch_size}")
        print(f"  - Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"  - Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
        print(f"  - Gradient checkpointing: {args.gradient_checkpointing}")
        print(f"  - Share reference model: {args.share_reference_model}")

    # Ensure determinism
    set_seed(args.seed)

    # Load tokenizer (shared between policy & ref)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    
    # Fix warning: ensure pad_token_id and eos_token_id are different
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        print(f"Fixed tokenizer: pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}")

    # Prepare model loading kwargs
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    # Add quantization if requested
    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    # Load policy model (trainable)
    policy_model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    policy_model.resize_token_embeddings(len(tokenizer))

    # Enable gradient checkpointing for memory savings
    if args.gradient_checkpointing:
        policy_model.gradient_checkpointing_enable()

    # Reference model handling
    if args.share_reference_model:
        # Use the same model for reference (memory efficient)
        ref_model = None
        if accelerator.is_main_process:
            print("Using shared reference model (memory efficient)")
    else:
        # Load separate reference model (more memory but potentially better results)
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
        if accelerator.is_main_process:
            print("Using separate reference model")

    # Dataset
    if accelerator.is_main_process:
        print("Sampling HH‑RLHF subset…")
    train_dataset = sample_hh_dataset(
        size=args.subset_size,
        strategy=args.subset_strategy,
        seed=args.subset_seed,
    )

    # Create DPO training arguments with memory optimizations
    training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",  # Save only at epoch end
        save_total_limit=2,  # Keep only 2 checkpoints
        dataloader_drop_last=True,  # Drop incomplete batches
        dataloader_pin_memory=False,  # Reduce memory usage
        remove_unused_columns=False,
        report_to=["wandb"] if args.wandb_project else None,
        max_length=args.max_len,
        max_prompt_length=256,  # Limit prompt length
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Clear cache before creating trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Trainer
    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # Kick off training
    if accelerator.is_main_process:
        print("Starting training…")
        print(f"GPU memory before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    trainer.train()

    # Save
    if accelerator.is_main_process:
        print("Saving fine‑tuned model to", args.output_dir)
    trainer.save_model(args.output_dir)

    if accelerator.is_main_process:
        print("Done ✔")


if __name__ == "__main__":
    main()
