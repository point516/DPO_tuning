"""
train_dpo.py – Alignment-tune Dolly-v2-3b with Direct Preference Optimization (DPO)
using a configurable subset of the Anthropic HH-RLHF preference dataset.

Run locally or on a single‑GPU cloud instance (e.g. RunPod A100). Example:

    python train_dpo.py \
        --model_name databricks/dolly-v2-3b \
        --subset_size 20000 \
        --subset_strategy random \
        --output_dir ./dpo_dolly_20k \
        --batch_size 128 \
        --gradient_accumulation_steps 4 \
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
from trl import DPOTrainer, DataCollatorForCompletionOnlyLM
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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.1, help="β coefficient in DPO objective")
    parser.add_argument("--max_len", type=int, default=512)

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

    accelerator = Accelerator(log_with="wandb" if args.wandb_project else None)
    if accelerator.is_main_process:
        print("Loading base & reference models…")

    # Ensure determinism
    set_seed(args.seed)

    # Load tokenizer (shared between policy & ref)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # Load policy model (trainable)
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    policy_model.resize_token_embeddings(len(tokenizer))

    # Reference model (frozen) – same weights as policy at init
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    # Dataset
    if accelerator.is_main_process:
        print("Sampling HH‑RLHF subset…")
    train_dataset = sample_hh_dataset(
        size=args.subset_size,
        strategy=args.subset_strategy,
        seed=args.subset_seed,
    )

    # Data collator – add BOS token before chosen/rejected completions
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template="<|assistant|>",  # Dolly instruct format
        instruction_template="<|user|>",
        padding=True,
    )

    # Trainer
    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=None,  # we’ll build TrainingArguments below
        beta=args.beta,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        dataset_text_field="prompt",
        collator=collator,
        max_length=args.max_len,
        peft_config=None,
    )

    # Build HF TrainingArguments through trainer utility for simplicity
    trainer._prepare_training_args(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        # save_steps=250,
        save_strategy="no",
        report_to=["wandb"] if args.wandb_project else None,
    )

    # Kick off training
    if accelerator.is_main_process:
        print("Starting training…")
    trainer.train()

    # Save
    if accelerator.is_main_process:
        print("Saving fine‑tuned model to", args.output_dir)
    trainer.save_model(args.output_dir)

    if accelerator.is_main_process:
        print("Done ✔")


if __name__ == "__main__":
    main()
