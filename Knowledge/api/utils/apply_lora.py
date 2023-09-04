"""
Apply the LoRA weights on top of a base model.

Usage:
python api/utils/apply_lora.py --base ~/model_weights/llama-7b --target ~/model_weights/baize-7b --lora project-baize/baize-lora-7B
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def apply_lora(base_model_path, target_model_path, lora_path):
    print(f"Loading the base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False, trust_remote_code=True)

    print(f"Loading the LoRA adapter from {lora_path}")

    lora_model = PeftModel.from_pretrained(base, lora_path)

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    print(f"Saving the target model to {target_model_path}")
    model.save_pretrained(target_model_path)
    base_tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--lora-path", type=str, required=True)

    args = parser.parse_args()

    apply_lora(args.base_model_path, args.target_model_path, args.lora_path)
