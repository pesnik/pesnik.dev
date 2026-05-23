"""
Merge LoRA adapters into base TinyLlama, save full model for MLC conversion.
Usage: python scripts/merge_lora.py [--lora REPO_ID] [--output ./merged-model]
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_REPO = "pesnik/portfolio-agent"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", default=LORA_REPO)
    parser.add_argument("--output", default="./merged-model")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Loading base: {BASE}")
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")

    print(f"Loading LoRA from: {args.lora}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, args.lora)
    print("Merging LoRA into base...")
    merged = model.merge_and_unload()

    print(f"Saving full model to {output}")
    merged.save_pretrained(output, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    tokenizer.save_pretrained(output)

    print(f"Done — model saved to {output.resolve()}")


if __name__ == "__main__":
    main()
