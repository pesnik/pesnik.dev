"""
Merge LoRA adapters into base model and export for WebGPU (MLC-LLM) deployment.

Run this after train_tinyllama.py on Colab:

  Step 1 — merge + push to HuggingFace:
    python scripts/merge_and_export.py --hf-token YOUR_TOKEN

  Step 2 — convert to MLC-LLM weights (run separately on the Colab):
    pip install mlc-llm
    mlc_llm convert_weight ./pesnik-merged --quantization q4f16_1 -o ./pesnik-mlc
    mlc_llm gen_config ./pesnik-mlc --quantization q4f16_1 --conv-template chatml -o ./pesnik-mlc
    # Push ./pesnik-mlc to HuggingFace pesnik/pesnik-webgpu

  The index.html WebGPU config already points to pesnik/pesnik-webgpu — no web changes needed.
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    base_model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    lora_adapter_id: str = "pesnik/pesnik-lora"
    merged_output_dir: str = "pesnik-merged"
    hf_merged_repo: str = "pesnik/pesnik"
    hf_token: str = ""
    push_to_hub: bool = False


def parse_args() -> Config:
    c = Config()
    parser = argparse.ArgumentParser(description="Merge LoRA + export for WebGPU")
    parser.add_argument("--base-model-id", default=c.base_model_id)
    parser.add_argument("--lora-adapter-id", default=c.lora_adapter_id)
    parser.add_argument("--merged-output-dir", default=c.merged_output_dir)
    parser.add_argument("--hf-merged-repo", default=c.hf_merged_repo)
    parser.add_argument("--hf-token", default=c.hf_token)
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()
    for k, v in vars(args).items():
        k = k.replace("-", "_")
        if hasattr(c, k):
            setattr(c, k, v)
    return c


def merge_lora(cfg: Config):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model: {cfg.base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print(f"Loading LoRA adapter: {cfg.lora_adapter_id}")
    model = PeftModel.from_pretrained(base, cfg.lora_adapter_id)

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {cfg.merged_output_dir}/")
    model.save_pretrained(cfg.merged_output_dir)
    tokenizer.save_pretrained(cfg.merged_output_dir)

    print(f"✓ Merged model saved to {cfg.merged_output_dir}/")
    return model, tokenizer


def push_merged(cfg: Config):
    from huggingface_hub import HfApi, login

    if not cfg.hf_token:
        print("⚠  No HF token — skipping push. Pass --hf-token to enable.")
        return

    login(token=cfg.hf_token)
    api = HfApi()
    api.create_repo(cfg.hf_merged_repo, exist_ok=True)
    api.upload_folder(
        folder_path=cfg.merged_output_dir,
        repo_id=cfg.hf_merged_repo,
        repo_type="model",
    )
    print(f"✓ Pushed merged model to https://huggingface.co/{cfg.hf_merged_repo}")


def print_mlc_instructions(cfg: Config):
    print("\n" + "═" * 60)
    print("Step 2 — Convert to MLC-LLM weights for WebGPU:")
    print("═" * 60)
    print(f"""
pip install mlc-llm

# Convert weights (q4f16_1 = 4-bit float16, same quant as the WebGPU WASM)
mlc_llm convert_weight ./{cfg.merged_output_dir} \\
    --quantization q4f16_1 \\
    -o ./pesnik-mlc

# Generate model config
mlc_llm gen_config ./pesnik-mlc \\
    --quantization q4f16_1 \\
    --conv-template chatml \\
    -o ./pesnik-mlc

# Push MLC weights to HuggingFace
huggingface-cli upload pesnik/pesnik-webgpu ./pesnik-mlc --repo-type model

# The WASM binary in index.html stays the same (TinyLlama 1.1B architecture).
# Only the model weights (pesnik/pesnik-webgpu) need updating.
""")


def main():
    cfg = parse_args()

    print("═══ pesnik — LoRA merge & WebGPU export ═══")
    print(f"  base:    {cfg.base_model_id}")
    print(f"  lora:    {cfg.lora_adapter_id}")
    print(f"  output:  {cfg.merged_output_dir}")
    print()

    merge_lora(cfg)

    if cfg.push_to_hub:
        push_merged(cfg)

    print_mlc_instructions(cfg)
    print("✓ Done. Run Step 2 above to complete WebGPU deployment.")


if __name__ == "__main__":
    main()
