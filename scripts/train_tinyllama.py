"""
LoRA fine-tune TinyLlama 1.1B on pesnik.dev portfolio data.

Runs on Colab free T4 (16GB VRAM). Uses QLoRA + SFTTrainer.

Usage:
    # Upload dataset first, then:
    python scripts/train_tinyllama.py --dataset data/train.jsonl

    # Or from Colab:
    !python train_tinyllama.py --hf-token YOUR_TOKEN

Model output is saved locally and optionally pushed to HuggingFace Hub.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dataset: str = "data/train.jsonl"
    output_dir: str = "pesnik-tinyllama-lora"
    hf_token: str = ""
    hf_repo_id: str = "pesnik/portfolio-agent"
    push_to_hub: bool = False

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

    # Training
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    max_seq_length: int = 1024
    warmup_steps: int = 20
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2


def parse_args() -> Config:
    c = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=c.model_id)
    parser.add_argument("--dataset", default=c.dataset)
    parser.add_argument("--output-dir", default=c.output_dir)
    parser.add_argument("--hf-token", default=c.hf_token)
    parser.add_argument("--hf-repo-id", default=c.hf_repo_id)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--lora-r", type=int, default=c.lora_r)
    parser.add_argument("--lora-alpha", type=int, default=c.lora_alpha)
    parser.add_argument("--num-epochs", type=int, default=c.num_train_epochs)
    parser.add_argument("--batch-size", type=int, default=c.per_device_train_batch_size)
    parser.add_argument("--lr", type=float, default=c.learning_rate)
    parser.add_argument("--max-seq-length", type=int, default=c.max_seq_length)
    args = parser.parse_args()

    for k, v in vars(args).items():
        k = k.replace("-", "_")
        if hasattr(c, k):
            setattr(c, k, v)
    return c


def install_deps():
    """Install required deps — safe to call even if already installed."""
    deps = [
        "torch",
        "transformers>=4.49",
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "bitsandbytes",
        "huggingface_hub",
        "torchao",
    ]
    for d in deps:
        try:
            __import__(d.replace("-", "_").split(">=")[0].split("[")[0])
        except ImportError:
            pass

    # Check bitsandbytes CUDA
    try:
        import bitsandbytes as bnb
        bnb_is_ok = hasattr(bnb, "cuda_setup")
    except Exception:
        bnb_is_ok = False

    print(f"✓ deps ready  |  bitsandbytes CUDA: {'yes' if bnb_is_ok else 'no (4-bit may fail)'}")


def load_dataset(path: str):
    from datasets import load_dataset
    ds = load_dataset("json", data_files=path, split="train")
    print(f"✓ loaded {len(ds)} samples from {path}")
    return ds


def get_model_and_tokenizer(cfg: Config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    print(f"✓ loaded {cfg.model_id}  |  params: {model.num_parameters():,}")
    return model, tokenizer


def apply_lora(model, cfg: Config):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules.split(","),
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train(model, tokenizer, dataset, cfg: Config):
    import trl

    train_kwargs = dict(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=True,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=True, fp16=False,
        dataloader_num_workers=2,
        report_to="none",
        remove_unused_columns=False,
    )

    if hasattr(trl, "SFTConfig"):
        # Modern TRL (>=0.18): SFTConfig + processing_class
        from trl import SFTConfig, SFTTrainer
        train_kwargs["max_length"] = cfg.max_seq_length
        train_kwargs["dataset_text_field"] = "messages"
        train_kwargs["packing"] = False
        args = SFTConfig(**train_kwargs)
        trainer = SFTTrainer(
            model=model, args=args, train_dataset=dataset,
            processing_class=tokenizer,
        )
    else:
        # Legacy TRL (<=0.16): direct kwargs
        from transformers import TrainingArguments
        from trl import SFTTrainer
        args = TrainingArguments(**train_kwargs)
        trainer = SFTTrainer(
            model=model, args=args, train_dataset=dataset,
            max_seq_length=cfg.max_seq_length,
            dataset_text_field="messages",
            packing=False,
        )
        trainer.tokenizer = tokenizer

    trainer.train()

    print(f"✓ training complete  |  saving to {cfg.output_dir}")
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    return trainer


def push_to_hub(cfg: Config):
    from huggingface_hub import HfApi, login

    if not cfg.hf_token:
        print("⚠  no HF token set, skipping push")
        return

    login(token=cfg.hf_token)
    api = HfApi()

    repo_id = cfg.hf_repo_id
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=cfg.output_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"✓ pushed to https://huggingface.co/{repo_id}")


def test_inference(cfg: Config):
    """Quick smoke test — load adapters and generate."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, cfg.output_dir)

    prompt = "<|user|>\nWho are you?\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
    print("\n── Smoke test ──")
    print(tokenizer.decode(out[0], skip_special_tokens=False))
    print("────────────────\n")


def main():
    cfg = parse_args()

    print("═══ pesnik.dev — TinyLlama LoRA Fine-tune ═══")
    print(f"  model:   {cfg.model_id}")
    print(f"  dataset: {cfg.dataset}")
    print(f"  output:  {cfg.output_dir}")
    print(f"  epochs:  {cfg.num_train_epochs}  |  lr: {cfg.lr}  |  rank: {cfg.lora_r}")
    print()

    install_deps()
    dataset = load_dataset(cfg.dataset)
    model, tokenizer = get_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg)
    train(model, tokenizer, dataset, cfg)

    if cfg.push_to_hub:
        push_to_hub(cfg)

    test_inference(cfg)
    print("✓ done — you can now use the LoRA adapters for inference")


if __name__ == "__main__":
    main()
