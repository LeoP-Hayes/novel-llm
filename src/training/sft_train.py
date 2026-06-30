"""
Qwen3 SFT 微调训练脚本

基于 transformers + peft + trl，对 Qwen3-30B-A3B 做 LoRA 微调。
支持 4×A5000 (96GB) 硬件配置。

用法:
    # Qwen3-14B 快速验证
    python -m src.training.sft_train --model Qwen/Qwen3-14B --epochs 1 --max_samples 500

    # Qwen3-30B-A3B 正式训练
    python -m src.training.sft_train --model Qwen/Qwen3-30B-A3B --epochs 3

    # DeepSpeed ZeRO-2
    deepspeed --num_gpus=4 src/training/sft_train.py --deepspeed
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer


# ============================================================
# 训练配置
# ============================================================

@dataclass
class TrainConfig:
    # 模型
    model_name: str = "Qwen/Qwen3-30B-A3B"
    use_4bit: bool = False            # Qwen3-30B-A3B 用 bf16 即可装下
    bf16: bool = True
    attn_implementation: str = "flash_attention_2"

    # LoRA
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: tuple = (
        "q_proj", "v_proj", "k_proj", "o_proj",
        "up_proj", "down_proj",
        # "gate_proj",  # 首轮不加，防 MoE 路由崩溃
    )

    # 训练超参
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8   # effective batch = 2×8×4GPU = 64
    max_seq_length: int = 8192
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    num_epochs: int = 3
    optimizer: str = "adamw_8bit"
    gradient_checkpointing: bool = True

    # 数据
    train_file: str = "data/sft/train.jsonl"
    val_file: str = "data/sft/val.jsonl"
    max_samples: Optional[int] = None   # 限制样本数（快速验证用）

    # 输出
    output_dir: str = "models/lora_adapters"
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3

    # DeepSpeed
    use_deepspeed: bool = False
    deepspeed_stage: int = 2


# ============================================================
# 数据加载
# ============================================================

def load_sft_data(train_file: str, val_file: str, max_samples: Optional[int] = None):
    """加载 SFT JSONL 数据，转换为 messages 格式"""
    def _load(path):
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                samples.append(item["messages"])  # 提取 messages 列表
        return samples

    train_data = _load(train_file)
    val_data = _load(val_file)

    if max_samples and max_samples < len(train_data):
        import random
        random.seed(42)
        train_data = random.sample(train_data, max_samples)
        val_data = random.sample(val_data, min(max_samples // 10, len(val_data)))

    print(f"📂 训练集: {len(train_data)} 条 | 验证集: {len(val_data)} 条")

    # 转换为 HuggingFace Dataset
    train_dataset = Dataset.from_list([{"messages": m} for m in train_data])
    val_dataset = Dataset.from_list([{"messages": m} for m in val_data])

    return train_dataset, val_dataset


# ============================================================
# 主流程
# ============================================================

def train(config: TrainConfig):
    # --- 1. 加载数据 ---
    train_dataset, val_dataset = load_sft_data(
        config.train_file, config.val_file, config.max_samples
    )

    # --- 2. 加载 Tokenizer ---
    print(f"🔧 加载 tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 3. 加载模型 ---
    print(f"🤖 加载模型: {config.model_name}")
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16 if config.bf16 else torch.float32,
        "attn_implementation": config.attn_implementation,
    }

    if config.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs,
    )

    if config.gradient_checkpointing:
        model.enable_input_require_grads()

    # --- 4. LoRA 配置 ---
    print(f"🔧 LoRA: rank={config.lora_rank}, alpha={config.lora_alpha}")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
    )

    # --- 5. 训练参数 ---
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler,
        warmup_steps=int(config.warmup_ratio * 100),  # warmup_ratio → warmup_steps
        num_train_epochs=config.num_epochs,
        optim=config.optimizer,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name=f"qwen3-novel-lora-r{config.lora_rank}",
        remove_unused_columns=False,
        dataloader_num_workers=0,       # JSONL 数据用 0 避免多进程问题
        deepspeed=(
            f"configs/ds_z{config.deepspeed_stage}.json"
            if config.use_deepspeed else None
        ),
    )

    # --- 6. 训练 ---
    print(f"🚀 开始训练...")
    tokenizer.model_max_length = config.max_seq_length

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()

    # --- 7. 保存 ---
    final_dir = Path(config.output_dir) / "final"
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"✅ 训练完成！LoRA 权重保存至: {final_dir}")

    return trainer


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Qwen3 SFT 微调")
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--deepspeed", action="store_true")
    p.add_argument("--deepspeed_stage", type=int, default=2)
    p.add_argument("--output_dir", default="models/lora_adapters")
    p.add_argument("--train_file", default="data/sft/train.jsonl")
    p.add_argument("--val_file", default="data/sft/val.jsonl")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = TrainConfig(
        model_name=args.model,
        num_epochs=args.epochs,
        max_samples=args.max_samples,
        lora_rank=args.lora_rank,
        use_deepspeed=args.deepspeed,
        deepspeed_stage=args.deepspeed_stage,
        output_dir=args.output_dir,
        train_file=args.train_file,
        val_file=args.val_file,
    )

    train(config)
