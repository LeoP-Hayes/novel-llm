"""
Qwen3 SFT 微调训练脚本
=======================

功能: 基于 transformers + peft + trl 对 Qwen3 系列模型做 LoRA 微调

核心技术:
  - LoRA: 只训练低秩适配器矩阵，冻结原始权重。显存需求约为全参数微调的 1/3
  - SFTTrainer: trl 库提供的监督微调训练器，自动处理 ChatML 格式的数据
  - DeepSpeed ZeRO: 多卡分布式训练的显存优化策略（ZeRO-2 分片优化器，ZeRO-3 分片参数）
  - Gradient Checkpointing: 用计算换显存，不存储中间激活值，在反向传播时重新计算

用法:
    # Qwen3-14B 快速验证（本地 Mac）
    python -m src.training.sft_train --model Qwen/Qwen3-14B --epochs 1 --max_samples 500

    # Qwen3-30B-A3B 正式训练（AutoDL A100）
    deepspeed --num_gpus=1 src/training/sft_train.py --model Qwen/Qwen3-30B-A3B --epochs 3 --deepspeed
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field  # dataclass: 用类定义配置，IDE 有代码提示
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset       # HuggingFace 数据集格式，比 list[dict] 更高效
from transformers import (
    AutoModelForCausalLM,          # 自动加载因果语言模型（GPT 类）
    AutoTokenizer,                 # 自动加载分词器
    TrainingArguments,             # 训练超参容器（学习率、epoch、保存策略等）
    BitsAndBytesConfig,            # QLoRA 4-bit 量化配置（显存紧张时启用）
)
from peft import (
    LoraConfig,                    # LoRA 配置（秩、alpha、目标层）
    get_peft_model,                # 给模型注入 LoRA 适配器
    TaskType,                      # 任务类型枚举（因果语言模型）
    PeftModel,                     # 加载已保存的 PEFT 模型
)
from trl import SFTTrainer         # trl 的监督微调训练器，封装了数据格式化、loss 计算等


# ============================================================
# 训练配置
# ============================================================
# 所有可调参数集中在 TrainConfig 中，方便切换模型/硬件/数据规模
# 通过命令行 --model --epochs 等参数覆盖默认值

@dataclass
class TrainConfig:
    # ========== 模型相关 ==========
    # 基座模型名称或本地路径，默认 Qwen3-30B-A3B（30B MoE/3B 激活参数/A100 bf16 可装下）
    model_name: str = "Qwen/Qwen3-30B-A3B"

    # 是否开启 4-bit 量化（显存不够时启用，如 RTX 4090 24GB）
    # A100 80GB 用 bf16 即可，use_4bit=False
    use_4bit: bool = False

    # bf16 混合精度训练（A100/H100 支持，V100 不支持需设为 False 改用 fp16）
    bf16: bool = True

    # 注意力实现方式。flash_attention_2 最快但需 CUDA 支持；Mac MPS 用 eager
    attn_implementation: str = "flash_attention_2"

    # ========== LoRA 相关 ==========
    # LoRA 秩（rank）：低秩矩阵的维度。rank 越大，适配器容量越大，但训练越慢
    # A100 充裕用 64，RTX 4090 用 32，本地验证用 8
    lora_rank: int = 64

    # LoRA 缩放系数：控制适配器输出权重 = alpha / rank
    # 一般设为 rank 的 2 倍（alpha=128, rank=64）
    lora_alpha: int = 128

    # LoRA dropout：防止适配器过拟合，0.05 表示随机丢弃 5% 的神经元
    lora_dropout: float = 0.0   # MoE 融合层(gate_up_proj)不支持 dropout，必须设为 0

    # LoRA 目标模块：在这些层的权重矩阵上添加低秩适配器
    # Qwen3 是 Transformer 架构，q_proj/k_proj/v_proj/o_proj 是注意力层
    # up_proj/down_proj 是 FFN 层（SwiGLU 的门控和输出）
    # gate_proj 是 MoE 的专家路由门控层 —— 首轮暂不加入，防止路由分布崩溃
    lora_target_modules: tuple = (
        "q_proj", "v_proj", "k_proj", "o_proj",    # 注意力层 4 个投影矩阵
        "gate_up_proj", "down_proj",                 # MoE FFN：gate_up_proj 是 gate+up 的融合层
    )

    # ========== 训练超参 ==========
    # 每张 GPU 每步处理的样本数
    per_device_batch_size: int = 2

    # 梯度累积步数：每 N 步才更新一次权重，模拟更大的 batch size
    # 有效 batch size = per_device_batch_size × gradient_accumulation × num_gpus
    #            = 2 × 8 × 1(A100 单卡) = 16
    gradient_accumulation_steps: int = 8

    # 最大序列长度（tokens）。超过此长度的样本被截断，短于此的样本被填充
    # A100 充裕用 8192，RTX 4090 需降到 4096，本地验证用 512
    max_seq_length: int = 8192

    # 学习率：LoRA 微调通常用 1e-4 ~ 5e-4，比全参数微调稍高
    # 全参数微调通常用 1e-5 ~ 5e-5
    learning_rate: float = 2e-4

    # 学习率调度器：cosine 余弦退火（从 lr 开始，余弦曲线降到接近 0）
    # 其他选项: linear, constant, polynomial
    lr_scheduler: str = "cosine"

    # 预热比例：前 5% 的训练步数学习率从 0 线性增长到 lr，防止训练初期不稳定
    warmup_ratio: float = 0.05

    # 训练轮数：整个数据集遍历的次数
    # 网文微调 2-3 轮即可，太多会过拟合（模型只会复述训练数据，丧失泛化能力）
    num_epochs: int = 3

    # 优化器：adamw_8bit 是 8-bit 量化的 AdamW（省显存 30%，A100/CUDA 推荐）
    # Mac MPS 不支持 8-bit，本地验证时需改为 adamw_torch
    optimizer: str = "adamw_8bit"

    # 梯度检查点：用计算换显存。开启后显存减少 ~30%，但训练速度下降 ~20%
    # A100 充裕可关闭加速，RTX 4090 最好开启
    gradient_checkpointing: bool = True

    # ========== 数据相关 ==========
    # SFT 训练数据路径（由 sft_data_builder.py 生成）
    train_file: str = "data/sft/train.jsonl"
    val_file: str = "data/sft/val.jsonl"

    # 限制训练样本数（None = 全量, 500 = 快速验证用）
    max_samples: Optional[int] = None

    # ========== 输出相关 ==========
    # 模型和日志保存目录
    output_dir: str = "models/lora_adapters"

    # 每 N 步记录一次训练指标（loss, learning_rate, 显存占用等）
    logging_steps: int = 50

    # 每 N 步保存一次 checkpoint（模型权重 + 优化器状态，可恢复训练）
    save_steps: int = 500

    # 每 N 步在验证集上评估一次
    eval_steps: int = 500

    # 最多保留几个历史 checkpoint（超出自动删除最旧的）
    save_total_limit: int = 3

    # ========== DeepSpeed 相关 ==========
    # 是否启用 DeepSpeed（单卡训练不需要，多卡训练建议开启）
    use_deepspeed: bool = False

    # DeepSpeed ZeRO stage:
    #   2 = 分片优化器状态 + 梯度（推荐，对通信带宽要求低）
    #   3 = 分片优化器 + 梯度 + 模型参数（极致省显存，但通信开销大）
    deepspeed_stage: int = 2


# ============================================================
# 数据加载
# ============================================================

def load_sft_data(train_file: str, val_file: str, max_samples: Optional[int] = None):
    """
    加载 SFT JSONL 数据，转换为 HuggingFace Dataset 格式

    JSONL 结构（每行一条样本）:
    {
      "messages": [
        {"role": "system", "content": "你是一个都市文娱小说作家..."},
        {"role": "user", "content": "【前文片段】...请续写第X章"},
        {"role": "assistant", "content": "<真实章节正文>"}
      ],
      "task_type": "continuation"
    }

    返回: (train_dataset, val_dataset)
    """
    def _load(path):
        """读取 JSONL 文件，每行一个 JSON 对象，提取 messages 字段"""
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                # 提取 messages 列表（ChatML 格式的 system/user/assistant 对话）
                # 训练时 SFTTrainer 会自动将这个列表格式化为 ChatML token 序列
                samples.append(item["messages"])
        return samples

    train_data = _load(train_file)
    val_data = _load(val_file)

    # 快速验证模式：只取前 N 条样本（--max_samples 500）
    if max_samples and max_samples < len(train_data):
        import random
        random.seed(42)  # 固定随机种子，保证每次抽到的样本相同
        train_data = random.sample(train_data, max_samples)
        val_data = random.sample(val_data, min(max_samples // 10, len(val_data)))

    print(f"📂 训练集: {len(train_data)} 条 | 验证集: {len(val_data)} 条")

    # 转换为 HuggingFace Dataset 格式（比 Python list 更高效，支持流式加载和 map 并行处理）
    # 每行数据是一个字典 {"messages": [...]}
    train_dataset = Dataset.from_list([{"messages": m} for m in train_data])
    val_dataset = Dataset.from_list([{"messages": m} for m in val_data])

    return train_dataset, val_dataset


# ============================================================
# 主训练流程
# ============================================================

def train(config: TrainConfig):
    """
    完整训练流程，按顺序执行以下步骤:
      1. 加载 SFT 数据
      2. 加载 Tokenizer（分词器）
      3. 加载基座模型
      4. 注入 LoRA 适配器
      5. 配置 TrainingArguments
      6. 创建 SFTTrainer 并开始训练
      7. 保存 LoRA 权重
    """

    # ===================================================================
    # 第 1 步: 加载数据
    # ===================================================================
    # 从 data/sft/train.jsonl 和 val.jsonl 读取训练/验证集
    # 如果设置了 max_samples，会随机抽取子集（快速验证用）
    train_dataset, val_dataset = load_sft_data(
        config.train_file, config.val_file, config.max_samples
    )

    # ===================================================================
    # 第 2 步: 加载 Tokenizer（分词器）
    # ===================================================================
    # Tokenizer 负责三件事:
    #   1. 将中文字符切分为 tokens（Qwen3 用 BPE 分词，中文字≈1-2 tokens）
    #   2. 添加特殊 token（<|im_start|>, <|im_end|> 等 ChatML 标记）
    #   3. 填充/截断到统一长度，并生成 attention_mask
    #
    # padding_side="right": 在序列右侧填充（生成式模型标准做法）
    # trust_remote_code=True: 允许执行模型仓库中的自定义代码
    print(f"🔧 加载 tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    # 如果模型没有定义 pad_token（填充标记），用 eos_token（结束标记）替代
    # 这是很多开源模型的通用做法
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ===================================================================
    # 第 3 步: 加载基座模型
    # ===================================================================
    # AutoModelForCausalLM 自动识别模型架构（Qwen3 是因果语言模型）
    #
    # dtype=torch.bfloat16: 用 16-bit Brain Float 格式存储权重
    #   - 省一半显存（vs fp32）
    #   - 数值范围与 fp32 相同（vs fp16 可能溢出）
    #   - 需要 A100/H100/RTX 30xx+ 支持
    #
    # attn_implementation="flash_attention_2": 用 Flash Attention 加速推理
    #   - 比原生 attention 快 2-3 倍
    #   - 省显存（不存储完整注意力矩阵）
    #
    # use_4bit=True 时: 用 QLoRA 4-bit 量化，模型权重压缩到原来的 1/4
    #   - RTX 4090 24GB 必备，A100 80GB 不需要
    print(f"🤖 加载模型: {config.model_name}")
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16 if config.bf16 else torch.float32,
        "attn_implementation": config.attn_implementation,
    }

    # QLoRA 4-bit 量化配置（仅显存紧张时启用）
    # load_in_4bit=True: 基座权重用 4-bit 存储
    # bnb_4bit_compute_dtype=bf16: 计算时反量化到 bf16
    # bnb_4bit_use_double_quant=True: 二次量化（对量化参数再量化，进一步省 0.4 bit/参数）
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

    # 梯度检查点时启用：确保模型输入需要梯度（否则 LoRA 适配器收不到梯度）
    if config.gradient_checkpointing:
        model.enable_input_require_grads()

    # ===================================================================
    # 第 4 步: 注入 LoRA 适配器
    # ===================================================================
    # LoRA (Low-Rank Adaptation): 在原始权重矩阵旁添加低秩矩阵 A×B
    #   原始: y = Wx
    #   LoRA: y = Wx + (alpha/r) × A × B × x
    #          ↑冻结↑       ↑可训练，参数量极小↑
    #
    # r=64(A100)/32(4090)/8(本地验证): 低秩矩阵的秩
    # alpha=128: 缩放系数，alpha/r 决定适配器输出的强度
    # target_modules: 在哪些层添加 LoRA
    #   - Attention 层 (q/v/k/o_proj): 影响模型"关注哪里"的能力
    #   - FFN 层 (up/down_proj): 影响模型"存储知识"的能力
    #   - gate_proj (MoE 专家路由): 首轮不加，避免路由崩溃
    print(f"🔧 LoRA: rank={config.lora_rank}, alpha={config.lora_alpha}")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,   # 因果语言模型任务
        r=config.lora_rank,              # 低秩矩阵维度
        lora_alpha=config.lora_alpha,    # 缩放系数
        lora_dropout=config.lora_dropout,# 正则化 dropout
        target_modules=list(config.lora_target_modules),  # 目标模块列表
    )
    # 注意: peft_config 传给 SFTTrainer 后，训练器会自动调用 get_peft_model 注入适配器
    # 不需要手动调用 get_peft_model(model, peft_config)

    # ===================================================================
    # 第 5 步: 配置训练参数
    # ===================================================================
    # TrainingArguments 是 HuggingFace 的统一训练配置类
    # 控制: 学习率、batch size、保存策略、评估策略、日志、分布式等
    training_args = TrainingArguments(
        # --- 路径 ---
        output_dir=config.output_dir,       # 模型 checkpoint 保存位置

        # --- Batch Size ---
        per_device_train_batch_size=config.per_device_batch_size,  # 每 GPU 每步的训练样本数
        per_device_eval_batch_size=1,                               # 评估时 batch size（1 省显存）

        # gradient_accumulation_steps: 梯度累积 —— 模拟大 batch 的省钱方法
        # 原理: 每 N 步才执行一次 optimizer.step()，中间只累积梯度不更新权重
        # 有效 batch = per_device_batch × accumulation_steps × num_gpus
        gradient_accumulation_steps=config.gradient_accumulation_steps,

        # --- 学习率 ---
        learning_rate=config.learning_rate,     # 初始学习率
        lr_scheduler_type=config.lr_scheduler,  # 调度器类型: cosine
        warmup_steps=int(config.warmup_ratio * 100),  # 预热步数

        # --- 训练轮数 ---
        num_train_epochs=config.num_epochs,

        # --- 优化器和精度 ---
        optim=config.optimizer,                # adamw_8bit (CUDA) / adamw_torch (MPS)
        bf16=config.bf16,                      # bf16 混合精度（A100 支持，V100 不支持）
        gradient_checkpointing=config.gradient_checkpointing,  # 用计算换显存

        # --- 日志 ---
        logging_steps=config.logging_steps,     # 每 N 步记录 loss/lr 到 wandb

        # --- Checkpoint ---
        save_steps=config.save_steps,           # 每 N 步保存一次（训练中断可从此恢复）
        save_total_limit=config.save_total_limit,  # 只保留最近 3 个 checkpoint
        eval_steps=config.eval_steps,           # 每 N 步在验证集上评估

        # --- 评估 ---
        eval_strategy="steps",                  # 按步数评估（而非按 epoch）
        load_best_model_at_end=True,            # 训练结束后自动加载验证 loss 最低的 checkpoint
        metric_for_best_model="eval_loss",      # 用验证集 loss 评判"最佳"
        greater_is_better=False,                # loss 越低越好

        # --- 日志和报告 ---
        report_to="wandb",                      # 用 Weights & Biases 可视化训练曲线
        run_name=f"qwen3-novel-lora-r{config.lora_rank}",  # wandb 中的实验名称

        # --- 数据加载 ---
        remove_unused_columns=False,            # 保留 messages 字段（不自动删除未使用列）
        dataloader_num_workers=0,               # 数据加载进程数（0=主进程，避免多进程死锁）

        # --- DeepSpeed ---
        # 指定 DeepSpeed 配置文件，如 configs/ds_z2.json
        # None 表示不使用 DeepSpeed（单卡训练）
        deepspeed=(
            f"configs/ds_z{config.deepspeed_stage}.json"
            if config.use_deepspeed else None
        ),
    )

    # ===================================================================
    # 第 6 步: 创建 SFTTrainer 并开始训练
    # ===================================================================
    # SFTTrainer 是 trl 库的核心训练器，封装了以下逻辑:
    #   1. 将 messages 格式化为 ChatML token 序列
    #   2. 对 assistant 部分计算 loss（system 和 user 部分不参与 loss 计算）
    #   3. 自动调用 get_peft_model 注入 LoRA 适配器
    #   4. 处理批次打包（packing）、序列截断、padding 等
    print(f"🚀 开始训练...")

    # 设置 tokenizer 的最大序列长度（超出截断，短则填充）
    tokenizer.model_max_length = config.max_seq_length

    trainer = SFTTrainer(
        model=model,                    # 基座模型（尚未注入 LoRA）
        args=training_args,             # 训练超参
        train_dataset=train_dataset,    # 训练集
        eval_dataset=val_dataset,       # 验证集
        peft_config=peft_config,        # LoRA 配置（训练器会自动注入）
        processing_class=tokenizer,     # 分词器（新版 trl 参数名）
    )
    # SFTTrainer 内部做的事:
    #   1. 调用 tokenizer.apply_chat_template(messages) 格式化每条数据
    #   2. 对完整序列做 tokenize
    #   3. 创建 labels（只对 assistant 部分计算 loss，system 和 user 部分的 label 设为 -100）
    #   4. 注入 LoRA 适配器（冻结基座 + 添加低秩矩阵）

    # 执行训练循环
    trainer.train()

    # ===================================================================
    # 第 7 步: 保存 LoRA 权重
    # ===================================================================
    # LoRA 权重只包含低秩矩阵 A×B，不包含冻结的基座权重
    # 文件大小: ~200MB (vs 基座模型 60GB)
    # 保存内容包括:
    #   - adapter_model.safetensors: LoRA 权重矩阵
    #   - adapter_config.json: LoRA 配置（rank, alpha, target_modules）
    #   - tokenizer 文件: 词汇表、特殊 token 配置等
    final_dir = Path(config.output_dir) / "final"
    trainer.model.save_pretrained(final_dir)    # 保存 LoRA 适配器权重
    tokenizer.save_pretrained(final_dir)         # 保存分词器
    print(f"✅ 训练完成！LoRA 权重保存至: {final_dir}")

    return trainer


# ============================================================
# CLI 命令行入口
# ============================================================
# 用 argparse 解析命令行参数，覆盖 TrainConfig 的默认值
# 示例: python -m src.training.sft_train --model Qwen/Qwen3-14B --epochs 1 --max_samples 500

def parse_args():
    """解析命令行参数"""
    p = argparse.ArgumentParser(description="Qwen3 SFT 微调")
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B", help="基座模型名或路径")
    p.add_argument("--epochs", type=int, default=3, help="训练轮数")
    p.add_argument("--max_samples", type=int, default=None, help="限制样本数（快速验证）")
    p.add_argument("--lora_rank", type=int, default=64, help="LoRA 秩")
    p.add_argument("--deepspeed", action="store_true", help="启用 DeepSpeed")
    p.add_argument("--deepspeed_stage", type=int, default=2, help="ZeRO stage (2/3)")
    p.add_argument("--output_dir", default="models/lora_adapters", help="输出目录")
    p.add_argument("--train_file", default="data/sft/train.jsonl", help="训练数据")
    p.add_argument("--val_file", default="data/sft/val.jsonl", help="验证数据")
    p.add_argument("--local_rank", type=int, default=-1, help="DeepSpeed 自动注入，不要手动设置")  # DeepSpeed 兼容
    p.add_argument("--attn_implementation", default="flash_attention_2", help="flash_attention_2/sdpa/eager")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 将命令行参数填入 TrainConfig
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
        attn_implementation=args.attn_implementation,
    )

    train(config)
