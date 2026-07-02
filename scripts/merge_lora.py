"""
合并 LoRA 权重到基座模型

用法:
    python scripts/merge_lora.py
    python scripts/merge_lora.py --base models/base/Qwen3-30B-A3B \\
                                  --lora models/lora_adapters/checkpoint-1014 \\
                                  --output models/novel-merged
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA 权重到基座模型")
    parser.add_argument("--base", default="models/base/Qwen3-30B-A3B", help="基座模型路径")
    parser.add_argument("--lora", default="models/lora_adapters/checkpoint-1014", help="LoRA 适配器路径")
    parser.add_argument("--output", default="models/novel-merged", help="合并后输出路径")
    args = parser.parse_args()

    # 1. 加载基座
    print("加载基座模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True, device_map="auto"
    )

    # 2. 加载 LoRA 适配器
    print("加载 LoRA 适配器...")
    model = PeftModel.from_pretrained(model, args.lora)

    # 3. 合并：LoRA 矩阵 A×B 融入原始权重 W
    print("合并权重 (merge_and_unload)...")
    model = model.merge_and_unload()

    # 4. 保存
    print(f"保存合并模型到 {args.output} ...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("✅ 合并完成！")


if __name__ == "__main__":
    main()
