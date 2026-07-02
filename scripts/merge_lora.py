"""
合并 LoRA 权重到基座模型
运行: python scripts/merge_lora.py
产出: /root/autodl-tmp/novel-llm/models/novel-merged/
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/root/autodl-tmp/novel-llm/models/Qwen3-30B-A3B"
LORA = "/root/autodl-tmp/novel-llm/output/lora_adapters/checkpoint-1014"
OUTPUT = "/root/autodl-tmp/novel-llm/models/novel-merged"

# 1. 加载基座
print("加载基座模型...")
tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    trust_remote_code=True, device_map="auto"
)

# 2. 加载 LoRA 适配器
print("加载 LoRA 适配器...")
from peft import PeftModel
try:
    model = PeftModel.from_pretrained(model, LORA)
except TypeError:
    print("peft 版本不兼容，安装兼容版本...")
    import subprocess
    subprocess.run(["pip", "install", "peft==0.13.0", "-q"])
    from peft import PeftModel as PM
    model = PM.from_pretrained(model, LORA)

# 3. 合并：LoRA 矩阵 A×B 融入原始权重 W
print("合并权重 (merge_and_unload)...")
model = model.merge_and_unload()

# 4. 保存
print(f"保存合并模型到 {OUTPUT} ...")
model.save_pretrained(OUTPUT)
tokenizer.save_pretrained(OUTPUT)
print("✅ 合并完成！")
