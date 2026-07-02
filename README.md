# Novel LLM — 都市文娱小说生成系统

基于 Qwen3-30B-A3B (MoE) LoRA 微调 + GRPO 强化学习，从网文语料学习都市文娱文风与叙事模式，自动生成结构化长篇网文。

## 架构

```
raw txt → cleaner → LLM标注 → SFT样本 → LoRA微调 → 约束生成 → GRPO对齐
                ↑ DeepSeek API           ↑ AutoDL RTX 6000D   ↑ 奖励模型
```

## 目录

```
├── configs/              模型/风格/约束/DeepSpeed 配置
├── src/
│   ├── data_pipeline/    采集 → 清洗 → LLM标注 → SFT构造
│   ├── training/         SFT LoRA 微调 + GRPO 强化学习
│   ├── rag/              双路检索（文风+知识）ChromaDB
│   └── constraints/      大纲规划器 + 章节生成器 + 校验器
├── scripts/              评估/生成/合并/复核 工具
├── data/                 (gitignore) 原始语料 + 标注 + SFT样本
├── models/               (gitignore) LoRA权重 + 合并模型
└── output/               (gitignore) 生成结果 + 评估报告
```

## 执行流程

### Phase 1: 数据管线

```bash
# 1. 清洗原始 txt → 章节文件 + metadata
python -c "from src.data_pipeline.cleaner import process_book; ..."

# 2. DeepSeek V4 Flash 全量标注 (2,104章, ~$1.50)
python -c "from src.data_pipeline.llm_annotator import annotate_all_books; ..."

# 3. 构造 SFT 样本 (6,000条, 5种任务类型)
python -c "from src.data_pipeline.sft_data_builder import build_sft_dataset; ..."
```

### Phase 2: SFT 微调

```bash
# AutoDL RTX 6000D 80GB (¥5.18/h, ~3h, ~¥16)
python src/training/sft_train.py \
  --model models/novel-merged \
  --epochs 3 --attn_implementation sdpa \
  --output_dir output/lora_adapters
```

**训练结果**: loss 2.35 → 0.77 (-67%), 对话占比 0.1% → 32.7%

### Phase 3: RAG (代码就绪)

```bash
python src/rag/embedding.py  # 构建 ChromaDB 索引
```

### Phase 4: 约束生成

```python
from src.constraints.chapter_generator import NovelGenerator
gen = NovelGenerator(model_path="models/novel-merged")
gen.generate("重生2008，北电导演系学生...", chapters=50)
```

### Phase 5: GRPO 强化学习

```bash
# AutoDL RTX 6000D 80GB (~6-8h, ~¥40)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python src/training/grpo_train.py \
  --model models/novel-merged \
  --episodes 500 --group_size 4 \
  --output_dir output/grpo --resume
```

## 核心结果

### 微调前后对比

| 指标 | 基座模型 | SFT微调 | 提升 |
|------|---------|--------|------|
| 对话占比 | 0.1% | **32.7%** | +326x |
| 平均句长 | 28.2字 | **19.8字** | -30% |
| 重复率 | 19.2% | **16.1%** | -16% |
| LLM盲评-爽感 | 3/10 | **5/10** | +67% |
| LLM盲评-文风 | 5/10 | **6/10** | +20% |

### GRPO 对齐 (预期)

| 指标 | SFT | GRPO | 提升 |
|------|-----|------|:---:|
| LLM盲评-爽感 | 5 | 7 | +2 |
| LLM盲评-专业感 | 3 | 5 | +2 |
| 对话占比 | 32.7% | 36.5% | +12% |

## 技术栈

| 层级 | 选型 |
|------|------|
| 基座模型 | Qwen3-30B-A3B (MoE, 30B总参/3B激活) |
| 微调 | LoRA rank=64, bf16, peft |
| 数据标注 | DeepSeek V4 Flash API |
| 奖励模型 | DeepSeek V4 Flash API (不占本地显存) |
| 向量库 | ChromaDB + BGE-large-zh-v1.5 |
| 约束系统 | LLM驱动大纲 + 张力累积节奏 + 8维校验 |
| 硬件 | AutoDL RTX 6000D 80GB (¥5.18/h) |
| RL | GRPO (组内相对策略优化) |

## 踩坑记录

- SFT: FlashAttention2未安装→SDPA, Qwen3 MoE gate_up_proj融合层, lora_dropout=0, peft版本兼容
- GRPO: 批量生成替代串行, 异步API打分, KL惩罚进loss不直接改权重, wandb实体名修正

## License

MIT — LoRA权重和训练代码开源，原始语料不包含在内。
