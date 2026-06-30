# AutoDL 远程微调全流程指南

## 策略

**训练**: AutoDL A100 → Qwen3-30B-A3B（云端算力）  
**推理**: MacBook Air M4 → Qwen3-14B GGUF（本地免费）

训练数据和 SFT 样本通用，模型切换只需改一行参数。

---

## 一、AutoDL 租用配置

| 配置项 | 推荐值 |
|--------|--------|
| 显卡 | **1× A100 80GB**（备选: 1× RTX 4090 24GB） |
| 显存/内存 | 80GB / 128GB+ |
| 系统盘 | 50GB 系统盘（免费）+ **100GB 数据盘**（存模型和数据集） |
| 计费 | ~¥4-5/小时（A100），~¥1.5-2/小时（4090） |
| 镜像 | PyTorch 2.5+ / CUDA 12.4 / Python 3.10 |

**选区域技巧**: 北京/内蒙古区价格最低，延迟对训练无影响。

**预估成本**: 
- A100: 约 8-12 小时训练 × ¥5 = **¥40-60**（训练）+ ¥2/小时 × 数据上传/下载 ≈ **¥50-80**
- 4090: 约 12-18 小时 × ¥2 = **¥25-40**

---

## 二、Docker 方案分析与推荐

### 可行性：✅ 完全可行，且强烈推荐

AutoDL 原生支持 Docker 镜像。好处：

| 无 Docker | 用 Docker |
|-----------|----------|
| 每次租新实例都要重装依赖（~20 分钟） | 开箱即用 |
| 版本冲突风险高（CUDA/PyTorch/transformers 不兼容） | 环境一次配好，永久复用 |
| 项目无法迁移到其他云平台 | 任何支持 Docker 的平台都能跑 |

### Dockerfile

在项目根目录创建 `Dockerfile`：

```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# 系统依赖
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Python 依赖（分层构建，利用 Docker 缓存）
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 额外: 训练必需（部分已包含在 requirements.txt 中）
RUN pip install --no-cache-dir \
    transformers>=4.50.0 \
    peft \
    accelerate \
    deepspeed \
    trl>=0.12.0 \
    bitsandbytes \
    wandb \
    sentencepiece

# 设置工作目录
WORKDIR /workspace/novel-llm

# 环境变量
ENV WANDB_MODE=offline
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HOME=/workspace/hf_cache
```

### 构建和推送镜像

```bash
# 本地构建（或直接在 AutoDL 实例上构建，网络更快）
docker build -t novel-llm:latest .

# 推送到阿里云容器镜像（免费）
# 1. 注册 https://cr.console.aliyun.com
# 2. 创建命名空间和仓库
docker tag novel-llm:latest registry.cn-beijing.aliyuncs.com/<命名空间>/novel-llm:latest
docker push registry.cn-beijing.aliyuncs.com/<命名空间>/novel-llm:latest

# 在 AutoDL 创建实例时填入镜像地址即可
```

> 如果不想折腾 Docker 推送，也可以在 AutoDL 租好实例后上传 Dockerfile 直接构建，云服务器网络快（100Mbps+），构建只需 5-10 分钟。

---

## 三、操作全流程

### 第 1 步: 打包项目

在本地 MacBook 上：

```bash
# 排除大文件（模型权重、raw 数据）
tar --exclude='.venv' \
    --exclude='models/base' \
    --exclude='models/lora_adapters_test' \
    --exclude='data/rag' \
    --exclude='__pycache__' \
    -czf novel-llm.tar.gz \
    configs/ \
    data/sft/ \
    data/clean/*/llm_annotations.json \
    data/clean/*/metadata.json \
    src/ \
    scripts/ \
    requirements.txt \
    .env
```

打包后文件大小约 50-100MB（含 SFT 数据和标注），上传到 AutoDL。

### 第 2 步: 创建 AutoDL 实例

1. 登录 [AutoDL](https://www.autodl.com)
2. 选择「GPU 实例」→ 单卡 A100 80GB
3. 选择镜像: 「PyTorch 2.5.1 / CUDA 12.4 / Python 3.10」
4. 数据盘扩容至 **100GB**
5. 创建并启动

### 第 3 步: 上传项目

```bash
# 在 AutoDL 提供的 JupyterLab 终端中:
cd /root/autodl-tmp

# 方式 A: scp 上传（Mac 本地终端执行）
scp -P <SSH端口> novel-llm.tar.gz root@<AutoDL_IP>:/root/autodl-tmp/

# 方式 B: 通过 AutoDL 网盘上传（Web界面拖拽）
# 方式 C: git clone（如果你的 GitHub 仓库已包含 SFT 数据）
git clone https://github.com/LeoP-Hayes/novel-llm.git

# 解压
tar -xzf novel-llm.tar.gz -C novel-llm/
```

### 第 4 步: 安装环境

```bash
cd /root/autodl-tmp/novel-llm

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 如果用 Docker: 跳过安装依赖，直接启动容器
# docker run -it --gpus all -v /root/autodl-tmp/novel-llm:/workspace/novel-llm <镜像名>
```

### 第 5 步: 下载基座模型

```bash
# 设置 HF 国内镜像加速
export HF_ENDPOINT=https://hf-mirror.com

# 下载 Qwen3-30B-A3B（约 60GB）
huggingface-cli download Qwen/Qwen3-30B-A3B \
  --local-dir /root/autodl-tmp/models/Qwen3-30B-A3B \
  --resume-download

# 同时下载 14B（后面推理用，约 28GB）
huggingface-cli download Qwen/Qwen3-14B \
  --local-dir /root/autodl-tmp/models/Qwen3-14B \
  --resume-download
```

> AutoDL 实例内网带宽高（100MB/s+），60GB 模型下载约 10 分钟。

### 第 6 步: 启动训练

```bash
# 确认 GPU 可用
nvidia-smi

# 确认显存
# A100 80GB: bf16 模型(60GB) + LoRA(5GB) + 优化器(10GB) ≈ 75GB ✅

# A100 80GB: DeepSpeed ZeRO-2
deepspeed --num_gpus=1 src/training/sft_train.py \
  --model /root/autodl-tmp/models/Qwen3-30B-A3B \
  --epochs 3 \
  --deepspeed \
  --deepspeed_stage 2 \
  --output_dir /root/autodl-tmp/output/lora_adapters

# RTX 4090 24GB: 必须 QLoRA 4bit
python -m src.training.sft_train \
  --model /root/autodl-tmp/models/Qwen3-30B-A3B \
  --epochs 3 \
  --lora_rank 32 \
  --max_samples 0 \
  --output_dir /root/autodl-tmp/output/lora_adapters
# 注意: RTX 4090 需要修改 TrainConfig 中 use_4bit=True
```

**训练监控**:
```bash
# 另开一个终端
watch -n 5 nvidia-smi

# 如果有 wandb，可以在本地浏览器查看
# 训练日志在 output/lora_adapters/logs/
```

### 第 7 步: 训练完成后的操作

```bash
# 合并 LoRA 到基座（可选，方便推理部署）
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_path = '/root/autodl-tmp/models/Qwen3-30B-A3B'
lora_path = '/root/autodl-tmp/output/lora_adapters/final'
output_path = '/root/autodl-tmp/output/merged_model'

model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()
model.save_pretrained(output_path)
tokenizer = AutoTokenizer.from_pretrained(base_path)
tokenizer.save_pretrained(output_path)
print('✅ 模型已合并')
"

# 下载到本地 MacBook
# 在 Mac 上执行:
scp -r -P <SSH端口> root@<AutoDL_IP>:/root/autodl-tmp/output/lora_adapters/final/ ./models/lora_adapters/final/
```

### 第 8 步: 本地 MacBook 推理

训练产出的 **LoRA 适配器**（约 200MB）下载到本地后，转到 14B 推理：

```bash
# 1. 下载 Qwen3-14B GGUF 量化版（推荐 Q4_K_M, 约 8GB）
# 从 huggingface.co/bartowski/Qwen3-14B-GGUF 下载

# 2. 安装 llama.cpp
brew install llama.cpp

# 3. 对 14B 做 SFT（复用同一份训练数据，仅换模型名）
python -m src.training.sft_train \
  --model Qwen/Qwen3-14B \
  --epochs 3 \
  --output_dir models/lora_adapters_14b

# 4. 使用微调后的 14B 生成小说
# （等 Phase 4 约束生成系统和 Phase 6 CLI Demo 实现后）
```

---

## 四、Docker vs 纯虚拟环境 对比

| | 纯虚拟环境 | Docker |
|---|---|---|
| 初始配置耗时 | 20-30 分钟（每次重新配） | 0 分钟（镜像拉取即用） |
| 环境可复现性 | 一般（pip 版本可能漂移） | 完全一致 |
| 打包大小 | N/A | ~8-15GB（含 CUDA 基础镜像） |
| 跨平台迁移 | 需重新配环境 | 任何支持 Docker + GPU 的平台 |
| 学习成本 | 低 | 中（需了解 Dockerfile 和镜像推送） |
| **推荐** | 如果只租 1-2 次 | **如果多次租用或跨平台** |

---

## 五、省钱技巧

1. **关机不收费**: AutoDL 按实例运行时间计费，不用时关机（数据盘保留，不收费）
2. **先用小模型验证**: 本地 Qwen2.5-0.5B 跑通全流程 → 4090 跑 Qwen3-14B 验证 → A100 跑 30B 正式训练
3. **下载用国内镜像**: `export HF_ENDPOINT=https://hf-mirror.com` 避免 HuggingFace 限速
4. **数据盘扩缩容**: 训练前扩到 100GB 存模型，训练后缩到 20GB 存关键产出即可
5. **A100 选竞价实例**: 如果有竞价实例（spot），价格低 30-50%，训练中断可恢复（有 checkpoint）

---

## 六、时间预估

| 步骤 | 耗时 |
|------|------|
| 环境安装 | 10 分钟（Docker）/ 20 分钟（手动） |
| 模型下载（60GB） | 10 分钟（内网高速） |
| 数据上传（100MB） | 1 分钟 |
| 训练（6,000 × 3 epoch） | 8-12 小时（A100）/ 12-18 小时（4090） |
| 模型下载到本地 | 5 分钟（200MB LoRA） |
| **总计** | **约 10-13 小时** |

---

## 七、故障排查

| 常见问题 | 解决方案 |
|---------|---------|
| CUDA Out of Memory | 降 batch_size=1，开启 gradient_checkpointing，检查是否有其他进程占用显存 (`fuser -v /dev/nvidia*`) |
| NCCL Timeout | 单卡训练通常不是 NCCL 问题，检查 `--num_gpus` 参数 |
| HF 模型下载慢 | 换国内镜像: `export HF_ENDPOINT=https://hf-mirror.com` |
| DeepSpeed 报错 | 检查 CUDA 版本与 PyTorch 匹配，或降级用 ZeRO-2 |
| AutoDL 实例无法 SSH | 在控制台重启实例，或通过 Web 终端登录 |

---

## 下一步

完成训练后，本地 MacBook 部署 Qwen3-14B GGUF 推理，使用本项目的约束生成系统（Phase 4）和 CLI Demo（Phase 6），即可生成完整的都市文娱小说。
