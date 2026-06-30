FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# 系统依赖
RUN apt-get update && apt-get install -y git vim && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir sentencepiece deepspeed

# 工作目录
WORKDIR /workspace/novel-llm

# 环境变量
ENV WANDB_MODE=offline
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HOME=/workspace/hf_cache
ENV HF_ENDPOINT=https://hf-mirror.com

# 入口
CMD ["/bin/bash"]
