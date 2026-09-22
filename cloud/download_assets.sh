#!/bin/bash
set -e

# $1: 数据盘挂载根路径 (如 AutoDL 为 /root/autodl-tmp，Vast.ai 为 /workspace)
DATA_DIR=${1:-"/workspace"}

echo ">>> 数据将存储至高性能数据盘: ${DATA_DIR}"
mkdir -p "${DATA_DIR}/hf_cache"
mkdir -p "${DATA_DIR}/models"

# 强制重定向 Hugging Face 全局下载目录
export HF_HOME="${DATA_DIR}/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1

# 引入凭证
if [ -f "cloud/.env" ]; then
    export $(cat cloud/.env | xargs)
fi

echo ">>> 开始高速拉取基座模型..."
huggingface-cli download Qwen/Qwen2.5-32B-Instruct \
    --local-dir "${DATA_DIR}/models/Qwen2.5-32B-Instruct" \
    --local-dir-use-symlinks False