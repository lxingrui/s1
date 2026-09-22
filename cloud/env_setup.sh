#!/bin/bash
set -e

echo "=== [1/4] 安装 uv 工具链 ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

echo "=== [2/4] 创建隔离虚拟环境 ==="
uv venv --python 3.10
source .venv/bin/activate

echo "=== [3/4] 安装 PyTorch 与核心底层 (根据显卡架构匹配) ==="
# 租用 A100/H100/H200 等企业卡默认直接拉取成熟的 cu124 构建即可
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install "transformers>=4.46.0" datasets accelerate trl vllm peft wandb hf_transfer

echo "=== [4/4] 安装 Flash-Attention ==="
uv pip install flash-attn --no-build-isolation

echo "=== 环境构建完毕！当前激活环境: $(which python) ==="