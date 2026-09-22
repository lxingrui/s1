#!/usr/bin/env bash
# =============================================================================
# runpod 一键启动脚本
# -----------------------------------------------------------------------------
# 跳过 SFT 训练，直接拉取 simplescaling/s1-32B，跑原版 Budget Forcing 评测。
# 内置 watchdog（超时自动终止）+ 飞书通知（开始/进度/完成/报错/超时）。
#
# 用法：
#   bash scripts/run_eval.sh                 # 直接跑
#   bash scripts/run_eval.sh --install       # 先装依赖再跑
#   bash scripts/run_eval.sh --limit 5       # 只跑 5 条
#   bash scripts/run_eval.sh --dry-run       # 冒烟测试
# 所有额外参数都会透传给 s1_eval.run_eval
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# ---------------------------- 加载环境变量 ----------------------------
if [ -f "config/.env" ]; then
    echo ">>> 加载 config/.env"
    set -a
    # shellcheck disable=SC1091
    . config/.env
    set +a
else
    echo ">>> 未找到 config/.env（可从 config/.env.example 复制）"
fi

# ---------------------------- runpod / 缓存路径 ----------------------------
# runpod 的持久化盘一般是 /workspace；容器盘是 /root
if [ -z "${HF_HOME:-}" ]; then
    if [ -d "/workspace" ]; then
        export HF_HOME="/workspace/hf_cache"
    else
        export HF_HOME="${PROJECT_ROOT}/.hf_cache"
    fi
fi
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
mkdir -p "${HF_HOME}"
echo ">>> HF_HOME=${HF_HOME}"

# 让 watchdog 的进程组终止生效（脚本自身成为进程组组长）
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

CONFIG_PATH="${S1_CONFIG:-config/config.yaml}"

# ---------------------------- 可选：安装依赖 ----------------------------
INSTALL=0
ARGS=()
for arg in "$@"; do
    if [ "${arg}" = "--install" ]; then
        INSTALL=1
    else
        ARGS+=("${arg}")
    fi
done

if [ "${INSTALL}" = "1" ]; then
    echo ">>> 安装依赖（runpod 建议：已自带 CUDA 镜像时优先装 wheel）..."
    python -m pip install --upgrade pip
    # PyTorch 底座：按需选择 CUDA 版本
    python -m pip install "torch>=2.5" --index-url https://download.pytorch.org/whl/cu124 || true
    python -m pip install -r requirements.txt
    # Budget Forcing 原版用 vLLM，若镜像未内置则安装
    python -m pip install vllm || echo ">>> vLLM 安装失败，将自动回退 transformers 后端"
    python -m pip install pyyaml openai
fi

# ---------------------------- 校验依赖 ----------------------------
python - <<'PY'
import importlib
for mod in ("yaml", "datasets", "transformers", "openai"):
    try:
        importlib.import_module(mod)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"缺少依赖 {mod}: {exc}\n请用 --install 或手动安装。")
print(">>> 依赖检查通过")
PY

# ---------------------------- 启动评测 ----------------------------
echo ">>> 启动评测： config=${CONFIG_PATH}"
echo ">>> 附加参数： ${ARGS[*]:-（无）}"

python -m s1_eval.run_eval --config "${CONFIG_PATH}" ${ARGS[@]+"${ARGS[@]}"}
