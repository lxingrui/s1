#!/usr/bin/env bash
# =============================================================================
# runpod 一键启动脚本
# -----------------------------------------------------------------------------
# 跳过 SFT 训练，直接拉取 simplescaling/s1.1-32B，跑原版 Budget Forcing 评测。
# 支持双卡数据并行（config.yaml: model.parallel=data），爆显存可切 tensor 张量并行。
# 内置 watchdog（超时自动终止）+ 飞书通知（开始/进度/完成/报错/超时）。
#
# 用法：
#   bash scripts/run_eval.sh                       # 直接跑
#   bash scripts/run_eval.sh --install             # 只补装缺失依赖（不动镜像自带 torch）
#   bash scripts/run_eval.sh --limit 5             # 只跑 5 条
#   bash scripts/run_eval.sh --dry-run             # 冒烟测试
#   bash scripts/run_eval.sh --parallel tensor     # 临时切张量并行
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

# RunPod Global Volume (FUSE.GEESEFS) 等网络/融合文件系统不支持 POSIX 符号链接，
# 会把 HF 缓存里的软链接变成 0 字节死链（config.json → JSONDecodeError）。
# 检测到这类文件系统时，让 HF 直接复制实体文件，不再建软链接。
if [ -z "${HF_HUB_DISABLE_SYMLINKS:-}" ]; then
    FS_TYPE="$(stat -f -c %T "${HF_HOME}" 2>/dev/null || echo unknown)"
    case "$(echo "${FS_TYPE}" | tr 'A-Z' 'a-z')" in
        *fuse*|*nfs*|*smb*|*cifs*|*9p*|*geese*)
            export HF_HUB_DISABLE_SYMLINKS=1
            echo ">>> 检测到网络/融合文件系统(${FS_TYPE})：已设 HF_HUB_DISABLE_SYMLINKS=1（缓存改为实体文件）"
            ;;
    esac
fi
export HF_HUB_DISABLE_SYMLINKS="${HF_HUB_DISABLE_SYMLINKS:-0}"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo ">>> 检测到 GPU："
    nvidia-smi -L 2>/dev/null | sed 's/^/    /' || true
    echo ">>> 可见 GPU 数：$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ') "
fi

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
    echo ">>> 只补装缺失依赖（绝不覆盖镜像自带的 torch / CUDA）..."

    # ---- 按 CUDA 驱动版本约束 torch，防止被升级到 cu13 而驱动不支持 ----
    CUDA_VER="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
    CUDA_MAJOR="${CUDA_VER%%.*}"
    CONSTRAINTS="/tmp/s1_pip_constraints.txt"
    : > "${CONSTRAINTS}"
    if [ "${CUDA_MAJOR}" = "12" ]; then
        echo "torch<2.10.0" >> "${CONSTRAINTS}"
        echo ">>> 驱动 CUDA ${CUDA_VER}：已锁定 torch<2.10.0（禁止升级到 cu13 版本）"
    elif [ "${CUDA_MAJOR}" = "13" ]; then
        echo "torch>=2.10.0" >> "${CONSTRAINTS}"
        echo ">>> 驱动 CUDA ${CUDA_VER}：允许新版 torch"
    else
        echo ">>> 未检测到明确的 CUDA 版本，不做 torch 版本约束"
    fi
    [ -s "${CONSTRAINTS}" ] || rm -f "${CONSTRAINTS}"
    export S1_PIP_CONSTRAINTS="${CONSTRAINTS}"

    python - <<'PY'
import importlib.util
import os
import subprocess
import sys

constraints = os.environ.get("S1_PIP_CONSTRAINTS", "")
has_constraints = bool(constraints) and os.path.exists(constraints)
pip = [sys.executable, "-m", "pip", "install"]
if has_constraints:
    pip += ["-c", constraints]
    print(">>> 使用版本约束文件:", constraints)

# 运行 s1_eval 的最小依赖：模块名 -> pip 包名
NEED = {
    "yaml": "pyyaml",
    "openai": "openai",
    "datasets": "datasets",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "hf_transfer": "hf_transfer",
}
missing = [pkg for mod, pkg in NEED.items() if importlib.util.find_spec(mod) is None]
if missing:
    print(">>> 缺失依赖，即将安装:", ", ".join(missing))
    subprocess.check_call(pip + missing)
else:
    print(">>> 基础依赖齐备")

# Budget Forcing 原版机制用 vLLM；必须带 torch 约束安装，
# 否则 pip 可能把 torch 升级成 cu13 版本、导致驱动不匹配。
if importlib.util.find_spec("vllm") is None:
    print(">>> 未检测到 vllm，尝试安装与当前 torch/CUDA 匹配的版本...")
    rc = subprocess.call(pip + ["vllm"])
    if rc != 0:
        print(">>> 带约束安装失败，改用 --no-deps（保持 torch 不被改动）...")
        rc = subprocess.call(
            [sys.executable, "-m", "pip", "install", "--no-deps", "vllm"]
        )
    if rc != 0:
        print(">>> vLLM 安装失败，将自动回退 transformers 后端")
else:
    print(">>> vllm 已存在，跳过")
PY
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
try:
    import torch

    print(">>> torch:", torch.__version__, "| CUDA 可用:", torch.cuda.is_available())
except Exception as exc:  # noqa: BLE001
    print(">>> [warn] torch 不可用:", exc)
try:
    import vllm

    print(">>> vllm:", getattr(vllm, "__version__", "unknown"))
except Exception:  # noqa: BLE001
    print(">>> 未安装 vllm（将使用 transformers 后端）")
PY

# ---------------------------- 启动评测 ----------------------------
echo ">>> 启动评测： config=${CONFIG_PATH}"
echo ">>> 附加参数： ${ARGS[*]:-（无）}"

python -m s1_eval.run_eval --config "${CONFIG_PATH}" ${ARGS[@]+"${ARGS[@]}"}
