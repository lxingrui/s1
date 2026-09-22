import json
import re
import urllib.request
from collections import defaultdict

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def fetch_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        return {"error": str(e)}


def inspect_pypi_pkg(pkg_name):
    data = fetch_json(f"https://pypi.org/pypi/{pkg_name}/json")
    if "error" in data:
        return {"error": data["error"]}
    info = data.get("info", {})
    version = info.get("version", "Unknown")
    requires_dist = info.get("requires_dist", []) or []

    torch_req = "无强制显式限制"
    for r in requires_dist:
        clean_r = r.split(";")[0].strip()
        if (
            clean_r.startswith("torch ")
            or clean_r.startswith("torch>=")
            or clean_r.startswith("torch==")
        ):
            torch_req = clean_r
            break

    return {"version": version, "torch_req": torch_req}


def inspect_flash_attn():
    url = "https://api.github.com/repos/Dao-AILab/flash-attention/releases/latest"
    data = fetch_json(url)
    if "error" in data:
        return {"error": data["error"]}

    tag = data.get("tag_name", "Unknown")
    assets = data.get("assets", [])

    cuda_torch_map = defaultdict(set)
    pattern = re.compile(r"cu(\d+)torch([\d\.]+)")
    for a in assets:
        name = a.get("name", "")
        match = pattern.search(name)
        if match:
            c, t = match.groups()
            cuda_torch_map[f"CUDA {c[:2]}.{c[2:]}"].add(f"torch {t}")

    return {
        "version": tag,
        "prebuilts": {k: sorted(list(v)) for k, v in sorted(cuda_torch_map.items())},
    }


def main():
    print("=" * 80)
    print("🚀 全栈 TTC (Test-Time Compute) 核心依赖生态兼容性大盘")
    print("=" * 80)

    # 1. 核心底层与推理引擎（C++/CUDA 强依赖组）
    critical_heavy_pkgs = ["vllm", "sglang", "deepspeed", "bitsandbytes", "triton"]
    print("\n📦 【核心底层与推理/训练重型库】(决定 CUDA/Torch 兼容性的生命线)")
    print(f"{'组件名称':<15}{'最新稳定版本':<18}{'绑定的 PyTorch 要求':<45}")
    print("-" * 80)

    for pkg in critical_heavy_pkgs:
        res = inspect_pypi_pkg(pkg)
        if "error" in res:
            print(f"{pkg:<15}{'获取失败':<18}{res['error']:<45}")
        else:
            print(f"{pkg:<15}{res['version']:<18}{res['torch_req']:<45}")

    # 2. Flash-Attention 独立检测
    print("\n⚡ 【Flash-Attention 预编译 Wheel 矩阵】(最容易卡死安装的硬核算子)")
    fa_res = inspect_flash_attn()
    if "error" in fa_res:
        print(f"  获取失败: {fa_res['error']}")
    else:
        print(f"  最新发布版本: {fa_res['version']}")
        matrix = fa_res.get("prebuilts", {})
        if matrix:
            for cuda, torches in matrix.items():
                print(f"    - {cuda}: 官方预编译覆盖 [{', '.join(torches)}]")
        else:
            print("    (官方 Release 正在构建或采用动态分发)")

    # 3. 符号求解、树搜索与验证工具组（纯逻辑库）
    logic_pkgs = [
        "sympy",
        "z3-solver",
        "math_verify",
        "networkx",
        "outlines",
        "trl",
        "lm-eval",
    ]
    print(
        "\n🧠 【TTC 算法、符号求解与树搜索调度库】(纯 Python/逻辑库，永远免编译零冲突)"
    )
    print(f"{'组件名称':<15}{'最新版本':<18}{'类别/用途':<45}")
    print("-" * 80)
    roles = {
        "sympy": "代数/微积分符号推导引擎",
        "z3-solver": "SMT/SAT 逻辑约束求解器",
        "math_verify": "DeepSeek-R1 / s1 官方标准答案判定库",
        "networkx": "ToT / GoT / MCTS 搜索状态树图结构管理",
        "outlines": "约束采样引擎 (强制结构化输出与标签生成)",
        "trl": "强化学习与 SFT 极简训练封装 (SFTTrainer)",
        "lm-eval": "EleutherAI 标准化基准评测套件",
    }
    for pkg in logic_pkgs:
        res = inspect_pypi_pkg(pkg)
        v = res.get("version", "Unknown")
        print(f"{pkg:<15}{v:<18}{roles.get(pkg, ''):<45}")

    print("\n" + "=" * 80)
    print("🎯 TTC 通用环境配置建议:")
    print("  1. 只要安装了支持 5070（sm_120）的 CUDA 13.0 + PyTorch 基础底座；")
    print("  2. 上表中的【纯 Python/逻辑库】可以随时在任何虚拟环境无脑安装使用；")
    print(
        "  3. 想要“随便找一个 TTC 论文都能跑”，只需在项目目录里用 uv 复制一个带基础底座的 .venv，"
    )
    print("     然后无缝接入上述组件即可！")
    print("=" * 80)


if __name__ == "__main__":
    main()
