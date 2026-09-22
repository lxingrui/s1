# 在 runpod 上运行 s1.1-32B Budget Forcing 评测

> 完整操作步骤(从零到结果、验收标准、排错速查)见 **[RUNBOOK.md](RUNBOOK.md)**;本文只讲 runpod 环境专项。

本文档对应环境:**`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`**(CUDA 12.8.1 / PyTorch 2.8.0 / Ubuntu 24.04),硬件 **2 × A100 SXM 80GB**。

---

## 0. 前置检查

| 项 | 要求 |
|---|---|
| GPU | 2 × A100 SXM 80GB(数据并行)或用张量并行 |
| 磁盘 | ≥ 100GB(模型 ≈ 64GB + 缓存)。建议挂 network volume 到 `/workspace` |
| 网络 | 能访问 HuggingFace 与 `open.feishu.cn` |
| Python | 镜像自带即可(脚本不会覆盖 torch/CUDA) |

---

## 1. 准备代码与密钥

```bash
git clone <你的仓库地址> s1 && cd s1
cp config/.env.example config/.env
vim config/.env
```

`config/.env` 至少要填:

```dotenv
GRADING_BASE_URL="https://api.deepseek.com/v1"   # 你的自定义打分端点
GRADING_API_KEY="sk-xxxx"
GRADING_MODEL="deepseek-chat"

FEISHU_WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/xxxx"
FEISHU_SECRET="xxxx"

HF_HOME="/workspace/hf_cache"
S1_OUTPUT_DIR="/workspace/outputs"
```

> `config/.env` 已被 `.gitignore` 忽略,不会进仓库。

---

## 2. 安装依赖(只补缺包)

```bash
bash scripts/run_eval.sh --install
```

该命令的行为:

1. 逐个检查 `pyyaml / openai / datasets / transformers / accelerate / hf_transfer`,**只安装缺失的**;
2. 检查 `vllm`,缺失时尝试 `pip install vllm`(由 pip 自动匹配 cu128 + torch 2.8);
3. **绝不执行 `pip install -r requirements.txt`**,以免把镜像自带的 torch 降级、破坏 CUDA 兼容;
4. 若 vLLM 装不上,评测会自动回退到 `transformers` 后端(慢一些但能跑通)。

---

## 3. 并行策略(双 A100 提速)

`config/config.yaml`:

```yaml
model:
  parallel: "data"          # data = 双卡数据并行(推荐先用这个)
  num_gpus: 2               # 使用 2 张卡
  tensor_parallel_size: 2   # 仅 parallel=tensor 时生效
```

| 模式 | 含义 | 适用 |
|---|---|---|
| `data` | **每张卡各放一份完整 32B 副本**,样本对半分,速度 ≈ ×卡数 | 首选;单卡 80GB 能放下 64GB 权重 + KV 时 |
| `tensor` | 单份模型切到 2 张卡 | `data` **爆显存(OOM)** 时切换,省显存但有通信开销 |

切换方式(二选一):

```bash
# 临时切换(不改文件)
bash scripts/run_eval.sh --parallel tensor

# 或改 config.yaml
#   model.parallel: "tensor"
```

> 数据并行下代码会**自动把每个 worker 的 `tensor_parallel_size` 设为 1**(每卡一份副本),无需手动改。
> 只用单卡时把 `model.num_gpus` 设为 `1`(或 `0` 自动检测)。

---

## 4. 运行

```bash
# ① 冒烟测试:每个 task 只取 2 条,验证模型能加载、能生成
bash scripts/run_eval.sh --dry-run --limit 2

# ② 正式评测
bash scripts/run_eval.sh

# ③ 一键(跑完自动用 runpodctl 停机,停止计费)
bash auto_run_and_kill.sh
```

跑完后也可手动停机:

```bash
runpodctl stop pod "$RUNPOD_POD_ID"
```

---

## 5. 期望的返回结果

### 5.1 控制台日志(示例,时间戳省略)

```
s1 Budget Forcing 评测启动（数据并行 × 2）| run_id=20260922_153000
输出目录: /workspace/outputs/run_20260922_153000
飞书通知: 开启
[parallel] 启动 shard 0/2（CUDA_VISIBLE_DEVICES=0）
[parallel] 启动 shard 1/2（CUDA_VISIBLE_DEVICES=1）
...
合并完成，共 530 条 → /workspace/outputs/run_20260922_153000/results.jsonl
全部完成。
```

各 shard 的日志在 `shard0.log` / `shard1.log`:

```
[shard 0/2] 启动（CUDA_VISIBLE_DEVICES=0）
[data] aime24: 共 30 条样本
[data] math500: 共 500 条样本
[shard 0/2] 分到 265/530 条样本
[bf] 已用 vLLM 后端加载模型。
[grading] 使用自定义打分端点: https://... | model=...
开始生成，共 265 条样本...
生成完成，用时 42.3 分钟。
  [1/265] task=aime24 grade=Yes
  [2/265] task=aime24 grade=No
  ...
[shard 0/2] 完成：265 条，生成用时 42.3 分钟
```

### 5.2 输出目录结构

```
/workspace/outputs/run_<时间戳>/
├── run.log          # 父进程日志
├── shard0.log       # 数据并行时,各 worker 日志
├── shard1.log
├── shard0.jsonl     # 数据并行时,各 worker 中间结果
├── shard1.jsonl
├── results.jsonl    # 合并后的最终结果(每行一条样本)
└── summary.json     # 汇总指标
```

### 5.3 `results.jsonl`(每行一个 JSON)

```json
{
  "task": "aime24",
  "index": 0,
  "question": "原始题目…",
  "reference": "42",
  "thinking": "模型思考链(含被强塞的 Wait)…",
  "prediction": "模型的最终答案…",
  "grade": "Yes",
  "reason": "打分模型给出的理由",
  "backend": "vllm"
}
```

- `grade` ∈ `{"Yes", "No", null}`;`null` 表示打分失败或无参考答案。
- `thinking` 里应能看到被强塞的 `Wait`(Budget Forcing 生效的证据)。

### 5.4 `summary.json`

```json
{
  "by_task": {
    "aime24":  {"total": 30,  "correct": 16, "accuracy": 0.5333},
    "math500": {"total": 500, "correct": 446, "accuracy": 0.892}
  },
  "overall": {"total": 530, "correct": 462, "accuracy": 0.8717},
  "run_id": "20260922_153000",
  "model": "simplescaling/s1.1-32B",
  "budget_forcing": {
    "enabled": true, "max_tokens_thinking": 32000, "num_ignore": 1, "ignore_str": "Wait"
  },
  "grader": "llm",
  "parallel": "data x2",
  "num_samples": 530,
  "elapsed_seconds": 5400.0
}
```

> 单进程模式还会多一个 `gen_seconds`(纯生成耗时);数据并行模式用 `elapsed_seconds` 表示总耗时。

### 5.5 参考量级与验收标准

**验收标准(全部应满足):**

1. 进程 **exit code = 0**;
2. `results.jsonl` **行数 = 各 task 样本数之和**(默认 30 + 500 = 530);
3. `summary.json` 的 `overall.accuracy` 落在参考区间;
4. 飞书按顺序收到 **start → (progress) → complete**;
5. `results.jsonl` 中 `grade` 基本都为 `Yes`/`No`(不是大量 `null`)。

**参考准确率区间**(s1.1-32B + Budget Forcing):

| 任务 | 样本数 | 参考准确率 |
|---|---|---|
| `aime24` | 30 | 40% – 60% |
| `math500` | 500 | 85% – 93% |

> 说明:具体数值取决于 `num_ignore`、`temperature`、以及所用打分模型;本仓库是独立复现
> (非官方 lm-eval-harness),与官方结果可能有几个百分点的差异。
> `num_ignore` 越大(如 8),AIME 类难题通常越好但更慢。官方结果见 HF
> `simplescaling/results`,以官方为准。

### 5.6 结果自检

| 现象 | 可能原因 |
|---|---|
| `grade` 大量为 `null` | 打分端点不通/密钥错误(`GRADING_*`) |
| `thinking` 极短、几乎全 `No` | Budget Forcing 未生效,或显存不足导致被截断 |
| 缺少 `summary.json` / 结果不全 | 某个 shard 失败,查看 `shard*.log` |
| `accuracy` 明显偏低 | 打分模型判定过严,或 `temperature > 0` 导致不稳 |

---

## 6. 常见问题

| 问题 | 处理 |
|---|---|
| **OOM(爆显存)** | 切张量并行:`--parallel tensor`;或调小 `model.max_model_len`、`model.gpu_memory_utilization` |
| **vLLM 装不上** | 忽略即可,会自动回退 `transformers` 后端 |
| **HF 下载慢/超时** | `.env` 里设 `HF_ENDPOINT="https://hf-mirror.com"` |
| **`hf_transfer` 报错** | `pip install hf_transfer`,或注释掉 `HF_HUB_ENABLE_HF_TRANSFER` |
| **飞书没有消息** | 检查 `FEISHU_WEBHOOK`;手测 `python send_feishu.py "测试"` |
| **想跑完后看日志再关机** | 设 `S1_PAUSE_ON_EXIT=1` |
| **runpod 不停机** | 容器内 `shutdown -h now` 无效,必须用 `runpodctl stop pod $RUNPOD_POD_ID` |

---

## 7. 时间预估(粗略)

| 阶段 | 说明 |
|---|---|
| 首次下载模型(64GB) | 数分钟 ~ 20 分钟,取决于带宽;有 network volume 缓存后重跑免下载 |
| 加载模型到 vLLM | 1 ~ 3 分钟 |
| AIME24 + MATH500(530 条) | 生成长度差异很大,双卡数据并行大约 **1 ~ 3 小时** |

实际耗时主要由**生成长度**决定(预算强制会显著增加思考 token 数)。
