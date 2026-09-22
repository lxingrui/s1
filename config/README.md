# 配置说明（config/）

本目录把所有「模型选择 / 超参数 / 提示词 / 打分模型 / watchdog / 通知」集中管理。

| 文件 | 说明 |
|---|---|
| `config.yaml` | 主配置：模型、数据集、Budget Forcing、打分、watchdog、飞书通知 |
| `.env` | 你的密钥（**不要提交到 git**），从 `.env.example` 复制而来 |
| `.env.example` | `.env` 的模板 |

> 完整运行指导见 [`docs/RUNBOOK.md`](../docs/RUNBOOK.md);runpod 环境专项见 [`docs/RUNPOD.md`](../docs/RUNPOD.md)。

## 快速开始

```bash
cp config/.env.example config/.env
# 编辑 config/.env 填入 GRADING_* / FEISHU_* / HF_HOME

# 查看最终生效的配置（${VAR} 已被替换）
python -m s1_eval.config config/config.yaml

# 冒烟测试（每个 task 1 条）
bash scripts/run_eval.sh --dry-run

# 正式评测
bash scripts/run_eval.sh
```

## 关键配置项

### 模型（跳过 SFT，直接拉官方模型）

```yaml
model:
  name: "simplescaling/s1.1-32B"    # 或 s1-32B / 自己的 ckpt
  tokenizer: "Qwen/Qwen2.5-32B-Instruct"
  backend: "auto"                   # auto / vllm / transformers
  parallel: "data"                  # data=双卡数据并行 / tensor=张量并行
  num_gpus: 2                       # data 并行使用的卡数；0=自动检测
  tensor_parallel_size: 2           # 仅 parallel=tensor 时生效
```

- **`data`(推荐先用)**:每张卡各放一份完整模型副本、样本切分 → 速度 ≈ ×卡数。
- **`tensor`**:单份模型切到多卡 → 显存不够(OOM)时切换。
- 临时覆盖:`bash scripts/run_eval.sh --parallel tensor`。

### Budget Forcing（强塞 "Wait"）

```yaml
budget_forcing:
  enabled: true
  max_tokens_thinking: 32000   # 思考总预算
  num_ignore: 1                # 强塞次数（NUM_IGNORE）
  ignore_str: "Wait"           # 强塞的提示词，可换 Hmm / Alternatively
  thinking_start: "<|im_start|>think"
  thinking_end: "<|im_start|>answer"
```

### 打分模型（自定义 OpenAI 兼容端点）

`config.yaml` 里的 `grading.*` 只写占位符，真实值从环境变量注入：

```yaml
grading:
  base_url: "${GRADING_BASE_URL:-}"
  api_key: "${GRADING_API_KEY:-}"
  model: "${GRADING_MODEL:-}"
  system_prompt: |
    ...
```

对应 `config/.env`：

```dotenv
GRADING_BASE_URL="https://api.deepseek.com/v1"
GRADING_API_KEY="sk-xxx"
GRADING_MODEL="deepseek-chat"
```

> `data/utils/inference_utils.py` 也已接入该端点：当打分模型名为 `custom`
> 或等于 `GRADING_MODEL` 时自动走自定义端点（`data/featurization.py` 已改为可配置）。

### Watchdog（超时自动终止）

```yaml
watchdog:
  enabled: true
  timeout_minutes: 180     # 超过后自动 SIGTERM → SIGKILL 整个进程组
```

### 飞书通知

```yaml
notify:
  feishu:
    webhook: "${FEISHU_WEBHOOK:-}"
    secret: "${FEISHU_SECRET:-}"
    events: [start, progress, complete, error, timeout]
    progress_every: 25
```

## 环境变量覆盖

不用改 `config.yaml`，用环境变量即可临时覆盖：

| 变量 | 作用 |
|---|---|
| `S1_CONFIG` | 指定配置文件路径 |
| `S1_OUTPUT_DIR` | 输出目录 |
| `S1_LIMIT` | 每个 task 最多评测条数 |
| `S1_MODEL_NAME` | 覆盖模型名 |
| `S1_TENSOR_PARALLEL_SIZE` | vLLM 张量并行 |
| `S1_PARALLEL` | 并行策略 `data` / `tensor` |
| `S1_NUM_GPUS` | data 并行使用的卡数 |
| `S1_TIMEOUT_MINUTES` | watchdog 超时 |
| `S1_BUDGET_FORCING` | `0/1` 开关 Budget Forcing |
| `S1_PAUSE_ON_EXIT` | `1` 时评测结束后不退出容器（方便看日志） |

## 输出

每次运行会在 `outputs/run_<时间戳>/` 下生成：

- `run.log`：完整日志
- `results.jsonl`：每条样本的思考链 / 答案 / 打分结果
- `summary.json`：各任务准确率与总体准确率
- （数据并行时）`shard<i>.log` / `shard<i>.jsonl`：各 GPU 的日志与中间结果
