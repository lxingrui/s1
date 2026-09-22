# s1.1-32B Budget Forcing 完整运行指导（Runbook）

> 目标：**跳过 SFT 训练**，直接拉取官方 `simplescaling/s1.1-32B`，复现原版
> **Budget Forcing（强塞 "Wait"）** 推理机制，用**自定义 OpenAI 兼容端点打分**，
> 全程有 **Watchdog 超时保护** 与 **飞书通知**。
>
> 环境专项（runpod 镜像/磁盘/停机）见 [`docs/RUNPOD.md`](RUNPOD.md)。

---

## 0. 30 秒速览（已部署好的机器上）

```bash
cd s1
cp config/.env.example config/.env && vim config/.env   # 首次才需要
bash scripts/run_eval.sh --install                      # 首次才需要
bash scripts/run_eval.sh --dry-run --limit 2 --no-notify   # 冒烟
bash scripts/run_eval.sh                                # 正式
```

---

## 1. 接下来你需要做什么（清单）

| # | 事项 | 命令 / 文件 | 状态 |
|---|---|---|---|
| 1 | 把代码推到机器上（本地改动先 commit+push） | `git clone <repo> && cd s1` | ☐ |
| 2 | 创建并填写密钥文件 | `cp config/.env.example config/.env` → `vim config/.env` | ☐ |
| 3 | 校验配置解析正确 | `python -m s1_eval.config config/config.yaml` | ☐ |
| 4 | 只补装缺失依赖 | `bash scripts/run_eval.sh --install` | ☐ |
| 5 | 冒烟测试（2 条） | `bash scripts/run_eval.sh --dry-run --limit 2 --no-notify` | ☐ |
| 6 | 正式评测 | `bash scripts/run_eval.sh` | ☐ |
| 7 | 验收结果 + 收飞书通知 | 看 `outputs/run_<ts>/summary.json` | ☐ |
| 8 | 停机（runpod） | `runpodctl stop pod "$RUNPOD_POD_ID"` | ☐ |

> **你现在只需要做第 1、2 步**，其余按顺序执行即可。

---

## 2. 步骤一：把代码放到机器上

```bash
git clone <你的仓库地址> s1
cd s1
```

- 若你的改动只在本地：先在本地 `git add -A && git commit -m "..." && git push`,再到机器上 `git clone`/`git pull`。
- `config/.env` 被 `.gitignore` 忽略,**不会**随仓库过去,所以第 3 步要在机器上重建。
- 本地专用的 `qwen0.5b/`、`.venv/` 不需要上传。

---

## 3. 步骤二：配置密钥（`config/.env`）

```bash
cp config/.env.example config/.env
vim config/.env
```

必填项:

```dotenv
# ---- 自定义打分端点（必须）----
GRADING_BASE_URL="https://api.deepseek.com/v1"
GRADING_API_KEY="sk-xxxxxxxxxxxxxxxx"
GRADING_MODEL="deepseek-chat"

# ---- 飞书通知（可留空=静默）----
FEISHU_WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/xxxxxxxx"
FEISHU_SECRET="xxxxxxxxxxxxxxxx"

# ---- 缓存/输出（runpod 建议放数据盘）----
HF_HOME="/workspace/hf_cache"
S1_OUTPUT_DIR="/workspace/outputs"
```

> ⚠️ `GRADING_*` 不填 → 打分自动降级为 `exact` 字符串匹配（会有 warning）。
> `FEISHU_*` 不填 → 通知静默关闭,评测照常跑。

**自检**:确认 `${VAR}` 已被替换(重点看 `grading` 与 `notify` 两段):

```bash
python -m s1_eval.config config/config.yaml
```

飞书连通性自测:

```bash
python send_feishu.py "配置自检：这是一条测试消息" --event info
```

---

## 4. 步骤三：安装依赖（只补缺包）

```bash
bash scripts/run_eval.sh --install
```

该命令:

1. 检查 `pyyaml / openai / datasets / transformers / accelerate / hf_transfer`,**只装缺失的**;
2. 检测 `vllm`,缺失则 `pip install vllm`(pip 自动匹配当前 torch/CUDA);装不上会自动回退 `transformers` 后端;
3. **不会执行 `pip install -r requirements.txt`**,因此不会覆盖镜像自带的 torch/CUDA。

期望输出(节选):

```
>>> 只补装缺失依赖（绝不覆盖镜像自带的 torch / CUDA）...
>>> 缺失依赖，即将安装: pyyaml, hf_transfer
...
>>> 未检测到 vllm，尝试安装与当前 torch/CUDA 匹配的版本...
>>> vllm 已存在，跳过
>>> 依赖检查通过
```

---

## 5. 步骤四：冒烟测试

```bash
bash scripts/run_eval.sh --dry-run --limit 2 --no-notify
```

- `--dry-run`:每个 task 只取 1 条;
- `--limit 2`:最多 2 条;
- `--no-notify`:先不打扰飞书。

**期望**:模型加载成功、打印评分行、写出结果、**exit code = 0**:

```
[HH:MM:SS] s1 Budget Forcing 评测启动（数据并行 × 2）| run_id=...
[HH:MM:SS] [parallel] 启动 shard 0/2（CUDA_VISIBLE_DEVICES=0）
[HH:MM:SS] [parallel] 启动 shard 1/2（CUDA_VISIBLE_DEVICES=1）
[HH:MM:SS] 合并完成，共 2 条 → .../results.jsonl
[HH:MM:SS] 全部完成。
```

---

## 6. 步骤五：正式运行

```bash
bash scripts/run_eval.sh
```

### 可用参数

`scripts/run_eval.sh` 会把除 `--install` 外的所有参数透传给 `s1_eval.run_eval`:

| 参数 | 作用 |
|---|---|
| `--config PATH` | 指定 YAML 配置(默认 `config/config.yaml`) |
| `--limit N` | 每个 task 最多评测 N 条 |
| `--dry-run` | 冒烟模式(每 task 1 条) |
| `--parallel data\|tensor` | 临时覆盖并行策略 |
| `--no-notify` | 关闭飞书通知 |
| `--no-watchdog` | 关闭超时看门狗 |

示例:

```bash
# 只跑 AIME24 前 5 条,切张量并行
bash scripts/run_eval.sh --limit 5 --parallel tensor
```

---

## 7. 步骤六：查看结果与验收

### 7.1 产物目录

```
$S1_OUTPUT_DIR/run_<时间戳>/
├── run.log          # 父进程日志
├── shard0.log       # 数据并行时各 worker 日志
├── shard1.log
├── shard0.jsonl     # 数据并行时各 worker 中间结果
├── shard1.jsonl
├── results.jsonl    # ★ 最终结果(每行一条样本)
└── summary.json     # ★ 汇总指标
```

### 7.2 `results.jsonl` 字段

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

- `grade` ∈ `{"Yes","No",null}`;`null` = 打分失败或无参考答案。
- `thinking` 中应能看到 `Wait`(Budget Forcing 生效的证据)。

### 7.3 `summary.json`

```json
{
  "by_task": {
    "aime24":  {"total": 30,  "correct": 16, "accuracy": 0.5333},
    "math500": {"total": 500, "correct": 446, "accuracy": 0.892}
  },
  "overall": {"total": 530, "correct": 462, "accuracy": 0.8717},
  "run_id": "20260922_153000",
  "model": "simplescaling/s1.1-32B",
  "budget_forcing": {"enabled": true, "max_tokens_thinking": 32000,
                     "num_ignore": 1, "ignore_str": "Wait"},
  "grader": "llm",
  "parallel": "data x2",
  "num_samples": 530,
  "elapsed_seconds": 5400.0
}
```

> 单进程模式额外有 `gen_seconds`(纯生成耗时);数据并行模式用 `elapsed_seconds` 表示总耗时。

### 7.4 验收标准

1. 进程 **exit code = 0**;
2. `results.jsonl` **行数 = 各 task 样本数之和**(默认 `30 + 500 = 530`);
3. `summary.json` 的 `overall.accuracy` 落在参考区间(见下);
4. 飞书依次收到 **start → (progress) → complete**;
5. `results.jsonl` 里 `grade` 基本为 `Yes`/`No`(不是大量 `null`)。

### 7.5 参考准确率区间

| 任务 | 样本数 | 参考准确率 |
|---|---|---|
| `aime24` | 30 | 40% – 60% |
| `math500` | 500 | 85% – 93% |

> 具体值取决于 `num_ignore`、`temperature` 与打分模型。本仓库是独立复现(非官方
> lm-eval-harness),与官方结果可能差几个百分点;`num_ignore` 越大(如 8)AIME 类难题通常更好但更慢。
> 官方结果见 HF `simplescaling/results`,以官方为准。

---

## 8. 双卡并行与 OOM 处理

`config/config.yaml`:

```yaml
model:
  parallel: "data"          # data=双卡数据并行(推荐先用) / tensor=张量并行
  num_gpus: 2
  tensor_parallel_size: 2   # 仅 parallel=tensor 生效
```

| 模式 | 机制 | 何时用 |
|---|---|---|
| `data` | 每张卡各放一份**完整 32B 副本**,样本对半分,速度 ≈ ×卡数 | 首选(单卡 80G 能放下 64G 权重 + KV) |
| `tensor` | 单份模型切到 2 张卡 | **`data` 爆显存(OOM)时**切换 |

切换:

```bash
bash scripts/run_eval.sh --parallel tensor     # 临时
# 或改 config.yaml: model.parallel: "tensor"    # 永久
```

数据并行下代码会**自动把每个 worker 的 `tensor_parallel_size` 置为 1**,无需手改。
单卡机器把 `model.num_gpus` 设为 `1`(或 `0` 自动检测)。

其它省显存手段:调小 `model.max_model_len`、调小 `model.gpu_memory_utilization`。

---

## 9. 一键自动化 + 自动停机

```bash
bash auto_run_and_kill.sh
```

它会:加载 `.env` → 跑 `scripts/run_eval.sh` → 无论成功失败都继续 → 最后停机。
在 runpod 上会走 `runpodctl stop pod "$RUNPOD_POD_ID"`。

> ⚠️ runpod 容器里 `shutdown -h now` **不会**停止计费,必须用 `runpodctl stop pod`。

---

## 10. 调参速查表

| 想做什么 | 改哪里 |
|---|---|
| 换模型(`s1-32B` / 自己的 ckpt) | `config.yaml` → `model.name` |
| 换 tokenizer | `model.tokenizer` |
| 换强塞的提示词(`Wait`→`Hmm`) | `budget_forcing.ignore_str` |
| 增加反思次数 | `budget_forcing.num_ignore` |
| 调思考预算 | `budget_forcing.max_tokens_thinking` |
| 关闭 Budget Forcing | `budget_forcing.enabled: false` 或 `S1_BUDGET_FORCING=0` |
| 只跑某个任务 | 删/注释 `eval.tasks` 里其它 task |
| 小样本试跑 | `--limit N` |
| 换打分模型 | `.env` → `GRADING_MODEL`(及 `BASE_URL`/`API_KEY`) |
| 打分改字符串精确匹配 | `eval.grader: "exact"` |
| 换系统提示词 | `eval.prompt.system` |
| 换并行策略 | `model.parallel` 或 `--parallel` |
| 调超时 | `watchdog.timeout_minutes` 或 `S1_TIMEOUT_MINUTES` |
| 关闭通知 | `notify.feishu.enabled: false` 或 `--no-notify` |

---

## 11. 环境变量速查

| 变量 | 作用 |
|---|---|
| `S1_CONFIG` | 配置文件路径 |
| `S1_OUTPUT_DIR` | 输出根目录 |
| `S1_LIMIT` | 每 task 条数上限 |
| `S1_MODEL_NAME` | 覆盖模型名 |
| `S1_PARALLEL` | `data` / `tensor` |
| `S1_NUM_GPUS` | 数据并行卡数 |
| `S1_TENSOR_PARALLEL_SIZE` | 张量并行度 |
| `S1_TIMEOUT_MINUTES` | watchdog 超时(分钟) |
| `S1_BUDGET_FORCING` | `0/1` 开关 |
| `S1_PAUSE_ON_EXIT` | `1` 时跑完不退出容器(看日志用) |
| `HF_HOME` / `HF_ENDPOINT` / `HF_HUB_ENABLE_HF_TRANSFER` | HuggingFace 缓存/镜像/加速 |

---

## 12. 排错速查

| 现象 | 原因 / 处理 |
|---|---|
| `打分模型未配置…回退 exact` | `.env` 缺 `GRADING_BASE_URL/MODEL`,补上 |
| `grade` 大量为 `null` | 打分端点不通/密钥错误 → `python -m s1_eval.grading` 自测 |
| `thinking` 极短、几乎全 `No` | Budget Forcing 未生效,或显存不足被截断 |
| 缺 `summary.json` / 结果不全 | 某 shard 失败,查 `shard*.log` |
| CUDA OOM | 切 `--parallel tensor`,或调小 `max_model_len` / `gpu_memory_utilization` |
| `vllm` 装不上 | 忽略,会自动回退 `transformers`(较慢) |
| HF 下载慢/超时 | `.env` 设 `HF_ENDPOINT="https://hf-mirror.com"` |
| `hf_transfer` 相关报错 | `pip install hf_transfer`,或注释掉 `HF_HUB_ENABLE_HF_TRANSFER` |
| 飞书无消息 | 检查 `FEISHU_WEBHOOK`;`python send_feishu.py "测试"` |
| 进程被 watchdog 杀掉 | 超时;调大 `watchdog.timeout_minutes` |
| runpod 不停机 | 用 `runpodctl stop pod "$RUNPOD_POD_ID"` |

---

## 13. 时间预估(粗略)

| 阶段 | 说明 |
|---|---|
| 首次下载模型(≈64GB) | 数分钟 ~ 20 分钟,视带宽;有 network volume 缓存后重跑免下载 |
| 加载到 vLLM | 1 ~ 3 分钟 |
| AIME24 + MATH500(530 条) | 双卡数据并行约 **1 ~ 3 小时**(取决于生成长度) |

---

## 附:已验证项

- ✅ 单进程评测路径端到端跑通(加载 → Budget Forcing 生成 → 打分 → 写盘,exit 0)
- ✅ **数据并行**分片(`shard0`=[0,2]、`shard1`=[1,3])与合并/汇总逻辑实测通过
- ✅ Watchdog 超时终止、飞书通知(开启/关闭)、配置 `${VAR}` 解析
- ⚠️ GPU 上的 vLLM 批量路径需在目标机实机确认(runpod 上 `--install` 会自动装)
