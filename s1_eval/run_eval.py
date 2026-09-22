"""评测主入口：跳过 SFT，直接拉取 s1.1-32B 跑 Budget Forcing + 自定义打分。

用法：
    python -m s1_eval.run_eval --config config/config.yaml
    python -m s1_eval.run_eval --limit 5 --dry-run
    python -m s1_eval.run_eval --no-watchdog --no-notify
    python -m s1_eval.run_eval --parallel tensor      # 临时切张量并行

它会：
  * 读取 config/config.yaml（+ config/.env）里的模型 / 超参 / 提示词 / 打分端点；
  * 按 model.parallel 决定并行方式：data=双卡各一份副本数据并行，tensor=张量并行；
  * 启动 Watchdog，超时自动终止并通知飞书；
  * 生成时强塞 "Wait"（Budget Forcing）；
  * 用自定义 OpenAI 兼容端点打分；
  * 开始 / 进度 / 完成 / 报错 / 超时 / 被终止 都会推送飞书。
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from s1_eval.budget_forcing import BudgetForcingPipeline
from s1_eval.config import Config, load_config
from s1_eval.datasets import Sample, load_all_tasks
from s1_eval.grading import GradingClient, answers_match
from s1_eval.notify import FeishuNotifier
from s1_eval.watchdog import Watchdog, ensure_process_group

# 日志回调类型：Logger 实例或任意可调用对象（便于单测传入空实现）
LogFn = Callable[[str], None]


# ===================================================================== #
# 日志
# ===================================================================== #
class Logger:
    def __init__(self, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # 长生命周期句柄（整个评测期间持有），不能放进 with
        self._fh = open(log_path, "a", encoding="utf-8")  # noqa: SIM115

    def __call__(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"  # noqa: DTZ005
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


# ===================================================================== #
# 参数
# ===================================================================== #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="s1 Budget Forcing 评测")
    p.add_argument("--config", default="config/config.yaml", help="YAML 配置路径")
    p.add_argument("--limit", type=int, default=None, help="每个 task 最多评测多少条")
    p.add_argument("--dry-run", action="store_true", help="只跑通流程（每 task 1 条）")
    p.add_argument("--no-watchdog", action="store_true", help="禁用 watchdog")
    p.add_argument("--no-notify", action="store_true", help="禁用飞书通知")
    p.add_argument(
        "--parallel",
        choices=["auto", "data", "tensor", "none"],
        default=None,
        help="覆盖 config.yaml 的 model.parallel",
    )
    # 以下为内部参数：数据并行父进程拉起 worker 时使用，用户一般不用手填
    p.add_argument("--shard-index", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--run-dir", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


# ===================================================================== #
# 主流程
# ===================================================================== #
def main() -> int:
    args = parse_args()
    cfg: Config = load_config(args.config)

    if args.dry_run:
        cfg.set_path("experiment.dry_run", True)
    if args.limit is not None:
        cfg.set_path("experiment.limit", args.limit)

    # 数据并行 worker：由父进程通过 --shard-index 拉起，只跑自己那份样本
    if args.shard_index is not None:
        return _run_worker(cfg, args)

    mode = _resolve_parallel_mode(cfg, args)
    n_gpu = _detect_gpu_count(cfg)
    if mode == "data":
        if n_gpu >= 2:
            return _run_data_parallel(cfg, args, n_gpu)
        print(f"[parallel] parallel=data，但只检测到 {n_gpu} 张 GPU，回退单进程。")
    return _run_single(cfg, args)


def _run_single(cfg: Config, args: argparse.Namespace) -> int:
    limit = _resolve_limit(cfg, args)

    # ---------- 输出目录 / 日志 ----------
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
    out_root = Path(cfg.get_path("experiment.output_dir", "outputs"))
    run_dir = out_root / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(run_dir / "run.log")
    run_start = time.time()
    log("=" * 70)
    log(f"s1 Budget Forcing 评测启动 | run_id={run_id}")
    log(f"输出目录: {run_dir}")

    import random

    random.seed(int(cfg.get_path("experiment.seed", 42)))

    notifier = FeishuNotifier.from_config(cfg)
    if args.no_notify:
        notifier.enabled = False
    log(f"飞书通知: {'开启' if notifier.enabled else '关闭'}")

    # ---------- watchdog ----------
    state: dict[str, Any] = {"timeout_fired": False, "finished": False}
    watchdog: Watchdog | None = None
    wd_cfg = cfg.get_path("watchdog", {}) or {}
    timeout_min = float(wd_cfg.get("timeout_minutes", 180))

    def on_timeout(elapsed: float) -> None:
        state["timeout_fired"] = True
        notifier.notify(
            "timeout",
            f"**运行状态**: 超过 {timeout_min:.0f} 分钟上限，已运行 "
            f"{elapsed / 60:.1f} 分钟\n\n**处理**: 自动终止进程（含子进程）",
        )

    if not args.no_watchdog and wd_cfg.get("enabled", True):
        ensure_process_group()
        watchdog = Watchdog.from_config(cfg, on_timeout=on_timeout, log=log)
        if watchdog is not None:
            watchdog.start()

    # ---------- 进程终止时也要通知 ----------
    def _signal_handler(signum: int, _frame: Any) -> None:
        name = signal.Signals(signum).name
        log(f"收到信号 {name}，准备退出。")
        if not state["timeout_fired"] and not state["finished"]:
            notifier.notify(
                "error",
                f"**运行状态**: 收到信号 `{name}`，进程被终止\n\n"
                f"**输出目录**: `{run_dir}`",
            )
        sys.exit(143 if signum == signal.SIGTERM else 130)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _signal_handler)
        except (ValueError, OSError):  # 非主线程等场景
            pass

    notifier.notify(
        "start",
        f"**运行状态**: 开始评测\n\n**模型**: `{cfg.get_path('model.name')}`\n\n"
        f"**Budget Forcing**: "
        f"{'开启' if cfg.get_path('budget_forcing.enabled', True) else '关闭'}"
        f"（ignore_str=`{cfg.get_path('budget_forcing.ignore_str')}`, "
        f"num_ignore={cfg.get_path('budget_forcing.num_ignore')}）\n\n"
        f"**输出目录**: `{run_dir}`",
    )

    exit_code = 0
    try:
        # ---------------- 加载数据 ----------------
        samples: list[Sample] = load_all_tasks(cfg, limit=limit, log=log)
        if not samples:
            raise RuntimeError("没有加载到任何样本，请检查 eval.tasks 配置。")

        # ---------------- 生成 + 打分 ----------------
        results, gen_seconds, use_llm_grader = _evaluate(
            cfg, samples, log, watchdog=watchdog, notifier=notifier
        )

        # ---------------- 汇总 ----------------
        summary = _summarize(results)
        summary.update(
            {
                "run_id": run_id,
                "model": cfg.get_path("model.name"),
                "budget_forcing": cfg.get_path("budget_forcing", {}),
                "grader": "llm" if use_llm_grader else "exact",
                "parallel": _resolve_parallel_mode(cfg, args),
                "gen_seconds": round(gen_seconds, 1),
                "num_samples": len(results),
                "elapsed_seconds": round(time.time() - run_start, 1),
            }
        )

        _write_results(run_dir, results, summary)
        log(f"结果已写入: {run_dir / 'results.jsonl'}")

        state["finished"] = True
        overall = summary.get("overall", {})
        notifier.notify(
            "complete",
            f"**运行状态**: 评测完成 ✅\n\n"
            f"**准确率**: {overall.get('accuracy', 0):.1%} "
            f"({overall.get('correct', 0)}/{overall.get('total', 0)})\n\n"
            f"**分任务**: {json.dumps(summary.get('by_task', {}), ensure_ascii=False)}\n\n"
            f"**输出目录**: `{run_dir}`",
        )
        log("全部完成。")

    except Exception as exc:  # noqa: BLE001
        import traceback

        tb = traceback.format_exc()
        log(f"运行出错: {exc}\n{tb}")
        notifier.notify(
            "error",
            f"**运行状态**: 评测失败 ❌\n\n**错误**: `{exc}`\n\n"
            f"**输出目录**: `{run_dir}`",
        )
        exit_code = 1
    finally:
        if watchdog is not None:
            watchdog.stop()
        log.close()
        _pause_for_pod()

    return exit_code


# ===================================================================== #
# 辅助
# ===================================================================== #
def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, dict[str, int]] = {}
    for r in results:
        task = r["task"]
        stats = by_task.setdefault(task, {"total": 0, "correct": 0, "graded": 0})
        stats["total"] += 1
        if r["grade"] == "Yes":
            stats["correct"] += 1
            stats["graded"] += 1
        elif r["grade"] == "No":
            stats["graded"] += 1

    formatted: dict[str, dict[str, Any]] = {}
    total_all = correct_all = 0
    for task, s in by_task.items():
        acc = (s["correct"] / s["total"]) if s["total"] else 0.0
        formatted[task] = {
            "total": s["total"],
            "correct": s["correct"],
            "accuracy": round(acc, 4),
        }
        total_all += s["total"]
        correct_all += s["correct"]

    overall_acc = (correct_all / total_all) if total_all else 0.0
    return {
        "by_task": formatted,
        "overall": {
            "total": total_all,
            "correct": correct_all,
            "accuracy": round(overall_acc, 4),
        },
    }


def _resolve_limit(cfg: Config, args: argparse.Namespace):
    """dry-run 优先；其次命令行 --limit；最后配置文件里的 experiment.limit。"""
    if cfg.get_path("experiment.dry_run", False):
        return 1
    if args.limit is not None:
        return args.limit
    return cfg.get_path("experiment.limit")


def _resolve_parallel_mode(cfg: Config, args: argparse.Namespace) -> str:
    mode = (
        args.parallel
        or os.environ.get("S1_PARALLEL")
        or cfg.get_path("model.parallel", "tensor")
    )
    mode = str(mode or "tensor").lower()
    return mode if mode in {"data", "tensor", "none"} else "tensor"


def _detect_gpu_count(cfg: Config | None = None) -> int:
    n = 0
    try:
        import torch

        n = torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        n = 0
    if n == 0:
        try:
            out = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            n = sum(1 for line in out.stdout.splitlines() if line.startswith("GPU "))
        except Exception:  # noqa: BLE001
            n = 0
    if cfg is not None:
        want = cfg.get_path("model.num_gpus", 0) or 0
        if int(want) > 0:
            n = min(n, int(want))
    return n


def _build_grader(cfg: Config, log: LogFn):
    """返回 (use_llm_grader, grader)；未配置端点时自动回退 exact。"""
    grader_name = str(cfg.get_path("eval.grader", "llm")).lower()
    grader = GradingClient.from_config(cfg, log=log)
    use_llm = grader_name == "llm"
    if use_llm and not grader.configured:
        log(
            "[warn] 打分模型未配置（GRADING_BASE_URL/GRADING_MODEL 为空），"
            "自动回退为 exact 字符串匹配。"
        )
        use_llm = False
    if use_llm:
        log(f"[grading] 使用自定义打分端点: {grader.base_url} | model={grader.model}")
    else:
        log("[grading] 使用 exact 字符串匹配。")
    return use_llm, grader


def _evaluate(
    cfg: Config,
    samples: list[Sample],
    log: Logger,
    watchdog: Watchdog | None = None,
    notifier: FeishuNotifier | None = None,
    progress_every: int | None = None,
) -> tuple[list[dict[str, Any]], float, bool]:
    """加载模型 → Budget Forcing 生成 → 打分。

    返回 ``(results, gen_seconds, use_llm_grader)``。
    """
    pipeline = BudgetForcingPipeline(cfg, log=log)
    pipeline.load()

    use_llm_grader, grader = _build_grader(cfg, log)

    log(f"开始生成，共 {len(samples)} 条样本...")
    t0 = time.time()
    generations = pipeline.generate([s["question"] for s in samples])
    gen_seconds = time.time() - t0
    log(f"生成完成，用时 {gen_seconds / 60:.1f} 分钟。")

    if progress_every is None:
        progress_every = int(cfg.get_path("notify.feishu.progress_every", 25) or 25)

    results: list[dict[str, Any]] = []
    for i, (sample, gen) in enumerate(zip(samples, generations)):
        if watchdog is not None:
            watchdog.touch()

        prediction = gen.get("answer") or gen.get("response") or ""
        if use_llm_grader:
            grade = grader.grade(sample["question"], prediction, sample["answer"])
        else:
            correct = answers_match(prediction, sample["answer"])
            grade = {
                "grade": "Yes" if correct else "No",
                "reason": "exact match" if correct else "mismatch",
            }

        results.append(
            {
                "task": sample["task"],
                "index": sample["index"],
                "question": sample["question"],
                "reference": sample["answer"],
                "thinking": gen.get("thinking", ""),
                "prediction": prediction,
                "grade": grade["grade"],
                "reason": grade["reason"],
                "backend": gen.get("backend"),
            }
        )

        done = i + 1
        log(f"  [{done}/{len(samples)}] task={sample['task']} grade={grade['grade']}")
        if (
            notifier is not None
            and notifier.enabled
            and progress_every
            and done % progress_every == 0
        ):
            notifier.notify(
                "progress",
                f"**进度**: {done}/{len(samples)}（{done / len(samples):.0%}）",
            )
    return results, gen_seconds, use_llm_grader


def _merge_shards(run_dir: Path, n_shards: int) -> list[dict[str, Any]]:
    """把各 worker 写出的 shard{i}.jsonl 合并，并按 (task, index) 排序。"""
    results: list[dict[str, Any]] = []
    for i in range(n_shards):
        shard_file = run_dir / f"shard{i}.jsonl"
        if not shard_file.exists():
            continue
        for line in shard_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                results.append(json.loads(line))
    results.sort(key=lambda r: (str(r.get("task", "")), int(r.get("index", 0))))
    return results


def _run_worker(cfg: Config, args: argparse.Namespace) -> int:
    """数据并行 worker：只跑 samples[shard::num_shards]，结果写 shard{i}.jsonl。"""
    shard = int(args.shard_index or 0)
    total = max(int(args.num_shards or 1), 1)
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else Path(cfg.get_path("experiment.output_dir", "outputs"))
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(run_dir / f"shard{shard}.log")
    log(
        f"[shard {shard}/{total}] 启动（CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}）"
    )

    # 数据并行下每张卡只放一份完整副本 → 张量并行必须为 1
    if total > 1:
        cfg.set_path("model.tensor_parallel_size", 1)

    samples = load_all_tasks(cfg, limit=_resolve_limit(cfg, args), log=log)
    if not samples:
        raise RuntimeError("没有加载到任何样本，请检查 eval.tasks 配置。")
    mine = samples[shard::total]
    log(f"[shard {shard}/{total}] 分到 {len(mine)}/{len(samples)} 条样本")

    results, gen_seconds, _ = _evaluate(
        cfg, mine, log, watchdog=None, notifier=None, progress_every=0
    )

    out = run_dir / f"shard{shard}.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
    log(
        f"[shard {shard}/{total}] 完成：{len(results)} 条，"
        f"生成用时 {gen_seconds / 60:.1f} 分钟 → {out}"
    )
    log.close()
    return 0


def _run_data_parallel(cfg: Config, args: argparse.Namespace, n_gpu: int) -> int:
    """父进程：按 GPU 数切分样本，拉起 n_gpu 个 worker，最后合并并汇总。"""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
    out_root = Path(cfg.get_path("experiment.output_dir", "outputs"))
    run_dir = out_root / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(run_dir / "run.log")
    run_start = time.time()
    log("=" * 70)
    log(f"s1 Budget Forcing 评测启动（数据并行 × {n_gpu}）| run_id={run_id}")
    log(f"输出目录: {run_dir}")

    notifier = FeishuNotifier.from_config(cfg)
    if args.no_notify:
        notifier.enabled = False
    log(f"飞书通知: {'开启' if notifier.enabled else '关闭'}")

    state: dict[str, Any] = {"timeout_fired": False, "finished": False}
    wd_cfg = cfg.get_path("watchdog", {}) or {}
    timeout_min = float(wd_cfg.get("timeout_minutes", 180))

    def on_timeout(elapsed: float) -> None:
        state["timeout_fired"] = True
        notifier.notify(
            "timeout",
            f"**运行状态**: 超过 {timeout_min:.0f} 分钟上限，已运行 "
            f"{elapsed / 60:.1f} 分钟\n\n"
            f"**处理**: 自动终止整个进程组（含 {n_gpu} 个 worker）",
        )

    watchdog: Watchdog | None = None
    if not args.no_watchdog and wd_cfg.get("enabled", True):
        ensure_process_group()
        watchdog = Watchdog.from_config(cfg, on_timeout=on_timeout, log=log)
        if watchdog is not None:
            watchdog.start()

    def _signal_handler(signum: int, _frame: Any) -> None:
        name = signal.Signals(signum).name
        log(f"收到信号 {name}，准备退出。")
        if not state["timeout_fired"] and not state["finished"]:
            notifier.notify(
                "error",
                f"**运行状态**: 收到信号 `{name}`，进程被终止\n\n"
                f"**输出目录**: `{run_dir}`",
            )
        sys.exit(143 if signum == signal.SIGTERM else 130)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _signal_handler)
        except (ValueError, OSError):
            pass

    notifier.notify(
        "start",
        f"**运行状态**: 开始评测（数据并行 × {n_gpu}）\n\n"
        f"**模型**: `{cfg.get_path('model.name')}`\n\n"
        f"**Budget Forcing**: "
        f"{'开启' if cfg.get_path('budget_forcing.enabled', True) else '关闭'}"
        f"（ignore_str=`{cfg.get_path('budget_forcing.ignore_str')}`, "
        f"num_ignore={cfg.get_path('budget_forcing.num_ignore')}）\n\n"
        f"**输出目录**: `{run_dir}`",
    )

    exit_code = 0
    procs: list[tuple[int, subprocess.Popen]] = []
    try:
        cmd_base = [
            sys.executable,
            "-m",
            "s1_eval.run_eval",
            "--config",
            args.config,
            "--num-shards",
            str(n_gpu),
            "--run-dir",
            str(run_dir),
            "--no-notify",
        ]
        if args.dry_run:
            cmd_base.append("--dry-run")
        if args.limit is not None:
            cmd_base += ["--limit", str(args.limit)]

        for i in range(n_gpu):
            child_env = os.environ.copy()
            child_env["CUDA_VISIBLE_DEVICES"] = str(i)
            cmd = cmd_base + ["--shard-index", str(i)]
            log(f"[parallel] 启动 shard {i}/{n_gpu}（CUDA_VISIBLE_DEVICES={i}）")
            procs.append((i, subprocess.Popen(cmd, env=child_env)))

        # 等待所有 worker：边等边喂 watchdog
        while not all(p.poll() is not None for _, p in procs):
            if watchdog is not None:
                watchdog.touch()
            time.sleep(5)

        failed = [(i, p.returncode) for i, p in procs if p.returncode != 0]
        if failed:
            raise RuntimeError(f"以下 shard 未成功退出: {failed}")

        results = _merge_shards(run_dir, n_gpu)
        if not results:
            raise RuntimeError("没有合并到任何结果，请检查各 shard 日志。")

        grader_is_llm = (
            str(cfg.get_path("eval.grader", "llm")).lower() == "llm"
            and GradingClient.from_config(cfg).configured
        )
        summary = _summarize(results)
        summary.update(
            {
                "run_id": run_id,
                "model": cfg.get_path("model.name"),
                "budget_forcing": cfg.get_path("budget_forcing", {}),
                "grader": "llm" if grader_is_llm else "exact",
                "parallel": f"data x{n_gpu}",
                "num_samples": len(results),
                "elapsed_seconds": round(time.time() - run_start, 1),
            }
        )
        _write_results(run_dir, results, summary)
        log(f"合并完成，共 {len(results)} 条 → {run_dir / 'results.jsonl'}")

        state["finished"] = True
        overall = summary.get("overall", {})
        notifier.notify(
            "complete",
            f"**运行状态**: 评测完成 ✅（数据并行 × {n_gpu}）\n\n"
            f"**准确率**: {overall.get('accuracy', 0):.1%} "
            f"({overall.get('correct', 0)}/{overall.get('total', 0)})\n\n"
            f"**分任务**: {json.dumps(summary.get('by_task', {}), ensure_ascii=False)}\n\n"
            f"**输出目录**: `{run_dir}`",
        )
        log("全部完成。")

    except Exception as exc:  # noqa: BLE001
        import traceback

        log(f"运行出错: {exc}\n{traceback.format_exc()}")
        notifier.notify(
            "error",
            f"**运行状态**: 评测失败 ❌\n\n**错误**: `{exc}`\n\n"
            f"**输出目录**: `{run_dir}`",
        )
        exit_code = 1
    finally:
        for _, p in procs:
            if p.poll() is None:
                p.terminate()
        if watchdog is not None:
            watchdog.stop()
        log.close()
        _pause_for_pod()

    return exit_code


def _write_results(
    run_dir: Path, results: list[dict[str, Any]], summary: dict[str, Any]
) -> None:
    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
    with open(run_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)


def _pause_for_pod() -> None:
    """runpod 上可选：进程结束后保持容器存活，方便查看日志。"""
    if str(os.environ.get("S1_PAUSE_ON_EXIT", "")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        print("S1_PAUSE_ON_EXIT=1，保持容器存活。按 Ctrl+C 退出。")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    sys.exit(main())
