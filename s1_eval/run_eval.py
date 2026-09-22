"""评测主入口：跳过 SFT，直接拉取 s1-32B 跑 Budget Forcing + 自定义打分。

用法：
    python -m s1_eval.run_eval --config config/config.yaml
    python -m s1_eval.run_eval --limit 5 --dry-run
    python -m s1_eval.run_eval --no-watchdog --no-notify

它会：
  * 读取 config/config.yaml（+ config/.env）里的模型 / 超参 / 提示词 / 打分端点；
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
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from s1_eval.budget_forcing import BudgetForcingPipeline
from s1_eval.config import Config, load_config
from s1_eval.datasets import Sample, load_all_tasks
from s1_eval.grading import GradingClient, answers_match
from s1_eval.notify import FeishuNotifier
from s1_eval.watchdog import Watchdog, ensure_process_group


# ===================================================================== #
# 日志
# ===================================================================== #
class Logger:
    def __init__(self, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(log_path, "a", encoding="utf-8")

    def __call__(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass


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

    limit = (
        1
        if cfg.get_path("experiment.dry_run", False)
        else (
            args.limit if args.limit is not None else cfg.get_path("experiment.limit")
        )
    )

    # ---------- 输出目录 / 日志 ----------
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
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

        # ---------------- 加载模型 ----------------
        pipeline = BudgetForcingPipeline(cfg, log=log)
        pipeline.load()

        # ---------------- 打分器 ----------------
        grader_name = str(cfg.get_path("eval.grader", "llm")).lower()
        grader = GradingClient.from_config(cfg, log=log)
        use_llm_grader = grader_name == "llm"
        if use_llm_grader and not grader.configured:
            log(
                "[warn] 打分模型未配置（GRADING_BASE_URL/GRADING_MODEL 为空），"
                "自动回退为 exact 字符串匹配。"
            )
            use_llm_grader = False
        if use_llm_grader:
            log(
                f"[grading] 使用自定义打分端点: {grader.base_url} | model={grader.model}"
            )
        else:
            log("[grading] 使用 exact 字符串匹配。")

        # ---------------- 生成 ----------------
        log(f"开始生成，共 {len(samples)} 条样本...")
        questions = [s["question"] for s in samples]
        t0 = time.time()
        generations = pipeline.generate(questions)
        gen_seconds = time.time() - t0
        log(f"生成完成，用时 {gen_seconds / 60:.1f} 分钟。")

        # ---------------- 打分 ----------------
        results: list[dict[str, Any]] = []
        progress_every = int(cfg.get_path("notify.feishu.progress_every", 25) or 25)
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

            record = {
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
            results.append(record)

            done = i + 1
            log(
                f"  [{done}/{len(samples)}] task={sample['task']} "
                f"grade={grade['grade']}"
            )
            if notifier.enabled and done % progress_every == 0:
                notifier.notify(
                    "progress",
                    f"**进度**: {done}/{len(samples)}（{done / len(samples):.0%}）",
                )

        # ---------------- 汇总 ----------------
        summary = _summarize(results)
        summary.update(
            {
                "run_id": run_id,
                "model": cfg.get_path("model.name"),
                "budget_forcing": cfg.get_path("budget_forcing", {}),
                "grader": "llm" if use_llm_grader else "exact",
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
