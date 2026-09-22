"""评测数据集加载。

支持两种来源：
1. HuggingFace ``datasets``（配置 dataset / split / question_field / answer_field）；
2. 本地 ``.jsonl`` / ``.json`` 文件（每行/每项含 question & answer 字段）。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = ["Sample", "load_task"]


class Sample(dict):
    """一条评测样本：{"question", "answer", "task", "index"}。"""


def _load_local(path: str, q_field: str, a_field: str) -> list[dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = json.loads(text)
        if isinstance(rows, dict):
            rows = rows.get("data", rows.get("rows", []))
    out = []
    for row in rows:
        out.append(
            {
                "question": str(row.get(q_field, "")),
                "answer": str(row.get(a_field, "")),
            }
        )
    return out


def load_task(
    task_cfg: dict[str, Any],
    limit: int | None = None,
    log: Callable[[str], None] | None = None,
) -> list[Sample]:
    """加载单个 task，返回 ``Sample`` 列表。"""
    log = log or print
    name = task_cfg.get("name", "task")
    dataset = str(task_cfg.get("dataset", ""))
    q_field = task_cfg.get("question_field", "question")
    a_field = task_cfg.get("answer_field", "answer")
    max_samples = task_cfg.get("max_samples")
    if limit is not None:
        max_samples = (
            limit if max_samples is None else min(int(max_samples), int(limit))
        )

    if dataset.endswith((".jsonl", ".json")) or Path(dataset).exists():
        log(f"[data] {name}: 从本地文件加载 {dataset}")
        rows = _load_local(dataset, q_field, a_field)
    else:
        from datasets import load_dataset

        split = task_cfg.get("split", "train")
        log(f"[data] {name}: 从 HF 加载 {dataset} (split={split})")
        ds = load_dataset(dataset, split=split, trust_remote_code=True)
        rows = []
        for row in ds:
            rows.append(
                {
                    "question": str(row.get(q_field, "")),
                    "answer": str(row.get(a_field, "")),
                }
            )

    if task_cfg.get("shuffle"):
        import random

        random.shuffle(rows)

    if max_samples is not None:
        rows = rows[: int(max_samples)]

    samples = [
        Sample(
            question=r["question"],
            answer=r["answer"],
            task=name,
            index=i,
        )
        for i, r in enumerate(rows)
    ]
    log(f"[data] {name}: 共 {len(samples)} 条样本")
    return samples


def load_all_tasks(
    config: Any,
    limit: int | None = None,
    log: Callable[[str], None] | None = None,
) -> list[Sample]:
    """加载配置里所有 task 并合并。"""
    log = log or print
    eval_cfg = (config or {}).get("eval", {}) or {}
    tasks = eval_cfg.get("tasks", []) or []
    all_samples: list[Sample] = []
    for task_cfg in tasks:
        all_samples.extend(load_task(task_cfg, limit=limit, log=log))
    return all_samples
