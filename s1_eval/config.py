"""集中式配置加载器。

读取 YAML 配置 + ``.env`` 环境变量，支持 `${VAR}` / `${VAR:-默认值}` 语法，
并允许用少量环境变量（S1_*）覆盖关键项。所有其它模块都通过这里拿配置。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

# 匹配 ${VAR} 或 ${VAR:-default}
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def load_dotenv(path: str | None) -> dict[str, str]:
    """一个极简的 .env 解析器（避免额外依赖）。

    已存在的环境变量不会被覆盖（``setdefault``），这样 shell 里显式 export
    的值优先级更高。
    """
    env: dict[str, str] = {}
    if not path:
        return env
    p = Path(path)
    if not p.exists():
        return env
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        env[key] = value
        os.environ.setdefault(key, value)
    return env


def _resolve(value: Any, env: dict[str, str]) -> Any:
    """递归地把字符串里的 ${VAR} 替换成环境变量值。"""
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            var, default = match.group(1), match.group(2)
            if var in env:
                return env[var]
            if var in os.environ:
                return os.environ[var]
            return default if default is not None else ""

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _resolve(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, env) for v in value]
    return value


def normalize_path(path_str: Any) -> Any:
    """把「看起来是本地路径」的字符串归一化为绝对路径。

    目的:避免把 ``root/qwen_tokenizer`` 这类**漏写前导斜杠**的本地目录,
    误当成 HuggingFace Hub 仓库 id(``org/name``)去发网络请求(401)。

    - 路径存在(含补上前导 ``/`` 后存在)→ 返回绝对路径;
    - 否则原样返回(例如真正的 HF repo id ``simplescaling/s1.1-32B``)。
    """
    if not isinstance(path_str, str) or not path_str:
        return path_str

    candidates = [path_str]
    # 含目录分隔符 / 以 ~ 或 . 开头时,尝试补上前导斜杠
    if "/" in path_str or path_str.startswith("~"):
        candidates.append("/" + path_str)

    for cand in candidates:
        expanded = os.path.abspath(os.path.expanduser(cand))
        if os.path.exists(expanded):
            return expanded
    return path_str


def _normalize_paths(cfg: Config) -> Config:
    """对模型相关路径做本地路径归一化。"""
    for key in ("name", "tokenizer", "cache_dir"):
        value = cfg.get_path(f"model.{key}")
        if value:
            cfg.set_path(f"model.{key}", normalize_path(value))
    return cfg


class Config(dict):
    """支持 ``cfg.model.name`` 和 ``cfg["model"]["name"]`` 两种访问方式。"""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - 报错更友好
            raise AttributeError(f"配置项不存在: {item}") from exc

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: dict[str, Any] = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value


def _apply_env_overrides(cfg: Config) -> Config:
    """允许用少量 S1_* 环境变量覆盖配置，方便在 CI / runpod 上临时改。"""
    env = os.environ

    if env.get("S1_OUTPUT_DIR"):
        cfg.set_path("experiment.output_dir", env["S1_OUTPUT_DIR"])
    if env.get("S1_LIMIT") and str(env["S1_LIMIT"]).isdigit():
        limit = int(env["S1_LIMIT"])
        for task in cfg.get_path("eval.tasks", []) or []:
            task["max_samples"] = min(task.get("max_samples", limit), limit)
    if env.get("S1_MODEL_NAME"):
        cfg.set_path("model.name", env["S1_MODEL_NAME"])
    if env.get("S1_TENSOR_PARALLEL_SIZE"):
        cfg.set_path("model.tensor_parallel_size", int(env["S1_TENSOR_PARALLEL_SIZE"]))
    if env.get("S1_PARALLEL"):
        cfg.set_path("model.parallel", env["S1_PARALLEL"])
    if env.get("S1_NUM_GPUS") and str(env["S1_NUM_GPUS"]).isdigit():
        cfg.set_path("model.num_gpus", int(env["S1_NUM_GPUS"]))
    if env.get("S1_TIMEOUT_MINUTES"):
        cfg.set_path("watchdog.timeout_minutes", float(env["S1_TIMEOUT_MINUTES"]))
    if env.get("S1_BUDGET_FORCING") is not None:
        cfg.set_path(
            "budget_forcing.enabled",
            str(env["S1_BUDGET_FORCING"]).lower() in {"1", "true", "yes", "on"},
        )
    return cfg


def load_config(
    config_path: str = "config/config.yaml",
    dotenv_path: str | None = None,
) -> Config:
    """加载配置。

    Args:
        config_path: YAML 配置路径。
        dotenv_path: .env 路径；默认取 YAML 同目录下的 ``.env``。
    """
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    if dotenv_path is None:
        candidate = cfg_path.parent / ".env"
        dotenv_path = str(candidate) if candidate.exists() else None

    env = dict(os.environ)
    load_dotenv(dotenv_path)
    env.update({k: v for k, v in os.environ.items()})

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    resolved = _resolve(raw, env)
    cfg = Config(resolved)
    cfg = _apply_env_overrides(cfg)
    return _normalize_paths(cfg)


if __name__ == "__main__":  # 便于排查配置是否正确
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    print(json.dumps(load_config(path), indent=2, ensure_ascii=False))
