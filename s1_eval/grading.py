"""打分模块：从 OpenAI 换成「任意 OpenAI 兼容端点」。

base_url / api_key / model 全部来自配置文件（见 config/config.yaml 的 grading 段），
底层用 openai SDK，因此兼容 DeepSeek、Qwen、自建 vLLM、ollama 等一切
提供 /chat/completions 的服务。
"""

from __future__ import annotations

import re
import time
from typing import Any

try:  # openai>=1.0
    from openai import OpenAI
except Exception:  # pragma: no cover - 运行环境缺依赖时给出清晰报错
    OpenAI = None  # type: ignore


_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _normalize(text: str | None) -> str:
    """归一化文本，便于精确匹配。"""
    if text is None:
        return ""
    text = str(text).strip()
    boxed = _BOXED_RE.findall(text)
    if boxed:
        text = boxed[-1]
    text = text.replace(",", "").replace("$", "").replace(" ", "")
    text = text.strip().rstrip(".")
    text = text.lower()
    return text


def _to_number(text: str) -> float | None:
    """把 "3"、"0.5"、"-1/2" 之类的字符串转成 float，失败返回 None。"""
    text = (text or "").strip()
    frac = re.fullmatch(r"(-?\d+)\s*/\s*(\d+)", text)
    if frac:
        try:
            return float(frac.group(1)) / float(frac.group(2))
        except ZeroDivisionError:
            return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def answers_match(prediction: str | None, reference: str | None) -> bool:
    """一个轻量的数学答案精确匹配（无需 math_verify 依赖）。"""
    pred_n, ref_n = _normalize(prediction), _normalize(reference)
    if not ref_n:
        return False
    if pred_n == ref_n:
        return True
    # 数值比较：允许 1/2 与 0.5、3 与 3.0 这类差异
    pred_num, ref_num = _to_number(pred_n), _to_number(ref_n)
    if pred_num is not None and ref_num is not None:
        return abs(pred_num - ref_num) < 1e-6
    return False


class GradingClient:
    """自定义打分模型客户端。"""

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        system_prompt: str = "",
        user_template: str = "{question}\n{attempt}\n{solution}",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        timeout: int = 120,
        retries: int = 5,
        retry_wait_seconds: float = 5,
        json_format: bool = False,
        log: Any | None = None,
    ) -> None:
        self.base_url = (base_url or "").strip()
        self.api_key = (api_key or "").strip()
        self.model = (model or "").strip()
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.timeout = int(timeout)
        self.retries = max(int(retries), 1)
        self.retry_wait_seconds = float(retry_wait_seconds)
        self.json_format = bool(json_format)
        self._log = log or print
        self._client = None

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, config: Any, log: Any | None = None) -> GradingClient:
        node = (config or {}).get("grading", {}) or {}
        return cls(
            base_url=node.get("base_url", ""),
            api_key=node.get("api_key", ""),
            model=node.get("model", ""),
            system_prompt=node.get("system_prompt", ""),
            user_template=node.get(
                "user_template", "{question}\n{attempt}\n{solution}"
            ),
            max_tokens=node.get("max_tokens", 2048),
            temperature=node.get("temperature", 0.0),
            timeout=node.get("timeout", 120),
            retries=node.get("retries", 5),
            retry_wait_seconds=node.get("retry_wait_seconds", 5),
            json_format=node.get("json_format", False),
            log=log,
        )

    # ------------------------------------------------------------------ #
    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model)

    def _get_client(self):
        if self._client is not None:
            return self._client
        if OpenAI is None:
            raise RuntimeError("未安装 openai，请先 `pip install openai`。")
        if not self.configured:
            raise RuntimeError(
                "打分模型未配置：请在 config/.env 里设置 "
                "GRADING_BASE_URL / GRADING_API_KEY / GRADING_MODEL。"
            )
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key or "EMPTY",  # 自建服务常不校验 key
            timeout=self.timeout,
        )
        return self._client

    def _build_user_prompt(self, question: str, attempt: str, solution: str) -> str:
        return self.user_template.format(
            question=question, attempt=attempt, solution=solution
        )

    # ------------------------------------------------------------------ #
    def grade(
        self, question: str, attempt: str, solution: str | None
    ) -> dict[str, Any]:
        """对单条样本打分，返回 ``{"grade": "Yes"/"No"/None, "reason": str}``。"""
        if solution is None:
            return {"grade": None, "reason": "无参考答案，跳过打分"}

        client = self._get_client()
        user_prompt = self._build_user_prompt(question, attempt, solution)

        last_err: str | None = None
        for attempt_idx in range(self.retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }
                if self.json_format:
                    kwargs["response_format"] = {"type": "json_object"}
                completion = client.chat.completions.create(**kwargs)
                content = (completion.choices[0].message.content or "").strip()

                verdict = None
                for line in reversed(content.splitlines()):
                    token = line.strip().strip(".").strip()
                    if token in ("Yes", "No"):
                        verdict = token
                        break
                if verdict is None:
                    lowered = content.lower()
                    if re.search(r"\byes\b", lowered):
                        verdict = "Yes"
                    elif re.search(r"\bno\b", lowered):
                        verdict = "No"
                return {"grade": verdict, "reason": content}
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                self._log(
                    f"[grading] 第 {attempt_idx + 1}/{self.retries} 次调用失败: {exc}"
                )
                time.sleep(self.retry_wait_seconds)
        return {"grade": None, "reason": f"打分失败: {last_err}"}

    def grade_many(self, items: list[dict[str, str]]) -> list[dict[str, Any]]:
        """批量打分；items: [{"question","attempt","solution"}, ...]"""
        return [
            self.grade(it["question"], it["attempt"], it.get("solution"))
            for it in items
        ]

    # ------------------------------------------------------------------ #
    def chat(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        json_format: bool | None = None,
    ) -> str:
        """通用对话接口，兼容原来 ``inference_utils.apiqa`` 的用法。"""
        client = self._get_client()
        use_json = self.json_format if json_format is None else json_format
        last_err: str | None = None
        for attempt_idx in range(self.retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt or self.system_prompt,
                        },
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }
                if use_json:
                    kwargs["response_format"] = {"type": "json_object"}
                completion = client.chat.completions.create(**kwargs)
                return (completion.choices[0].message.content or "").strip()
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                self._log(
                    f"[grading] chat 第 {attempt_idx + 1}/{self.retries} 次失败: {exc}"
                )
                time.sleep(self.retry_wait_seconds)
        raise RuntimeError(f"自定义打分端点调用失败: {last_err}")


if __name__ == "__main__":  # 手工联调：python -m s1_eval.grading
    from s1_eval.config import load_config

    cfg = load_config()
    client = GradingClient.from_config(cfg)
    print("base_url =", client.base_url)
    print("model    =", client.model)
    print("configured =", client.configured)
    if client.configured:
        result = client.grade("1+1=?", "Answer: 2", "2")
        print("test grade:", result)
