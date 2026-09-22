"""飞书（Feishu / Lark）Webhook 通知。

把原来的 ``send_feishu.py`` 逻辑封装成可复用类，并改为从配置 / 环境变量读取
webhook 与 secret（不再硬编码密钥）。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import urllib.request
from typing import Any

# 事件名 -> 卡片标题前缀 / 颜色
_EVENT_STYLE = {
    "start": ("🚀", "blue"),
    "progress": ("⏳", "blue"),
    "complete": ("✅", "green"),
    "error": ("❌", "red"),
    "timeout": ("⏰", "orange"),
    "info": ("🔔", "blue"),
}


def gen_sign(secret: str, timestamp: int) -> str:
    """飞书官方标准签名算法 (HMAC-SHA256)。"""
    string_to_sign = f"{timestamp}\n{secret}"
    hmac_code = hmac.new(
        string_to_sign.encode("utf-8"), digestmod=hashlib.sha256
    ).digest()
    return base64.b64encode(hmac_code).decode("utf-8")


class FeishuNotifier:
    """配置驱动的飞书机器人通知器。"""

    def __init__(
        self,
        webhook: str = "",
        secret: str = "",
        enabled: bool = True,
        source: str = "s1-32B 预算强制复现",
        events: list | None = None,
        progress_every: int = 25,
    ) -> None:
        self.webhook = (webhook or "").strip()
        self.secret = (secret or "").strip()
        self.enabled = bool(enabled) and bool(self.webhook)
        self.source = source
        self.events = set(
            events or ["start", "progress", "complete", "error", "timeout"]
        )
        self.progress_every = max(int(progress_every or 1), 1)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FeishuNotifier:
        node = (config or {}).get("notify", {}).get("feishu", {}) or {}
        return cls(
            webhook=node.get("webhook", ""),
            secret=node.get("secret", ""),
            enabled=node.get("enabled", True),
            source=node.get("source", "s1-32B 预算强制复现"),
            events=node.get("events"),
            progress_every=node.get("progress_every", 25),
        )

    # ------------------------------------------------------------------ #
    def should_notify(self, event: str) -> bool:
        return self.enabled and event in self.events

    def send(
        self,
        title: str,
        message: str,
        event: str = "info",
        template: str | None = None,
    ) -> bool:
        """发送一条飞书卡片。永远不抛异常（通知失败不影响主流程）。"""
        if not self.enabled:
            return False

        icon, default_color = _EVENT_STYLE.get(event, _EVENT_STYLE["info"])
        color = template or default_color
        timestamp = int(time.time())

        payload: dict[str, Any] = {
            "msg_type": "interactive",
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {
                    "title": {"tag": "plain_text", "content": f"{icon} {title}"},
                    "template": color,
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {"tag": "lark_md", "content": message},
                    }
                ],
            },
        }
        # 只有配置了 secret 才做签名（飞书未勾选签名校验时不需要）
        if self.secret:
            payload["timestamp"] = str(timestamp)
            payload["sign"] = gen_sign(self.secret, timestamp)

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook, data=data, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                res = json.loads(resp.read().decode("utf-8"))
            ok = res.get("StatusCode") == 0 or res.get("code") == 0
            if not ok:
                print(f"[notify] 飞书返回异常: {res}")
            return ok
        except Exception as exc:  # noqa: BLE001 - 通知失败不能影响主流程
            print(f"[notify] 飞书推送异常: {exc}")
            return False

    # ------------------------------------------------------------------ #
    def notify(
        self,
        event: str,
        message: str,
        title: str | None = None,
        template: str | None = None,
    ) -> bool:
        """按事件类型发送；未订阅该事件则跳过。"""
        if not self.should_notify(event):
            return False
        icon, _ = _EVENT_STYLE.get(event, _EVENT_STYLE["info"])
        title = title or f"{self.source} · {event}"
        body = f"**实验项目**: {self.source}\n\n{message}"
        return self.send(title, body, event=event, template=template)


if __name__ == "__main__":  # 手工测试：python -m s1_eval.notify "消息"
    import sys

    from s1_eval.config import load_config

    cfg = load_config()
    n = FeishuNotifier.from_config(cfg)
    n.notify("info", sys.argv[1] if len(sys.argv) > 1 else "测试消息")
