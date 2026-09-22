"""Watchdog：超过设定时限后自动终止整个进程（含子进程），并触发回调。

在 runpod 上跑长任务时，若卡死或超出预算，watchdog 会：
1. 调用 ``on_timeout`` 回调（一般用来发飞书通知）；
2. 向进程组发送 SIGTERM，宽限期后仍存活则发送 SIGKILL。
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from typing import Any


class Watchdog:
    """一个基于守护线程的软超时看门狗。

    Args:
        timeout_seconds: 超时时间（秒）。
        on_timeout: 超时触发时的回调（在 watchdog 线程里执行）。
        heartbeat_seconds: 心跳日志间隔；<=0 表示不打日志。
        grace_seconds: SIGTERM 之后等待多久再 SIGKILL。
        kill_process_group: 是否杀掉整个进程组（runpod 多进程场景建议 True）。
        log: 日志函数，默认 print。
    """

    def __init__(
        self,
        timeout_seconds: float,
        on_timeout: Callable[[float], Any] | None = None,
        heartbeat_seconds: float = 60,
        grace_seconds: float = 30,
        kill_process_group: bool = True,
        log: Callable[[str], None] | None = None,
        daemon: bool = True,
    ) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.on_timeout = on_timeout
        self.heartbeat_seconds = float(heartbeat_seconds)
        self.grace_seconds = float(grace_seconds)
        self.kill_process_group = kill_process_group
        self._log = log or print
        self._daemon = daemon

        self._start_time = time.time()
        self._last_touch = time.time()
        self._stop_event = threading.Event()
        self._fired = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(
        cls,
        config: Any,
        on_timeout: Callable[[float], Any] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> Watchdog | None:
        """从配置构造；若 ``watchdog.enabled`` 为 False 则返回 None。"""
        node = (config or {}).get("watchdog", {}) or {}
        if not node.get("enabled", True):
            return None
        return cls(
            timeout_seconds=float(node.get("timeout_minutes", 180)) * 60.0,
            on_timeout=on_timeout,
            heartbeat_seconds=float(node.get("heartbeat_seconds", 60)),
            grace_seconds=float(node.get("grace_seconds", 30)),
            kill_process_group=bool(node.get("kill_process_group", True)),
            log=log,
            daemon=True,
        )

    # ------------------------------------------------------------------ #
    @property
    def elapsed(self) -> float:
        return time.time() - self._start_time

    @property
    def remaining(self) -> float:
        return max(self.timeout_seconds - self.elapsed, 0.0)

    def touch(self) -> None:
        """记录一次心跳；成功推进则返回正常。"""
        self._last_touch = time.time()

    def start(self) -> Watchdog:
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._run, name="s1-watchdog", daemon=self._daemon
        )
        self._thread.start()
        self._log(
            f"[watchdog] 已启动：超时 {self.timeout_seconds / 60:.1f} 分钟，"
            f"宽限 {self.grace_seconds:.0f} 秒，kill_process_group={self.kill_process_group}"
        )
        return self

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2)

    def __enter__(self) -> Watchdog:
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        last_heartbeat = time.time()
        while not self._stop_event.is_set():
            # 睡眠间隔取心跳间隔和 1 秒中的较小者，保证超时响应及时
            interval = min(max(self.heartbeat_seconds, 0) or 1.0, 1.0)
            if self._stop_event.wait(interval):
                break

            if self.elapsed >= self.timeout_seconds:
                if not self._fired:
                    self._fired = True
                    self._handle_timeout()
                return

            if (
                self.heartbeat_seconds > 0
                and time.time() - last_heartbeat >= self.heartbeat_seconds
            ):
                last_heartbeat = time.time()
                self._log(
                    f"[watchdog] 心跳：已运行 {self.elapsed / 60:.1f} 分钟，"
                    f"剩余 {self.remaining / 60:.1f} 分钟"
                )

    # ------------------------------------------------------------------ #
    def _handle_timeout(self) -> None:
        self._log(
            f"[watchdog] ⏰ 超时！已运行 {self.elapsed / 60:.1f} 分钟"
            f"（上限 {self.timeout_seconds / 60:.1f} 分钟），准备终止进程。"
        )
        if self.on_timeout is not None:
            try:
                self.on_timeout(self.elapsed)
            except Exception as exc:  # noqa: BLE001 - 回调失败也要继续杀进程
                self._log(f"[watchdog] on_timeout 回调异常: {exc}")
        self._terminate()

    def _terminate(self) -> None:
        pid = os.getpid()
        kill_target = pid
        try:
            if self.kill_process_group:
                # 用进程组（负数 pid）确保连同子进程一起终止
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                self._log("[watchdog] 已向进程组发送 SIGTERM。")
            else:
                os.kill(pid, signal.SIGTERM)
                self._log("[watchdog] 已向主进程发送 SIGTERM。")
        except Exception as exc:  # noqa: BLE001
            self._log(f"[watchdog] 发送 SIGTERM 失败: {exc}")

        time.sleep(max(self.grace_seconds, 0))

        try:
            if self.kill_process_group:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            else:
                os.kill(kill_target, signal.SIGKILL)
            self._log("[watchdog] 宽限期结束，已发送 SIGKILL。")
        except Exception as exc:  # noqa: BLE001
            self._log(f"[watchdog] 发送 SIGKILL 失败（进程可能已退出）: {exc}")

        # 兜底：直接退出解释器
        os._exit(124)


def ensure_process_group() -> None:
    """让当前进程成为新的进程组组长，方便 watchdog 整组终止。"""
    try:
        os.setpgrp()
        print(f"[watchdog] 已创建独立进程组 (pgid={os.getpgrp()})")
    except Exception as exc:  # noqa: BLE001 - 某些环境不允许，忽略
        print(f"[watchdog] 创建进程组失败（忽略）: {exc}")


if __name__ == "__main__":  # 手工测试：python -m s1_eval.watchdog 3
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    wd = Watchdog(timeout_seconds=seconds, heartbeat_seconds=1)
    wd.start()
    print(f"模拟任务运行 {seconds + 3:.0f} 秒...")
    time.sleep(seconds + 3)
