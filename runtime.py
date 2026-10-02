# -*- coding: utf-8 -*-
"""进程内运行时设施：LRU+TTL 缓存、熔断器与后台任务登记册（不依赖 ctx）。"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable


class TTLCache:
    """带容量上限与过期时间的缓存。

    容量必须有界：识别缓存里存的是"图片哈希 → 结论"，无界增长会在长期运行的 bot 上
    慢慢吃光内存。淘汰策略是 LRU（``OrderedDict`` 的天然顺序）。
    """

    def __init__(self, *, max_entries: int = 256, ttl_seconds: float = 3600.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._max_entries = max(1, max_entries)
        self._ttl = max(0.0, ttl_seconds)
        self._clock = clock
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at and self._clock() >= expires_at:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def put(self, key: str, value: Any, *, ttl_seconds: float | None = None) -> None:
        ttl = self._ttl if ttl_seconds is None else max(0.0, ttl_seconds)
        expires_at = self._clock() + ttl if ttl else 0.0
        if key in self._store:
            del self._store[key]
        self._store[key] = (expires_at, value)
        while len(self._store) > self._max_entries:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


@dataclass
class _BreakerState:
    consecutive_failures: int = 0
    open_until: float = 0.0


class CircuitBreaker:
    """连续失败达到阈值后暂停调用一段时间，随后自动恢复试探。

    没有这一层，一个挂掉的反查源会在每条带图消息上都把超时耗满，把整条链路拖垮。
    """

    def __init__(self, *, failures: int = 2, cooldown_seconds: float = 60.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._threshold = max(1, failures)
        self._cooldown = max(0.0, cooldown_seconds)
        self._clock = clock
        self._states: dict[str, _BreakerState] = {}

    def allow(self, service: str) -> bool:
        state = self._states.get(service)
        if state is None or not state.open_until:
            return True
        if self._clock() >= state.open_until:
            # 冷却结束，放行一次做试探；成功就会把计数清零。
            state.open_until = 0.0
            return True
        return False

    def record_success(self, service: str) -> None:
        state = self._states.setdefault(service, _BreakerState())
        state.consecutive_failures = 0
        state.open_until = 0.0

    def record_failure(self, service: str) -> None:
        state = self._states.setdefault(service, _BreakerState())
        state.consecutive_failures += 1
        if state.consecutive_failures >= self._threshold:
            state.open_until = self._clock() + self._cooldown

    def is_open(self, service: str) -> bool:
        state = self._states.get(service)
        return bool(state and state.open_until and self._clock() < state.open_until)

    def remaining_cooldown(self, service: str) -> float:
        state = self._states.get(service)
        if state is None or not state.open_until:
            return 0.0
        return max(0.0, state.open_until - self._clock())

    def reset(self, service: str | None = None) -> None:
        if service is None:
            self._states.clear()
        else:
            self._states.pop(service, None)


class TaskRegistry:
    """后台任务登记册：防僵尸任务、防「exception was never retrieved」、超预算取消。

    不碰 ctx：日志通过构造时注入的 ``log`` 回调上报（签名与 ``logging`` 一致，
    plugin.py 直接把 ``_log`` 传进来）。
    """

    def __init__(self, log: Callable[..., None] | None = None) -> None:
        self._log = log
        #: 有意公开：plugin.py 把它别名成 ``_tasks``，既有测试直接 gather 这个集合
        #: 等后台任务收尾。清空必须走 ``cancel_all``/``clear``（就地），别让别名失效。
        self.tasks: set[asyncio.Task] = set()

    def track(self, task: asyncio.Task) -> None:
        self.tasks.add(task)
        task.add_done_callback(self.retire)

    def retire(self, task: asyncio.Task) -> None:
        """任务结束后摘掉引用，并**把异常取出来**。

        只 ``discard`` 不取异常的话，被取消/失败的任务会在 GC 时报
        "Task exception was never retrieved"，看起来像又出了一条新故障。
        """
        self.tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.exception()
        except asyncio.CancelledError:  # pragma: no cover - 竞态兜底
            pass

    async def cancel_all(self) -> None:
        """取消全部登记任务并等它们收尾（on_unload 用）。就地清空，别名不失效。"""
        for task in list(self.tasks):
            if not task.done():
                task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()

    async def await_with_budget(
        self, tasks: "list[asyncio.Task]", budget: float
    ) -> "set[asyncio.Task]":
        """等这批识别任务到 ``budget`` 秒，**超预算的直接取消**，返回已完成的那批。

        取消是必须的，不是"顺手清理"：超预算的图已经把结论交给 MaiBot（原图放行），它再跑
        下去没有任何人会读它的结果，却会

        * 继续占着 ``max_concurrency`` 的信号量——后面的消息全排在它后面，一条慢图能把
          接下来几分钟的识图全拖住；
        * 继续吃 Host 的模型配额与回退链（真机实录：13:14:03 判定超预算，那张图的任务到
          13:14:18 才报"视觉请求超时"，白跑 15 秒）；
        * 在日志里留下一句看起来像本轮结论、其实是弃单的报错。

        早先的写法只 ``track`` 登记不取消，等于把任务留成僵尸。
        """
        done, pending = await asyncio.wait(tasks, timeout=budget)
        if not pending:
            return done
        # 一条日志说清整件事：以前是每个未完成任务各打一条，多图时刷屏还看不出总量。
        if self._log is not None:
            self._log(
                "warning",
                "单条消息识别超预算（%.0fs）：%d 张图未完成，已取消识别并原样交给 MaiBot",
                budget, len(pending),
            )
        for task in pending:
            task.cancel()
            self.track(task)
        return done
