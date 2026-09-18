# -*- coding: utf-8 -*-
"""进程内运行时设施：LRU+TTL 缓存与熔断器（不依赖 ctx）。"""

from __future__ import annotations

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
