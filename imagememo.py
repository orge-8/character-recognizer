# -*- coding: utf-8 -*-
"""会话最近图片记忆：``/识图修正`` 不引用消息时吃的"刚发的那几张"。

纯 Python 实现（时钟注入，可脱机单测），从 plugin.py 拆出（重构路线 A）。
kwargs → session_key 的提取留在 plugin.py（与 ``_stream_id`` 同源），本模块只管存取。
"""

import time
from typing import Callable

#: 每个会话保留的最近图片数。``/识图修正`` 不引用消息时用的就是这段历史——
#: 只留最后一张的话，"不引用"就等于"只能补一张卡"，跟要求引用没区别。
RECENT_IMAGE_LIMIT = 8
#: 这段历史的有效时间窗。没有它会出事：十分钟前发的图会在下一次命令里被当成"刚发的"。
RECENT_IMAGE_SECONDS = 900.0
#: 保留图片记忆的会话数上限（总字节数另有上限）。
RECENT_IMAGE_SESSIONS = 64
#: 总字节数上限：一张图动辄几 MB，不限就是几百 MB 常驻内存。
RECENT_IMAGE_MAX_BYTES = 32 * 1024 * 1024


class RecentImageStore:
    """按会话保留最近识别过的图片（带标签与时间戳），三重上限防内存膨胀。

    三重上限：单会话张数、会话数、总字节数。dict 保序，所以"最早插入的"就是最该
    淘汰的。时钟通过 ``clock`` 注入：测试喂假时钟即可验证过期与淘汰，不必真等 15 分钟。

    ``buffers`` 是有意公开的：plugin.py 把它别名成 ``_latest_images``——既有冒烟用例
    会绕过 ``remember`` 直接塞条目，别名保证两条路径写的是同一份数据。外部只该做
    读与条目赋值；清空请走 ``clear()``（就地清空，别名不失效）。
    """

    def __init__(
        self,
        *,
        per_session_limit: int = RECENT_IMAGE_LIMIT,
        max_age_seconds: float = RECENT_IMAGE_SECONDS,
        max_sessions: int = RECENT_IMAGE_SESSIONS,
        max_bytes: int = RECENT_IMAGE_MAX_BYTES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._per_session_limit = per_session_limit
        self._max_age_seconds = max_age_seconds
        self._max_sessions = max_sessions
        self._max_bytes = max_bytes
        self._clock = clock
        #: 会话键 → [(图片, 标签, 收到时刻)]。dict 保序 = 淘汰顺序。
        self.buffers: "dict[str, list[tuple[bytes, str, float]]]" = {}
        self._total_bytes = 0

    def remember(self, session_key: str, image: bytes, label: str) -> None:
        """记住这个会话刚识别过的图片，**保留一小段历史**。

        为什么不是只留最后一张：``/识图修正`` 不引用消息时用的是"刚发的那几张"，
        只留一张就永远只能补一张卡——那跟要求引用没区别。每条带时间戳，读的时候按
        时间窗过滤（见 ``recent``）。
        """
        buffer = self.buffers.setdefault(session_key, [])
        buffer.append((image, label, self._clock()))
        self._total_bytes += len(image)
        while len(buffer) > self._per_session_limit:
            data, _, _ = buffer.pop(0)
            self._total_bytes -= len(data)
        while len(self.buffers) > 1 and (
            len(self.buffers) > self._max_sessions
            or self._total_bytes > self._max_bytes
        ):
            oldest = next(iter(self.buffers))
            if oldest == session_key:  # 只剩当前会话时不再淘汰，否则刚存进来的当场被扔
                break
            dropped = self.buffers.pop(oldest)
            self._total_bytes -= sum(len(item[0]) for item in dropped)
        self._total_bytes = max(0, self._total_bytes)

    def recent(
        self, session_key: str, limit: int = 1, window: "float | None" = None
    ) -> "list[tuple[bytes, str, float]]":
        """取该会话最近 ``window`` 秒内的图片（新的在后），最多 ``limit`` 张。

        时间窗是必要的：没有它，十分钟前发的图会在下一次 ``/识图修正`` 里被当成
        "刚发的"用上，而用户完全看不出这些卡是从哪张图来的。``window`` 缺省用
        构造时的 ``max_age_seconds``。
        """
        horizon = self._max_age_seconds if window is None else window
        now = self._clock()
        fresh = [item for item in self.buffers.get(session_key, ()) if now - item[2] <= horizon]
        return fresh[-max(1, limit):]

    def session_keys(self) -> "list[str]":
        """按插入顺序返回全部会话键（前缀扫描用，如按 ``stream_id:`` 找整个会话）。"""
        return list(self.buffers)

    def clear(self) -> None:
        """就地清空：plugin.py 的 ``_latest_images`` 是本 dict 的别名，重建会让别名失效。"""
        self.buffers.clear()
        self._total_bytes = 0

    @property
    def total_count(self) -> int:
        """全部会话的图片总条数（状态行用）。"""
        return sum(len(buffer) for buffer in self.buffers.values())
