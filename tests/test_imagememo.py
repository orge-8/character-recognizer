# -*- coding: utf-8 -*-
"""imagememo.RecentImageStore 单元测试：三重上限 + 时间窗过滤。

时钟是注入的（``clock`` 参数），测试用假时钟推进时间，不必真等 15 分钟。
"""

from imagememo import RecentImageStore


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_per_session_limit_evicts_oldest() -> None:
    """单会话张数上限：超出后最旧的被淘汰，新的保留。"""
    store = RecentImageStore(per_session_limit=3, clock=FakeClock())
    for index in range(5):
        store.remember("s:u", f"图{index}".encode(), f"标签{index}")
    kept = store.recent("s:u", limit=10)
    assert [item[1] for item in kept] == ["标签2", "标签3", "标签4"]
    assert store.total_count == 3


def test_recent_filters_by_time_window() -> None:
    """时间窗：超出 window 的旧图不该被当成「刚发的」。"""
    clock = FakeClock()
    store = RecentImageStore(max_age_seconds=900.0, clock=clock)
    store.remember("s:u", b"old", "旧图")
    clock.now += 800.0
    store.remember("s:u", b"fresh", "新图")
    assert [item[1] for item in store.recent("s:u", limit=10)] == ["旧图", "新图"]
    clock.now += 200.0  # 旧图距现在 1000s，出窗
    recent = store.recent("s:u", limit=10)
    assert [item[1] for item in recent] == ["新图"]
    # 显式 window 覆盖构造默认值
    assert store.recent("s:u", limit=10, window=0.0) == []


def test_max_bytes_evicts_oldest_session() -> None:
    """总字节上限：超限时淘汰最早的会话，但**不淘汰刚写入的当前会话**。"""
    store = RecentImageStore(max_bytes=100, clock=FakeClock())
    store.remember("s1:u", b"x" * 60, "s1")
    store.remember("s2:u", b"x" * 60, "s2")  # 总 120 > 100，s1 应被淘汰
    assert store.session_keys() == ["s2:u"]
    # 只剩当前会话时不再淘汰（否则刚存进来的当场被扔）
    store.remember("s2:u", b"y" * 200, "s2-more")
    assert store.session_keys() == ["s2:u"]
    assert store.total_count == 2


def test_max_sessions_evicts_oldest_session() -> None:
    """会话数上限：超出后按插入顺序淘汰最早的会话。"""
    store = RecentImageStore(max_sessions=2, clock=FakeClock())
    store.remember("s1:u", b"a", "1")
    store.remember("s2:u", b"b", "2")
    store.remember("s3:u", b"c", "3")
    assert store.session_keys() == ["s2:u", "s3:u"]
    assert store.recent("s1:u", limit=1) == []


def test_clear_resets_and_keeps_alias_valid() -> None:
    """clear 是就地清空：持有 buffers 别名的外部（plugin._latest_images）不失效。"""
    store = RecentImageStore(clock=FakeClock())
    alias = store.buffers
    store.remember("s:u", b"a", "1")
    store.clear()
    assert store.total_count == 0
    assert alias == {}
    store.remember("s:u", b"b", "2")
    assert len(alias) == 1  # 别名与 store 仍是同一份数据
