# -*- coding: utf-8 -*-
"""反查源解析与熔断行为的回归测试（全程离线）。"""

import asyncio
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

import sources  # noqa: E402
from models import SourceHit  # noqa: E402
from runtime import CircuitBreaker  # noqa: E402
from sources import SourceConfig, SourceError, parse_anime_trace, run_source  # noqa: E402


def test_parse_anime_trace_extracts_character_and_work() -> None:
    payload = {
        "data": [
            {
                "not_confident": False,
                "character": [
                    {"character": "アロナ", "work": "ブルーアーカイブ"},
                    {"character": "プラナ", "work": "ブルーアーカイブ"},
                ],
            }
        ]
    }
    hits = parse_anime_trace(payload)
    assert [hit.raw_name for hit in hits] == ["アロナ", "プラナ"]
    assert all(hit.work == "ブルーアーカイブ" for hit in hits)
    assert all(hit.confident for hit in hits)
    assert all(hit.source == "anime_trace" for hit in hits)


def test_parse_anime_trace_marks_low_confidence_and_sorts_it_last() -> None:
    """``not_confident`` 是条目级布尔。低置信信息仍要保留，但必须排在后面。"""
    payload = {
        "data": [
            {"not_confident": True, "character": [{"character": "猜的", "work": "某作品"}]},
            {"not_confident": False, "character": [{"character": "准的", "work": "某作品"}]},
        ]
    }
    hits = parse_anime_trace(payload)
    assert hits[0].raw_name == "准的"
    assert hits[0].confident is True
    assert hits[1].raw_name == "猜的"
    assert hits[1].confident is False


def test_parse_anime_trace_tolerates_garbage() -> None:
    for payload in ({}, {"data": None}, {"data": "x"}, {"data": [1, 2]},
                    {"data": [{"character": "not-a-list"}]},
                    {"data": [{"character": [{"character": "", "work": ""}]}]}):
        assert parse_anime_trace(payload) == ()


def test_parse_anime_trace_dedupes_and_respects_limit() -> None:
    entry = {"not_confident": False, "character": [{"character": "重复", "work": "作品"}]}
    assert len(parse_anime_trace({"data": [entry, entry]})) == 1
    many = {"data": [{"not_confident": False,
                      "character": [{"character": f"角色{i}", "work": "作品"} for i in range(10)]}]}
    assert len(parse_anime_trace(many, max_candidates=3)) == 3


def test_parse_anime_trace_skips_implausible_names() -> None:
    payload = {"data": [{"not_confident": False, "character": [
        {"character": "unknown", "work": "作品"},
        {"character": "1", "work": "作品"},
        {"character": "   ", "work": "作品"},
        {"character": "阿罗娜", "work": "作品"},
    ]}]}
    assert [hit.raw_name for hit in parse_anime_trace(payload)] == ["阿罗娜"]


# ---------------------------------------------------------------- run_source


def test_run_source_skips_when_disabled() -> None:
    config = SourceConfig(name="anime_trace", enabled=False, url="http://127.0.0.1:1")
    hits, error = asyncio.run(run_source(config, b"x"))
    assert hits == () and error == ""


def test_run_source_reports_unimplemented_adapter() -> None:
    config = SourceConfig(name="nonexistent", enabled=True)
    hits, error = asyncio.run(run_source(config, b"x"))
    assert hits == ()
    assert "未实现" in error


def test_run_source_absorbs_network_failure_and_trips_breaker() -> None:
    """连不上时必须返回可读错误并计入熔断，而不是把异常抛上去。"""

    async def failing(image_bytes, config):
        raise SourceError("连接失败：测试")

    original = dict(sources.ADAPTERS)
    sources.ADAPTERS["anime_trace"] = failing
    breaker = CircuitBreaker(failures=2, cooldown_seconds=60.0)
    try:
        config = SourceConfig(name="anime_trace", enabled=True)
        for _ in range(2):
            hits, error = asyncio.run(run_source(config, b"x", breaker=breaker))
            assert hits == () and "连接失败" in error
        assert breaker.allow("anime_trace") is False
        hits, error = asyncio.run(run_source(config, b"x", breaker=breaker))
        assert hits == () and "熔断" in error
    finally:
        sources.ADAPTERS.clear()
        sources.ADAPTERS.update(original)


def test_uncompressionable_image_does_not_trip_breaker() -> None:
    """压不下来是本地条件问题，不该把整个源停掉（否则用户以为服务挂了）。"""

    async def non_transient(image_bytes, config):
        raise SourceError("图片无法压到上传上限：环境缺少 Pillow", transient=False)

    original = dict(sources.ADAPTERS)
    sources.ADAPTERS["anime_trace"] = non_transient
    breaker = CircuitBreaker(failures=1, cooldown_seconds=60.0)
    try:
        config = SourceConfig(name="anime_trace", enabled=True)
        for _ in range(3):
            hits, error = asyncio.run(run_source(config, b"x", breaker=breaker))
            assert "Pillow" in error
        assert breaker.allow("anime_trace") is True, "非暂时性失败不应触发熔断"
    finally:
        sources.ADAPTERS.clear()
        sources.ADAPTERS.update(original)


def test_unexpected_exception_is_contained() -> None:
    async def broken(image_bytes, config):
        raise ValueError("adapter 内部崩了")

    original = dict(sources.ADAPTERS)
    sources.ADAPTERS["anime_trace"] = broken
    try:
        hits, error = asyncio.run(run_source(
            SourceConfig(name="anime_trace", enabled=True), b"x"
        ))
        assert hits == ()
        assert "ValueError" in error
    finally:
        sources.ADAPTERS.clear()
        sources.ADAPTERS.update(original)


def test_probe_endpoint_reports_failure_without_raising() -> None:
    config = SourceConfig(name="saucenao", enabled=True, url="http://127.0.0.1:1", timeout_seconds=1.0)
    summary = asyncio.run(sources.probe_endpoint(config, b"\x89PNG\r\n\x1a\n" + b"\x00" * 40))
    assert summary["ok"] is False
    assert summary["error"]


# ---------------------------------------------------------------- 限流识别

def test_rate_limit_reason_extracts_readable_message() -> None:
    """AnimeTrace 实测的 429 正文是多语言 JSON。

    整段丢给用户没有信息量（真机日志里就是这样一坨），要抠出那句中文。
    """
    detail = (
        '{"code":17737,"zh_message":"请求过于频繁，请稍后再试",'
        '"msg":"リクエストが多すぎます。しばらくしてからもう一度お試しください",'
        '"detail":"Too Many Requests"}'
    )
    assert sources._rate_limit_reason(detail) == "请求过于频繁，请稍后再试"


def test_rate_limit_reason_survives_truncated_or_empty_body() -> None:
    """正文被截断时 JSON 解析不出来，也必须给一句话而不是抛异常。"""
    assert sources._rate_limit_reason('{"code":17737,"zh_mess') == '{"code":17737,"zh_mess'
    assert sources._rate_limit_reason("") == "服务未给出原因"
    assert sources._rate_limit_reason("Too Many Requests") == "Too Many Requests"


def test_source_error_marks_rate_limited_separately() -> None:
    """限流必须能一路传到展示层：它是"这次没查成"，不是"图里没角色"。"""
    error = SourceError("被限流：请求过于频繁（HTTP 429）", rate_limited=True)
    assert error.rate_limited is True
    assert error.transient is True, "限流仍要计入熔断，否则会继续对着限流窗口打"
    assert SourceError("连接失败").rate_limited is False
    assert SourceError("图片无法压到上传上限", transient=False).rate_limited is False


def test_collect_hits_preserves_source_order() -> None:
    merged = sources.collect_hits(
        (SourceHit("anime_trace", raw_name="A"), SourceHit("anime_trace", raw_name="B")),
        (SourceHit("saucenao", raw_name="C"),),
    )
    assert [hit.raw_name for hit in merged] == ["A", "B", "C"]
