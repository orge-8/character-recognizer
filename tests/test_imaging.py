# -*- coding: utf-8 -*-
"""``prepare_upload_async`` 的记忆化与线程化回归（全程离线，不依赖 Pillow）。"""

import asyncio
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

import imaging  # noqa: E402


def test_prepare_upload_async_passes_small_images_through() -> None:
    raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
    data, mime = asyncio.run(imaging.prepare_upload_async(raw, max_bytes=1_000_000))
    assert data is raw, "没超限就不该重编码，白白损失画质"
    assert mime == "image/png"


def test_prepare_upload_async_rejects_empty() -> None:
    try:
        asyncio.run(imaging.prepare_upload_async(b"", max_bytes=1024))
    except ValueError:
        return
    raise AssertionError("空图必须抛 ValueError")


def test_prepare_upload_async_memoizes_per_size_limit() -> None:
    """同一张图同一档上限只压一次：describe / identify / 反查源共享压缩结果。"""
    imaging._UPLOAD_CACHE = None  # 隔离模块级缓存
    calls = {"count": 0}
    original = imaging.prepare_upload

    def counting(image_bytes, *, max_bytes, edge_ladder=imaging.EDGE_LADDER,
                 quality_ladder=imaging.QUALITY_LADDER):
        calls["count"] += 1
        return b"compressed", "image/jpeg"

    imaging.prepare_upload = counting
    try:
        raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 4096
        first = asyncio.run(imaging.prepare_upload_async(raw, max_bytes=1024))
        second = asyncio.run(imaging.prepare_upload_async(raw, max_bytes=1024))
        third = asyncio.run(imaging.prepare_upload_async(raw, max_bytes=2048))
    finally:
        imaging.prepare_upload = original
        imaging._UPLOAD_CACHE = None

    assert first == second == third == (b"compressed", "image/jpeg")
    assert calls["count"] == 2, f"同档应命中缓存、不同档各压一次，实际压了 {calls['count']} 次"
