# -*- coding: utf-8 -*-
"""视觉限额的纯函数边界（脱机，不需要 FakeHost）。

这些函数之所以被抽成模块级纯函数，就是因为**它们必须只有一个真相来源**：
超时夹取与 token 下限都在运行期算，`/识图状态` 报的、日志提示的、真正传给模型的
必须是同一个数。真机踩过"说的和做的不一样"——配置写着 120、实际按 110 的预算走，
用户看完状态栏以为配置没生效，又去反复改配置。

所以这里的用例全部在钉"生效值本身"，而不是"有没有调用某个方法"。
"""

from __future__ import annotations

from types import SimpleNamespace

import plugin as plugin_module

NONE = object()  # 占位：表示"不设这个字段"


def _config(
    *,
    image_timeout: object = NONE,
    message_timeout: object = NONE,
    max_tokens: object = NONE,
) -> SimpleNamespace:
    """拼一个最小可用的配置替身（只需 ``plugin`` / ``vision`` 两节）。"""
    plugin_section: dict[str, object] = {}
    if image_timeout is not NONE:
        plugin_section["image_timeout_seconds"] = image_timeout
    if message_timeout is not NONE:
        plugin_section["message_timeout_seconds"] = message_timeout
    vision_section: dict[str, object] = {}
    if max_tokens is not NONE:
        vision_section["max_tokens"] = max_tokens
    return SimpleNamespace(plugin=SimpleNamespace(**plugin_section), vision=SimpleNamespace(**vision_section))


# --------------------------------------------------------------- 超时夹取


def test_timeout_clamped_when_over_budget() -> None:
    """真机实录：单图 120s > 消息预算 110s ⇒ 永远轮不到超时先触发。"""
    config = _config(image_timeout=120.0, message_timeout=110.0)
    effective = plugin_module.effective_image_timeout(config)
    assert effective == 110.0 * plugin_module.MESSAGE_BUDGET_RESERVE_RATIO
    assert effective < 110.0


def test_timeout_untouched_when_within_budget() -> None:
    """没超预算就不要动它——夹取只用来修坏掉的不变量，不是常态化改写配置。"""
    config = _config(image_timeout=50.0, message_timeout=110.0)
    assert plugin_module.effective_image_timeout(config) == 50.0


def test_timeout_falls_back_to_configured_when_budget_missing() -> None:
    """配置缺字段（老 config.toml + 热重载的中间态）时不能崩，也不能算出 1s。"""
    assert plugin_module.effective_image_timeout(_config(image_timeout=50.0)) == 50.0
    assert plugin_module.effective_image_timeout(_config(image_timeout=50.0, message_timeout=0)) == 50.0
    assert plugin_module.effective_image_timeout(_config()) == 0.0


def test_timeout_never_degenerates_to_zero() -> None:
    """夹取结果永远 >= 1s：算出 0 会让每次识别都瞬间超时，比不夹更糟。"""
    config = _config(image_timeout=170.0, message_timeout=10.0)
    assert plugin_module.effective_image_timeout(config) >= 1.0


# --------------------------------------------------------------- token 下限


def test_max_tokens_raised_to_floor() -> None:
    """真机真机实测 700 就触顶（推理模型的思考 token 也占这个上限），必须运行期抬起。"""
    assert plugin_module.effective_max_tokens(_config(max_tokens=700)) == plugin_module.VISION_MAX_TOKENS_FLOOR


def test_max_tokens_respects_larger_configured_value() -> None:
    """用户显式配得更大时不能反被压小。"""
    assert plugin_module.effective_max_tokens(_config(max_tokens=4096)) == 4096


def test_max_tokens_floor_is_meaningfully_above_the_observed_truncation() -> None:
    """下限要真的高于实测触顶值，否则这个兜底等于没兜。"""
    assert plugin_module.VISION_MAX_TOKENS_FLOOR > 700


def test_max_tokens_handles_missing_field() -> None:
    assert plugin_module.effective_max_tokens(_config()) == plugin_module.VISION_MAX_TOKENS_FLOOR


# --------------------------------------------------------------- 状态栏一致性


def test_limits_line_reports_effective_values_and_flags_clamping() -> None:
    config = _config(image_timeout=120.0, message_timeout=110.0, max_tokens=700)
    line = plugin_module._limits_line(config)
    effective_timeout = plugin_module.effective_image_timeout(config)
    assert f"{effective_timeout:.0f}s" in line
    assert f"{plugin_module.effective_max_tokens(config)}" in line
    assert "夹取" in line and "抬高" in line


def test_limits_line_is_quiet_when_nothing_was_adjusted() -> None:
    """全部生效即配置时不要加戏——常态日志/回执里多一行解释只会淹掉真正的异常。"""
    config = _config(image_timeout=50.0, message_timeout=110.0, max_tokens=4096)
    line = plugin_module._limits_line(config)
    assert "夹取" not in line and "抬高" not in line
