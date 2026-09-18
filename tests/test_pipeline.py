# -*- coding: utf-8 -*-
"""图片处理、消息注入、提示词与运行时的纯函数回归（全程离线）。"""

import asyncio
import base64
import json
import sys
import zlib
import struct
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

import imaging  # noqa: E402
import inject  # noqa: E402
import prompts  # noqa: E402
import runtime as runtime_module  # noqa: E402
import textutil  # noqa: E402
import vision  # noqa: E402
from models import FusionResult, SourceHit, TIER_CONFLICT  # noqa: E402


def make_png(width: int = 2, height: int = 2, rgb=(255, 0, 0)) -> bytes:
    """纯标准库造一个合法 PNG（不依赖 Pillow）。"""
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        payload = tag + data
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


# ---------------------------------------------------------------- imaging

def test_sniff_mime_type_by_magic_number() -> None:
    assert imaging.sniff_mime_type(make_png()) == "image/png"
    assert imaging.sniff_mime_type(b"\xff\xd8\xff\xe0rest") == "image/jpeg"
    assert imaging.sniff_mime_type(b"GIF89a....") == "image/gif"
    assert imaging.sniff_mime_type(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert imaging.sniff_mime_type(b"unknown") == "image/png"


def test_decode_base64_accepts_data_url_and_whitespace() -> None:
    raw = make_png()
    encoded = base64.b64encode(raw).decode()
    assert imaging.decode_base64_image(encoded) == raw
    assert imaging.decode_base64_image(f"data:image/png;base64,{encoded}") == raw
    assert imaging.decode_base64_image("!!!not base64!!!") is None
    assert imaging.decode_base64_image(None) is None
    assert imaging.decode_base64_image("") is None


def test_prepare_upload_passes_small_images_through() -> None:
    raw = make_png()
    data, mime = imaging.prepare_upload(raw, max_bytes=1_000_000)
    assert data is raw, "没超限就不该重编码，白白损失画质"
    assert mime == "image/png"


def test_prepare_upload_fails_readably_without_pillow() -> None:
    """环境里没有 Pillow 时必须给一句能照做的中文错误，而不是 ImportError 堆栈。"""
    oversized = b"\x89PNG\r\n\x1a\n" + b"\x00" * 4096
    try:
        import PIL  # noqa: F401
    except ImportError:
        try:
            imaging.prepare_upload(oversized, max_bytes=1024)
        except ValueError as exc:
            assert "Pillow" in str(exc)
            return
        raise AssertionError("超限且无 Pillow 时应抛可读的 ValueError")
    # 有 Pillow 时，非图片数据应给出"无法解析"的可读错误
    try:
        imaging.prepare_upload(oversized, max_bytes=1024)
    except ValueError as exc:
        assert "压缩" in str(exc) or "解析" in str(exc)


def test_extract_image_payload_reports_what_is_available() -> None:
    encoded = base64.b64encode(make_png()).decode()
    data, note = imaging.extract_image_payload({"type": "image", "binary_data_base64": encoded})
    assert data is not None and "binary_data_base64" in note

    data, note = imaging.extract_image_payload({"type": "image", "url": "https://example.com/a.png"})
    assert data is None and "需下载" in note

    data, note = imaging.extract_image_payload({"type": "image"})
    assert data is None and "字段=" in note

    assert imaging.extract_image_payload("not-a-dict")[0] is None


def test_download_image_rejects_non_http() -> None:
    assert imaging.download_image("file:///etc/passwd") is None
    assert imaging.download_image("") is None


# ---------------------------------------------------------------- inject

def test_build_system_item_shape() -> None:
    item = inject.build_system_item("hello")
    assert item["item_type"] == "SystemMessageItem"
    assert "logical_turn_id" in item["meta"], "缺这个键会被 Host 校验拒掉"
    assert item["meta"]["logical_turn_id"] is None
    assert item["meta"]["item_id"]
    assert item["parts"] == [{"type": "text", "text": "hello"}]


def test_inject_into_items_is_idempotent_and_echoes_schema_version() -> None:
    marker = "【test】"
    kwargs = {"items": [], "item_schema_version": 1}
    assert inject.inject_into_items(kwargs, f"{marker} 内容", marker).startswith("已注入")
    assert len(kwargs["items"]) == 1
    assert kwargs["item_schema_version"] == 1

    before = len(kwargs["items"])
    assert "幂等" in inject.inject_into_items(kwargs, f"{marker} 内容", marker)
    assert len(kwargs["items"]) == before


def test_inject_skips_unsupported_schema_version() -> None:
    kwargs = {"items": [], "item_schema_version": 99}
    status = inject.inject_into_items(kwargs, "内容", "【x】")
    assert "跳过" in status and kwargs["items"] == []


def test_apply_injection_falls_back_across_payload_shapes() -> None:
    marker = "【x】"
    messages = {"messages": [{"role": "user", "content": "hi"}]}
    assert inject.apply_injection(messages, f"{marker} 资料", marker).startswith("已注入")
    assert messages["messages"][-1]["role"] == "system"

    plain = {"prompt": "原始提示词"}
    inject.apply_injection(plain, f"{marker} 资料", marker)
    assert plain["prompt"].startswith("原始提示词") and marker in plain["prompt"]

    # 三种形态都不匹配时要明说跳过，而不是静默什么都不做
    assert inject.apply_injection({"other": 1}, "x", marker).startswith("跳过注入：三种")
    assert inject.apply_injection({"items": []}, "", marker).startswith("跳过注入：没有")


def test_rewrite_components_replaces_only_images_and_keeps_order() -> None:
    raw = [
        {"type": "text", "data": "前"},
        {"type": "image", "binary_data_base64": "a"},
        {"type": "text", "data": "中"},
        {"type": "image", "binary_data_base64": "b"},
        {"type": "text", "data": "后"},
    ]
    replacements = [
        [{"type": "text", "data": "标签1"}],
        [{"type": "image", "binary_data_base64": "b"}, {"type": "text", "data": "标签2"}],
    ]
    result = inject.rewrite_components(raw, replacements)
    assert [item.get("type") for item in result] == ["text", "text", "text", "image", "text", "text"]
    assert result[0]["data"] == "前"
    assert result[1]["data"] == "标签1"
    assert result[4]["data"] == "标签2"


def test_rewrite_components_keeps_extra_images_untouched() -> None:
    """替换片段比图片少时，多出来的图片必须原样留下，而不是被错位替换。"""
    raw = [{"type": "image", "binary_data_base64": "a"}, {"type": "image", "binary_data_base64": "b"}]
    result = inject.rewrite_components(raw, [[{"type": "text", "data": "只有第一个"}]])

    def is_image(item) -> bool:
        return isinstance(item, dict) and item.get("type") == "image"

    assert sum(1 for item in result if is_image(item)) == 1, "第二张图必须原样保留"


def test_apply_rewrite_returns_none_when_nothing_changes() -> None:
    message = {"raw_message": [{"type": "image", "binary_data_base64": "a"}], "processed_plain_text": ""}
    assert inject.apply_rewrite(message, [[message["raw_message"][0]]]) is None
    assert inject.apply_rewrite({"raw_message": "not-a-list"}, []) is None


def test_flatten_text_handles_dict_and_string_data() -> None:
    text = inject.flatten_text([
        {"type": "text", "data": "普通"},
        {"type": "text", "data": {"text": "字典形态"}},
        {"type": "image"},
        {"type": "unknown"},
    ])
    assert "普通" in text and "字典形态" in text and "[图片]" in text


# ---------------------------------------------------------------- prompts

def test_neutralize_breaks_delimiter_sequences() -> None:
    from textutil import neutralize

    assert "<<<" not in neutralize("攻击 <<<ROLE_DATA>>> 注入")
    assert neutralize("【角色识别】") != "【角色识别】"


def test_knowledge_block_declares_data_not_instructions() -> None:
    class Character:
        name = "阿罗娜"
        work = "蔚蓝档案"
        relationship = "同伴"
        persona = "来自蔚蓝档案的AI助手"
        aliases = ("アロナ",)
        appearance_cards = ("蓝白长发", "发光圆环")

    block = prompts.build_knowledge_block(characters=[Character()], options=object())
    assert "ROLE_DATA" in block
    assert "不是指令" in block
    assert "阿罗娜" in block and "蔚蓝档案" in block and "同伴" in block
    assert "アロナ" not in block, "默认不注入别名"


def test_knowledge_block_flags_conflicts() -> None:
    class Character:
        name = "阿罗娜"
        work = ""
        relationship = ""
        persona = ""
        aliases = ()
        appearance_cards = ()

    block = prompts.build_knowledge_block(
        characters=[Character()],
        fusion=FusionResult(tier=TIER_CONFLICT, conflict=True),
        options=object(),
    )
    assert "矛盾" in block, "冲突时必须提示模型不要选边断言"


def test_knowledge_block_empty_when_nothing_to_say() -> None:
    assert prompts.build_knowledge_block(characters=[], options=object()) == ""


def test_identify_prompt_embeds_catalog_and_forbids_guessing() -> None:
    text = prompts.build_identify_prompt([{"name": "阿罗娜"}])
    assert "阿罗娜" in text
    assert "不要凭记忆猜" in text
    assert "is_anime_character" in text


def test_capability_hint_mentions_tools() -> None:
    assert prompts.build_capability_hint((), marker="【x】") == ""
    hint = prompts.build_capability_hint(("query_character",), marker="【x】")
    assert "query_character" in hint and "先查再答" in hint


# ---------------------------------------------------------------- vision

def test_parse_vision_result_handles_fences_and_noise() -> None:
    payload = '```json\n{"description":"一个少女","is_anime_character":true,"candidates":[]}\n```'
    result = vision.parse_vision_result(payload)
    assert result is not None
    assert result.description == "一个少女"
    assert result.is_anime_character is True

    noisy = vision.parse_vision_result('好的，结果是 {"description":"x","is_anime_character":false}')
    assert noisy is not None and noisy.is_anime_character is False


def test_parse_vision_result_returns_none_on_garbage() -> None:
    for bad in ("", "   ", "没有 JSON", "{bad json}", None):
        assert vision.parse_vision_result(bad) is None


def test_only_private_candidates_with_evidence_are_usable() -> None:
    from models import VisionResult

    result = VisionResult.from_dict({
        "description": "x",
        "is_anime_character": True,
        "candidates": [
            {"kind": "unknown", "name": "公开角色", "evidence": ["a", "b"]},
            {"kind": "private", "name": "证据不足", "evidence": ["a"]},
            {"kind": "private", "name": "有冲突", "evidence": ["a", "b"], "conflicts": ["发色不符"]},
            {"kind": "private", "name": "可用", "evidence": ["发色", "头饰"]},
        ],
    })
    assert result is not None
    assert result.candidate_names == ("可用",)


def test_vision_result_rejects_missing_description() -> None:
    from models import VisionResult

    assert VisionResult.from_dict({"is_anime_character": True}) is None
    assert VisionResult.from_dict("not-a-dict") is None
    # is_anime_character 缺失时降级为 False，而不是丢掉整条结果
    result = VisionResult.from_dict({"description": "只有描述"})
    assert result is not None and result.is_anime_character is False


# ---------------------------------------------------------------- runtime

def test_ttl_cache_expires_and_evicts_lru() -> None:
    now = {"value": 0.0}
    cache = runtime_module.TTLCache(max_entries=2, ttl_seconds=10.0, clock=lambda: now["value"])
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    cache.put("c", 3)  # 容量 2，最久未用的 b 应被淘汰
    assert cache.get("b") is None
    assert cache.get("a") == 1 and cache.get("c") == 3

    now["value"] = 100.0
    assert cache.get("a") is None, "过期条目必须失效"


def test_ttl_cache_zero_ttl_never_expires() -> None:
    cache = runtime_module.TTLCache(ttl_seconds=0.0)
    cache.put("k", "v")
    assert cache.get("k") == "v"


def test_circuit_breaker_opens_and_recovers() -> None:
    now = {"value": 0.0}
    breaker = runtime_module.CircuitBreaker(failures=2, cooldown_seconds=30.0, clock=lambda: now["value"])
    assert breaker.allow("svc") is True
    breaker.record_failure("svc")
    assert breaker.allow("svc") is True, "未到阈值不该熔断"
    breaker.record_failure("svc")
    assert breaker.allow("svc") is False
    assert breaker.remaining_cooldown("svc") > 0

    now["value"] = 31.0
    assert breaker.allow("svc") is True, "冷却结束应放行一次做试探"
    breaker.record_success("svc")
    assert breaker.allow("svc") is True


def test_circuit_breaker_resets_on_success() -> None:
    breaker = runtime_module.CircuitBreaker(failures=3, cooldown_seconds=60.0)
    breaker.record_failure("svc")
    breaker.record_success("svc")
    breaker.record_failure("svc")
    assert breaker.allow("svc") is True, "成功应清零失败计数"
    breaker.reset("svc")
    assert breaker.allow("svc") is True


# ---------------------------------------------------------------- 视觉候选丢弃原因

def test_vision_dropout_reasons_are_distinguishable() -> None:
    """「没有可用候选」下面藏着四种原因，处理方式各不相同，不能报成同一句。"""
    from models import VisionCandidate, VisionResult

    empty = VisionResult(description="x")
    assert "没有给出任何候选" in vision.describe_vision_dropouts(empty)

    unknown_kind = VisionResult(candidates=(VisionCandidate(kind="unknown", name="鸣澜",
                                                            evidence=("a", "b")),))
    assert "kind=unknown" in vision.describe_vision_dropouts(unknown_kind)

    conflicted = VisionResult(candidates=(VisionCandidate(kind="private", name="鸣澜",
                                                          evidence=("a", "b"),
                                                          conflicts=("发饰颜色不同",)),))
    text = vision.describe_vision_dropouts(conflicted)
    assert "冲突特征" in text and "发饰颜色不同" in text

    thin = VisionResult(candidates=(VisionCandidate(kind="private", name="鸣澜",
                                                    evidence=("只有一条",)),))
    assert "证据不足" in vision.describe_vision_dropouts(thin)


def test_parse_vision_result_keeps_raw_output_for_diagnosis() -> None:
    """原始输出必须留下来：候选被丢弃时，要看得到模型到底写了什么。"""
    content = ('```json\n{"description":"蓝发少女","is_anime_character":true,'
               '"candidates":[{"kind":"private","name":"鸣澜","evidence":["蓝发","金瞳"]}]}\n```')
    result = vision.parse_vision_result(content)
    assert result is not None
    assert result.candidate_names == ("鸣澜",)
    assert "蓝发少女" in result.raw
    assert vision.parse_vision_result("完全不是 JSON") is None


def test_identify_prompt_tells_model_missing_is_not_conflict() -> None:
    """提示词必须点明「画像里有但图上没看到」不算矛盾。

    不写这一句，模型会把"这张图看不到头饰"填进 conflicts，而 usable_name 只要看到
    任何 conflicts 就作废整个候选——结果是"图片与画像高度吻合却永远匹配不上"。
    """
    text = prompts.build_identify_prompt([{"name": "鸣澜", "appearance_cards": ["蓝发"]}])
    assert "不算矛盾" in text
    assert "至少 2 条" in text


# ------------------------------------------- unknown 候选救援（2026-09-18 真机回归）

#: 真机原始输出（09-18 07:50）。模型把「鸣澜」的 5 条外观卡**逐字**抄成了 evidence、
#: conflicts 留空，却填 kind=unknown、name="" —— 一个字段没填，整条候选作废。
REAL_UNKNOWN_OUTPUT = (
    '{"description":"蓝发少女，金色瞳孔，身穿深蓝色短袖制服配白色荷叶边领口与蝴蝶结，'
    '白色百褶裙，黑色玛丽珍鞋，左耳后有黑色圆形饰品，右手举至额前敬礼姿势。",'
    '"is_anime_character":true,"candidates":[{"kind":"unknown","name":"","work":"",'
    '"evidence":["浅蓝色长发，刘海整齐覆盖额头，发尾微卷",'
    '"金色瞳孔，眼神明亮清晰",'
    '"深蓝色短袖制服上衣，领口有白色荷叶边和蝴蝶结",'
    '"白色百褶短裙，搭配白色中筒袜和黑色玛丽珍鞋",'
    '"头顶有黑色圆形耳机状饰品，带有发光蓝环"],"conflicts":[]}]}'
)

MINGLAN_CATALOG = [{
    "id": "char-minglan",
    "name": "鸣澜",
    "aliases": [],
    "work": "",
    "appearance_cards": [
        "浅蓝色长发，刘海整齐覆盖额头，发尾微卷",
        "金色瞳孔，眼神明亮清晰",
        "头顶有黑色圆形耳机状饰品，带有发光蓝环",
        "深蓝色短袖制服上衣，领口有白色荷叶边和蝴蝶结",
        "白色百褶短裙，搭配白色中筒袜和黑色玛丽珍鞋",
    ],
}]


def test_rescue_recovers_candidate_whose_evidence_matches_cards() -> None:
    """真机回归：证据逐条对上外观卡时，不能因为模型没填 name 就把整条候选扔掉。"""
    parsed = vision.parse_vision_result(REAL_UNKNOWN_OUTPUT)
    assert parsed is not None
    assert parsed.candidate_names == (), "原始解析结果确实是空——模型填了 unknown"

    rescued, note = vision.rescue_unlabeled_candidates(parsed, MINGLAN_CATALOG)
    assert rescued.candidate_names == ("鸣澜",)
    assert "鸣澜" in note and "5 条" in note


def test_rescue_needs_real_card_overlap_not_generic_words() -> None:
    """泛泛的特征（"长发""少女"）不能拿来认人，否则等于没有门槛。"""
    parsed = vision.parse_vision_result(
        '{"description":"x","is_anime_character":true,"candidates":'
        '[{"kind":"unknown","name":"","evidence":["长发","少女"],"conflicts":[]}]}'
    )
    rescued, note = vision.rescue_unlabeled_candidates(parsed, MINGLAN_CATALOG)
    assert rescued.candidate_names == ()
    assert note == ""


def test_rescue_refuses_candidates_that_declare_conflicts() -> None:
    """模型说"有矛盾"就不能救——那是它在明确表达不同意见。"""
    parsed = vision.parse_vision_result(
        '{"description":"x","is_anime_character":true,"candidates":'
        '[{"kind":"unknown","name":"","evidence":["浅蓝色长发，刘海整齐覆盖额头，发尾微卷",'
        '"金色瞳孔，眼神明亮清晰"],"conflicts":["发色明显不同"]}]}'
    )
    rescued, _ = vision.rescue_unlabeled_candidates(parsed, MINGLAN_CATALOG)
    assert rescued.candidate_names == ()


def test_rescue_fixes_kind_when_name_is_already_in_library() -> None:
    """名字逐字在库、只是 kind 标成了 unknown —— 直接认下。"""
    parsed = vision.parse_vision_result(
        '{"description":"x","is_anime_character":true,"candidates":'
        '[{"kind":"unknown","name":"鸣澜",'
        '"evidence":["浅蓝色长发，刘海整齐覆盖额头，发尾微卷",'
        '"金色瞳孔，眼神明亮清晰"],"conflicts":[]}]}'
    )
    rescued, note = vision.rescue_unlabeled_candidates(parsed, MINGLAN_CATALOG)
    assert rescued.candidate_names == ("鸣澜",)
    assert "kind" in note


def test_local_confirm_default_threshold_leaves_room_for_score_jitter() -> None:
    """默认阈值必须给检索分波动留余量，别贴着观测带设。

    真机实测（09-18）：同一张图、同一份外观卡，检索分在 0.34~0.40 之间浮动
    （``description`` 由视觉模型每次现生成，措辞一变向量就变）。阈值设成 0.35
    会把 0.34 那次挡掉——表现是"同一张图有时贴得上、有时贴不上"。
    """
    plugin = _plugin_module()
    default = plugin.CharacterRecognizerConfig().fusion.local_confirm_min_score
    assert default <= 0.30, f"默认 {default} 落在观测波动带（0.34~0.40）里，会导致时贴时不贴"
    assert default >= 0.20, f"默认 {default} 太低，会让「只有向量略像」的候选也过关"


def test_identify_prompt_forbids_matching_on_hair_color_alone() -> None:
    """「任意 2 条吻合」会让模型挑最省力的两条（发色 + 瞳色）就认人。

    真机教训（09-18 09:21）：一张**黑白女仆装**的蓝发少女被认成「鸣澜」——模型只给了
    "浅蓝色长发""金色瞳孔"两条证据（全落在发色发型与眼睛两类），压根没提服装。
    而鸣澜是深蓝制服配白百褶裙。提示词必须要求三类都对得上，并点明只看发色瞳色是错的。
    """
    text = prompts.build_identify_prompt(MINGLAN_CATALOG)
    assert "服装主调" in text
    assert "只看发色或瞳色就认人是最常见的错误" in text
    assert "女仆装" in text, "要给具体反例，否则模型不知道什么叫「明显不同」"
    assert "任何一类明显不同" in text


def test_identify_prompt_defines_when_to_label_private() -> None:
    """提示词必须说清"什么情况下该填 private"。

    只说"private 时 name 要逐字来自库"，模型会理解成"我保证不了逐字 → 填 unknown"，
    于是明明认出来（证据全对）也不给名字。
    """
    text = prompts.build_identify_prompt(MINGLAN_CATALOG)
    assert "先比对特征再定 kind" in text
    assert "不要留空" in text
    assert "都对不上" in text


# ------------------------------------------------------ 增量抽卡与卡片整理

def test_appearance_prompt_asks_only_for_new_features_when_library_has_cards() -> None:
    """建卡提示词必须带上已有卡片。

    不带的话，第二张图必然把"浅蓝色长发"再写一遍——攒到第十张就全是措辞变体，
    真机上 15 条里有一半是这么来的。
    """
    without = prompts.build_appearance_prompt()
    assert "只输出它们没覆盖到的新特征" not in without

    with_cards = prompts.build_appearance_prompt(["浅蓝色长发，刘海整齐", "金色瞳孔"])
    assert "只输出它们没覆盖到的新特征" in with_cards
    assert "浅蓝色长发，刘海整齐" in with_cards and "金色瞳孔" in with_cards
    assert "输出空数组" in with_cards, "没有新特征时要说得出「空数组」，否则模型会硬编一条"


def test_compress_prompt_pins_the_two_deal_breakers() -> None:
    """压缩提示词里那两条约束是分界线：不合并造型、不丢细节。

    真机卡片里「蓝色短发」与「浅蓝色长直发」是**两种造型**。无约束的"精简"会把它们并成
    一条，丢掉造型多样性——而识别时正是靠它判断当前这张图属于哪个形态。
    """
    text = prompts.build_compress_prompt(["蓝色短发，带刘海", "浅蓝色长直发，齐刘海"])
    assert "不同造型各自成条，绝不合并" in text
    assert "保留全部区分性细节" in text
    assert "蓝色短发，带刘海" in text and "1." in text


def _compress(cards: list[str], generate, **kwargs):
    return asyncio.run(vision.compress_appearance_cards(
        cards=cards, provider=kwargs.pop("provider", "host"), generate=generate, **kwargs
    ))


def test_compress_is_skipped_when_there_is_nothing_to_gain() -> None:
    """条数不够、通道不支持时直接跳过——不该为没收益的场景花一次调用。"""

    async def never_called(**kwargs):  # pragma: no cover - 被调用就说明判据错了
        raise AssertionError("不该发起调用")

    merged, note = _compress(["浅蓝色长发", "金色瞳孔"], never_called)
    assert merged is None and "不足" in note

    merged, note = _compress([f"卡片 {index}" for index in range(10)], never_called,
                             provider="gemini")
    assert merged is None and "host" in note


def test_compress_rejects_suspicious_results() -> None:
    """结果不合理时宁可不采用：条数没减少＝没干活，少于 3 条＝把造型也合掉了。"""
    cards = [f"卡片 {index}" for index in range(10)]

    def make_generate(payload: dict):
        async def _generate(**kwargs):
            return {"success": True, "response": json.dumps(payload, ensure_ascii=False)}
        return _generate

    merged, note = _compress(cards, make_generate({"appearance_cards": cards}))
    assert merged is None and "没减少" in note

    merged, note = _compress(cards, make_generate({"appearance_cards": ["只剩一条了"]}))
    assert merged is None and "造型差异" in note

    merged, note = _compress(cards, make_generate({"appearance_cards": ["合并甲", "合并乙", "合并丙"]}))
    assert merged == ["合并甲", "合并乙", "合并丙"] and note == ""


def test_compress_reports_why_it_did_not_apply() -> None:
    """失败原因必须分开报：超时 / 解析失败 / 调用失败，处理方式完全不同。

    真机第一次跑就栽在这一点——模型耗时 56.8s 超了当时 45s 的限，而日志只说"未生效"，
    看不出该调超时、换模型还是改提示词。
    """
    cards = [f"卡片 {index}" for index in range(10)]

    async def slow(**kwargs):
        await asyncio.sleep(10)
        return {"success": True, "response": "{}"}

    merged, note = _compress(cards, slow, timeout_seconds=0.01)
    assert merged is None and "超时" in note

    async def not_json(**kwargs):
        return {"success": True, "response": "这不是 JSON"}

    merged, note = _compress(cards, not_json)
    assert merged is None and "JSON" in note

    async def broken(**kwargs):
        return {"success": False, "error": "模型未配置"}

    merged, note = _compress(cards, broken)
    assert merged is None and "未配置" in note, "调用失败也要带上底层原因，别只说『未生效』"


# ------------------------------------------------------------ 外观卡分组展示

def test_appearance_cards_are_classified_by_kind() -> None:
    """分类必须吃得住真机的写法——下面这些串是 09-18 日志里实际存进库的卡片。"""
    cases = {
        "浅蓝色长发，刘海整齐覆盖额头，发尾微卷": "发色发型",
        "渐变蓝短发带齐刘海，头顶带有闪电造型的呆毛": "发色发型",
        "金色瞳孔，眼神明亮清晰": "眼睛",
        "眼睛为明亮的暖黄色虹膜": "眼睛",
        "深蓝色短袖制服上衣，领口有白色荷叶边和蝴蝶结": "服装",
        "白色百褶短裙，搭配白色中筒袜和黑色玛丽珍鞋": "服装",
        "头顶有黑色圆形耳机状饰品，带有发光蓝环": "配饰",
        "头戴类似黑胶唱片造型的头饰": "配饰",
        "散发着淡淡的香气": "其他",
        # 真机 13:53 的成品卡。第一张因为含「装饰」，曾被裸「装」字误判成服装。
        "头戴带有红边装饰的宽檐黑色大斗笠": "配饰",
        "身穿黑色短抹胸搭配透明外罩衫与灰黄拼色短裙": "服装",
        "白色中筒袜配黑色玛丽珍鞋，鞋面有白色拼接设计": "服装",
        "深蓝色短袖制服配白色立领与蝴蝶结，袖口有金色饰环": "服装",
    }
    for card, expected in cases.items():
        assert textutil.classify_appearance_card(card) == expected, f"{card} → 期望 {expected}"


def test_appearance_grouping_keeps_order_and_drops_empty_kinds() -> None:
    groups = textutil.group_appearance_cards(
        ["金色瞳孔", "浅蓝色长发", "深蓝色制服上衣", "", "   "]
    )
    assert list(groups) == ["眼睛", "发色发型", "服装"], "空卡要丢掉，组内保持原顺序"
    assert groups["眼睛"] == ["金色瞳孔"]


def test_appearance_render_names_the_missing_core_kind() -> None:
    """缺哪一类必须说出来——"三类齐不齐"是核对候选的依据，光看条数看不出来。"""
    plugin = _plugin_module()
    text = plugin._render_appearance_cards(["浅蓝色长发，刘海整齐", "金色瞳孔，眼神明亮"])
    assert "外观卡 2 条" in text
    assert "发色发型 1" in text and "眼睛 1" in text
    assert "缺「服装」" in text

    complete = plugin._render_appearance_cards(
        ["浅蓝色长发", "金色瞳孔", "深蓝色制服上衣", "黑色圆环头饰"]
    )
    assert "缺「" not in complete, "三类齐全时不该有缺口提示"
    assert "配饰 1" in complete, "配饰要显示，但不算核心三类"


# ------------------------------------------------------------ 角色登记的空闲窗口

def test_pending_addition_idle_window_measures_since_the_last_batch() -> None:
    """超时看的是"最后一张图"，不是登记那一刻。

    用绝对期限的话，图越多越容易半路被判超时——恰恰是连续收集最不能出的事。
    """
    plugin = _plugin_module()
    window = plugin.CHARACTER_ADD_IDLE_SECONDS
    pending = plugin.PendingAddition(created_at=0.0, name="鸣澜")
    assert pending.expired(window) is False, "窗口内不该超时"
    assert pending.expired(window + 0.01) is True

    pending.touched()  # 收到新图
    assert pending.expired(window) is False, "收到新图后窗口必须重新计时"


# ---------------------------------------------------------------- 识图诊断

def _plugin_module():
    """加载 plugin.py：诊断渲染是模块级纯函数，不需要起插件实例。"""
    import fakehost

    return fakehost.load_plugin_module(PLUGIN_ROOT, module_name="plugin_diag_test")


def test_diagnosis_names_the_retrieval_threshold_as_the_blocker() -> None:
    """模型确认了、但检索分没过线——这是唯一"该去调阈值"的失败点，必须指名。"""
    plugin = _plugin_module()
    text = plugin._render_diagnosis({
        "cache": "未命中",
        "reverse": "无命中",
        "characters": 1,
        "retrieval": "鸣澜 0.31",
        "retrieval_scores": [("鸣澜", 0.31)],
        "vision": "确认了候选",
        "vision_names": ["鸣澜"],
        "threshold": 0.35,
        "tier": "TE",
        "label": "图片[未识别]",
    })
    assert "本地库兜底通路" in text
    assert "0.31" in text and "0.35" in text
    assert "local_confirm_min_score" in text


def test_diagnosis_carries_the_vision_stage_reason_through() -> None:
    """"VLM 没确认"要把它自己报的原因带出来，否则用户只会去猜。"""
    plugin = _plugin_module()
    vlm_miss = plugin._render_diagnosis({
        "retrieval": "鸣澜 0.60", "retrieval_scores": [("鸣澜", 0.60)],
        "vision": "无可用候选（「鸣澜」被判有冲突特征：发饰颜色不同）", "vision_names": [],
        "threshold": 0.35, "tier": "TE", "label": "图片[未识别]",
    })
    assert "双证缺一半" in vlm_miss
    assert "冲突特征" in vlm_miss

    retrieval_miss = plugin._render_diagnosis({
        "retrieval": "没有候选过线", "retrieval_scores": [],
        "vision": "未调用（检索没有选出候选）", "vision_names": [],
        "threshold": 0.35, "tier": "TE", "label": "图片[未识别]",
    })
    assert "未调用（检索没有选出候选）" in retrieval_miss
    assert "local_confirm_min_score" not in retrieval_miss, "不相关的失败点不许提示调阈值"


def test_diagnosis_shows_vision_raw_output() -> None:
    """模型原始输出要露出来——候选被丢时，"它到底写了什么"是唯一能定案的证据。"""
    plugin = _plugin_module()
    text = plugin._render_diagnosis({
        "vision": "无可用候选",
        "vision_raw": '{"is_anime_character":true,"candidates":[]}',
        "tier": "TE", "label": "图片[未识别]", "threshold": 0.35,
    })
    assert "视觉模型原始输出" in text
    assert "is_anime_character" in text


def test_diagnosis_names_source_disagreement_as_the_blocker() -> None:
    """双证齐了却被反查分歧挡住时，必须说清是分歧，而不是含糊地"未触发"。"""
    plugin = _plugin_module()
    text = plugin._render_diagnosis({
        "retrieval": "鸣澜 0.60", "retrieval_scores": [("鸣澜", 0.60)],
        "vision": "确认了候选", "vision_names": ["鸣澜"],
        "reverse_disputes": ["普拉娜"],
        "local_confirmed": [],
        "threshold": 0.35, "tier": "TC", "label": "图片[未识别]",
    })
    assert "分歧时不贴" in text and "普拉娜" in text


def test_diagnosis_reports_successful_local_fallback() -> None:
    plugin = _plugin_module()
    text = plugin._render_diagnosis({
        "tier": plugin.TIER_LOCAL,
        "label": "图片[鸣澜]",
        "local_confirmed": [["char-1", "鸣澜"]],
        "threshold": 0.35,
    })
    assert "已触发" in text and "图片[鸣澜]" in text


def test_diagnosis_without_history_tells_user_what_to_do() -> None:
    plugin = _plugin_module()
    assert "先在本会话发一张图" in plugin._render_diagnosis(None)
