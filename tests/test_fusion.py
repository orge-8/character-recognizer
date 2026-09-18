# -*- coding: utf-8 -*-
"""多源融合决策的回归测试（纯逻辑，无 ctx、无网络）。"""

import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from fusion import FusionOptions, describe_hits, fuse, saucenao_confidence  # noqa: E402
from models import (  # noqa: E402
    TIER_CONFIRMED,
    TIER_CONFLICT,
    TIER_LOCAL,
    TIER_NONE,
    TIER_SINGLE,
    TIER_WORK_ONLY,
    SourceHit,
)
from textutil import clean_booru_tag, normalize_name  # noqa: E402

# 最小本地库替身。**键必须与真实 CharacterRepository 一样先归一**，
# 否则替身比真件宽松/严格，测出来的行为就不是线上的行为。
LIBRARY = {
    normalize_name(key): value
    for key, value in {
        "初音未来": ("char-miku", "初音未来"),
        "初音ミク": ("char-miku", "初音未来"),
        "hatsune miku": ("char-miku", "初音未来"),
        "阿罗娜": ("char-alona", "阿罗娜"),
        "アロナ": ("char-alona", "阿罗娜"),
        "普拉娜": ("char-plana", "普拉娜"),
    }.items()
}


def resolve(name: str):
    return LIBRARY.get(normalize_name(name))


def test_single_source_is_tier_b_and_not_auto_applied() -> None:
    """单源命中默认不贴标签：错贴一个人名比不贴更伤。"""
    result = fuse(
        [SourceHit("anime_trace", raw_name="阿罗娜", work="蔚蓝档案", confident=True)],
        resolve=resolve,
    )
    assert result.tier == TIER_SINGLE
    assert result.auto_apply is False
    assert result.display_names == ("阿罗娜",)
    assert result.label() == "图片[未识别]"


def test_single_source_auto_applies_only_when_explicitly_enabled() -> None:
    result = fuse(
        [SourceHit("anime_trace", raw_name="阿罗娜", work="蔚蓝档案", confident=True)],
        resolve=resolve,
        options=FusionOptions(auto_apply_single_source=True),
    )
    assert result.tier == TIER_SINGLE
    assert result.auto_apply is True
    assert result.label() == "图片[阿罗娜]"


def test_two_sources_agreeing_across_languages_is_confirmed() -> None:
    """日文名与中文名经本地库别名解析到同一角色 → 才算真的"两源一致"。"""
    result = fuse(
        [
            SourceHit("anime_trace", raw_name="初音ミク", work="VOCALOID", confident=True),
            SourceHit("saucenao", raw_name="初音未来", work="VOCALOID", confident=True, score=92.0),
        ],
        resolve=resolve,
    )
    assert result.tier == TIER_CONFIRMED
    assert result.auto_apply is True
    assert result.label() == "图片[初音未来]"
    assert len(result.candidates) == 1, "跨语言写法必须聚成同一个候选，否则会误判成冲突"


def test_conflicting_sources_are_blocked() -> None:
    result = fuse(
        [
            SourceHit("anime_trace", raw_name="阿罗娜", work="蔚蓝档案", confident=True),
            SourceHit("saucenao", raw_name="普拉娜", work="蔚蓝档案", confident=True, score=95.0),
        ],
        resolve=resolve,
    )
    assert result.tier == TIER_CONFLICT
    assert result.conflict is True
    assert result.auto_apply is False
    assert result.label() == "图片[未识别]"
    assert "阿罗娜" in result.reason and "普拉娜" in result.reason


def test_low_confidence_dissent_is_not_a_conflict() -> None:
    """A 明确 X、B 低置信 Y —— 采信 X，不该因为 B 的噪音把结果丢进冲突层。"""
    result = fuse(
        [
            SourceHit("anime_trace", raw_name="阿罗娜", work="蔚蓝档案", confident=True),
            SourceHit("saucenao", raw_name="普拉娜", work="蔚蓝档案", confident=False, score=70.0),
        ],
        resolve=resolve,
    )
    assert result.conflict is False
    assert result.tier == TIER_SINGLE
    assert result.display_names == ("阿罗娜",)


def test_trace_moe_alone_is_work_only_and_never_a_character_source() -> None:
    result = fuse(
        [SourceHit("trace_moe", raw_name="", work="孤独摇滚", confident=True, score=0.94)],
        resolve=resolve,
    )
    assert result.tier == TIER_WORK_ONLY
    assert result.auto_apply is False
    assert result.works == ("孤独摇滚",)
    assert result.candidates == ()


def test_trace_moe_does_not_inflate_source_count() -> None:
    """关键回归：trace.moe 只给作品，绝不能把单源角色命中"抬"成双源确认。"""
    result = fuse(
        [
            SourceHit("anime_trace", raw_name="阿罗娜", work="蔚蓝档案", confident=True),
            SourceHit("trace_moe", raw_name="", work="蔚蓝档案", confident=True, score=0.9),
        ],
        resolve=resolve,
    )
    assert result.tier == TIER_SINGLE, "作品信号不得让角色命中升级为多源确认"
    assert result.auto_apply is False
    assert result.works == ("蔚蓝档案",)


def test_all_low_confidence_does_not_auto_apply() -> None:
    result = fuse(
        [SourceHit("anime_trace", raw_name="阿罗娜", work="蔚蓝档案", confident=False)],
        resolve=resolve,
    )
    assert result.tier == TIER_WORK_ONLY
    assert result.auto_apply is False
    assert result.candidates[0].confident is False


def test_no_hits_is_tier_none() -> None:
    result = fuse([], resolve=resolve)
    assert result.tier == TIER_NONE
    assert result.label() == "图片[未识别]"
    assert result.works == ()


def test_empty_characters_field_does_not_crash() -> None:
    """SauceNAO 的 characters 常为空 —— 那时它只贡献作品，不该报错或误判。"""
    result = fuse(
        [SourceHit("saucenao", raw_name="", work="Character Title", confident=True, score=88.0)],
        resolve=resolve,
    )
    assert result.auto_apply is False
    assert result.candidates == ()
    assert result.works == ("Character Title",)


def test_degraded_retrieval_blocks_auto_label_even_when_confirmed() -> None:
    """检索降级时候选池本身可能不完整，因此即使多源一致也不许自动贴。"""
    result = fuse(
        [
            SourceHit("anime_trace", raw_name="初音ミク", work="VOCALOID", confident=True),
            SourceHit("saucenao", raw_name="初音未来", work="VOCALOID", confident=True, score=95.0),
        ],
        resolve=resolve,
        degraded=True,
    )
    assert result.auto_apply is False
    assert result.label() == "图片[未识别]"
    assert result.degraded is True


def test_vlm_agreement_upgrades_single_source_to_confirmed() -> None:
    result = fuse(
        [SourceHit("anime_trace", raw_name="アロナ", work="蔚蓝档案", confident=True)],
        resolve=resolve,
        vlm_names=["阿罗娜"],  # 视觉模型独立给出中文名，经库解析到同一角色
    )
    assert result.tier == TIER_CONFIRMED
    assert result.auto_apply is True
    assert result.label() == "图片[阿罗娜]"


def test_vlm_disagreement_does_not_upgrade() -> None:
    result = fuse(
        [SourceHit("anime_trace", raw_name="アロナ", work="蔚蓝档案", confident=True)],
        resolve=resolve,
        vlm_names=["普拉娜"],
    )
    assert result.tier == TIER_SINGLE
    assert result.auto_apply is False


def test_label_carries_relationship_suffix() -> None:
    result = fuse(
        [
            SourceHit("anime_trace", raw_name="阿罗娜", work="蔚蓝档案", confident=True),
            SourceHit("saucenao", raw_name="阿罗娜", work="蔚蓝档案", confident=True, score=90.0),
        ],
        resolve=resolve,
    )
    assert result.label({"char-alona": "同伴"}) == "图片[阿罗娜（同伴）]"


def test_relationship_suffix_never_duplicates_multi_character_labels() -> None:
    """多角色标签不该给每个名字都挂同一个关系后缀（那会读成"和解"）。"""
    result = fuse(
        [
            SourceHit("anime_trace", raw_name="阿罗娜", work="蔚蓝档案", confident=True),
            SourceHit("saucenao", raw_name="阿罗娜", work="蔚蓝档案", confident=True, score=90.0),
        ],
        resolve=resolve,
    )
    label = result.label({"char-alona": "同伴"})
    assert label.count("（同伴）") == 1


# ---------------------------------------------------------------- 名字归一

def test_booru_tag_is_humanized() -> None:
    assert clean_booru_tag("hatsune_miku") == "Hatsune Miku"
    assert clean_booru_tag("Hatsune_Miku") == "Hatsune Miku"
    assert clean_booru_tag("") == ""
    # 带数字的 tag 不做 title 化，避免把 1girl 类噪音变成"名字"
    assert clean_booru_tag("1girl") == "1girl"


def test_booru_tag_bridges_to_animetrace_name() -> None:
    """真实的跨源场景：AnimeTrace 给日文名，SauceNAO 给 booru tag。"""
    result = fuse(
        [
            SourceHit("anime_trace", raw_name="初音ミク", work="VOCALOID", confident=True),
            SourceHit("saucenao", raw_name=clean_booru_tag("hatsune_miku"), work="VOCALOID",
                      confident=True, score=91.0),
        ],
        resolve=resolve,
    )
    assert result.tier == TIER_CONFIRMED
    assert result.label() == "图片[初音未来]"


def test_saucenao_confidence_thresholds() -> None:
    options = FusionOptions(saucenao_confident_similarity=85.0, saucenao_weak_similarity=60.0)
    assert saucenao_confidence(95.0, options) is True
    assert saucenao_confidence(70.0, options) is False
    assert saucenao_confidence(20.0, options) is None


# ---------------------------------------------------- 反查失声时的本地库兜底

def test_local_confirm_applies_when_no_source_reports_anything() -> None:
    """反查源一个名字都没报（真机常见：AnimeTrace 整段 429），本地库+VLM 双证仍然能贴。

    没有这条通路时，限流窗口里"库里有这个人、每张图都报未识别"——用户攒的库
    恰好在最需要它的时刻消失。
    """
    result = fuse((), local_confirmed=[("char-alona", "阿罗娜")])
    assert result.tier == TIER_LOCAL
    assert result.auto_apply is True
    assert result.label() == "图片[阿罗娜]"
    assert "本地库" in result.reason


def test_local_confirm_requires_both_evidence_and_healthy_retrieval() -> None:
    # 没有本地确认候选 → 维持原判
    assert fuse(()).tier == TIER_NONE
    # 检索降级：候选池本身可能不完整，"没搜到别人"不等于"就是他"
    degraded = fuse((), local_confirmed=[("char-alona", "阿罗娜")], degraded=True)
    assert degraded.auto_apply is False
    assert degraded.label() == "图片[未识别]"
    # 显式关掉该通路
    off = FusionOptions(auto_apply_local_confirm=False)
    assert fuse((), local_confirmed=[("char-alona", "阿罗娜")], options=off).tier == TIER_NONE


def test_local_confirm_is_vetoed_when_a_source_points_elsewhere() -> None:
    """反查源（低置信）说的是**另一个**角色 —— 这时本地双证不再自动贴。

    低置信不足以一票否决（源自己都不确定），但"指向别处"是分歧，分歧要停下：
    错贴一个人名比不贴更伤。
    """
    result = fuse(
        [SourceHit("anime_trace", raw_name="普拉娜", work="蔚蓝档案", confident=False)],
        resolve=resolve,
        local_confirmed=[("char-alona", "阿罗娜")],
    )
    assert result.auto_apply is False
    assert result.label() == "图片[未识别]"


def test_local_confirm_survives_a_source_name_outside_the_library() -> None:
    """反查的名字不在库里（多半只是别名没登记）不该否决兜底。

    "解析不到" ≠ "分歧"：跨语言场景里源常给日文名而库里只有中文正名。把它当分歧，
    整个兜底在跨语言图上就永远失效——而这类图恰恰是这个插件的主场。
    """
    result = fuse(
        [SourceHit("anime_trace", raw_name="アロナ", work="蔚蓝档案", confident=False)],
        resolve=lambda _name: None,  # 库里没登记这个名字
        local_confirmed=[("char-alona", "阿罗娜")],
    )
    assert result.tier == TIER_LOCAL
    assert result.auto_apply is True
    assert result.label() == "图片[阿罗娜]"


def test_local_confirm_applies_when_weak_source_agrees() -> None:
    """低置信候选指向**同一个**角色时不该被当成障碍——那是三方同向。

    真机实测：反查源返回 3 条低置信命中（tier 落到 TC）把兜底整条挡住，
    导致"库里有角色、检索选中了、却没贴上"。
    """
    result = fuse(
        [SourceHit("anime_trace", raw_name="アロナ", work="蔚蓝档案", confident=False)],
        resolve=resolve,
        local_confirmed=[("char-alona", "阿罗娜")],
    )
    assert result.tier == TIER_LOCAL
    assert result.auto_apply is True
    assert result.label() == "图片[阿罗娜]"


def test_local_confirm_is_ignored_when_sources_agree() -> None:
    """有正常反查结论时，兜底通路不参与，理由不得被改写。"""
    result = fuse(
        [
            SourceHit("anime_trace", raw_name="初音ミク", work="VOCALOID", confident=True),
            SourceHit("saucenao", raw_name="Hatsune Miku", work="VOCALOID", confident=True),
        ],
        resolve=resolve,
        local_confirmed=[("char-alona", "阿罗娜")],
    )
    assert result.tier == TIER_CONFIRMED
    assert result.display_names == ("初音未来",)


# ---------------------------------------------------------------- 描述渲染

def test_describe_hits_is_plain_text() -> None:
    text = describe_hits([
        SourceHit("anime_trace", raw_name="阿罗娜", work="蔚蓝档案", confident=True),
        SourceHit("trace_moe", raw_name="", work="蔚蓝档案", confident=True, score=0.9),
    ])
    assert "AnimeTrace" in text and "阿罗娜" in text
    assert "trace.moe" in text
    assert "图片[" not in text, "描述函数不该生成标签"
