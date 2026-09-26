# -*- coding: utf-8 -*-
"""检索层的回归测试。

本文件里最重要的一条是 ``test_last_added_character_still_retrievable``：
它定点了参考实现最硬的天花板（``characters[:40]``），并且自己证明了前提成立。
"""

import asyncio
import hashlib
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from models import Character, Query, SourceHit  # noqa: E402
from retrieval import (  # noqa: E402
    EmbeddingIndex,
    RetrievalOptions,
    build_reverse_confidence,
    cosine,
    name_appears_in,
    retrieve,
)
from textutil import normalize_name, tokenize  # noqa: E402

DIM = 24


def _stable_bucket(token: str) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % DIM


class EmbedStub:
    """确定性的"词袋"向量：共享 token 越多的文本，余弦越接近 1。

    不是真模型，但足以验证"检索路径是否按相似度排序"这件事本身。
    """

    def __init__(self, fail: bool = False, dim: int = DIM) -> None:
        self.fail = fail
        self.dim = dim
        self.calls: list[list[str]] = []

    async def __call__(self, texts):
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("embedding 任务未配置")
        vectors = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in tokenize(text):
                vector[_stable_bucket(token)] += 1.0
            vectors.append(vector)
        return vectors

    @property
    def embedded_texts(self) -> list[str]:
        return [text for call in self.calls for text in call]


def run(coro):
    return asyncio.run(coro)


def character(index: int, name: str, appearance: str, work: str = "测试作品") -> Character:
    return Character(
        character_id=f"char-{index:03d}",
        name=name,
        work=work,
        appearance_cards=(appearance,),
        tags=(f"tag{index}",),
    )


# ---------------------------------------------------------------- 头条回归

def test_last_added_character_still_retrievable() -> None:
    """定点回归：参考实现取前 40 个，第 50 个新教的角色永远进不了提示词。

    这条测试自己证明前提成立——先断言"取前 N 会漏掉它"，再断言检索能召回。
    没有前半句，测试就只是"碰巧过了"，不能说明修掉了什么。
    """
    characters = [
        character(i, f"角色{i:02d}", f"编号{i:02d}的通用外观特征 蓝色制服 短发") for i in range(49)
    ]
    naruto = character(49, "漩涡鸣人", "金色刺猬头发 橙色忍者服 脸颊有胡须纹 护额")
    characters.append(naruto)  # 最后才教的角色

    naive_first_n = [item.character_id for item in characters[:12]]
    assert naruto.character_id not in naive_first_n, (
        "前提不成立：参考实现的「取前 N」本该漏掉末尾追加的角色"
    )

    result = run(retrieve(
        characters,
        Query(description="图片里是一个金色刺猬头、穿橙色忍者服、脸颊有胡须纹的少年"),
        options=RetrievalOptions(max_prompt_characters=12, embedding_enabled=False),
    ))

    selected_ids = [item.character.character_id for item in result.selected]
    assert naruto.character_id in selected_ids, "相关性检索必须把新教的角色召回"
    assert selected_ids[0] == naruto.character_id, "它应当是相关性最高的那个"
    assert len(selected_ids) <= 12, "max_prompt_characters 是上限而非必须填满的目标"
    assert len(selected_ids) < 12, (
        "预算内的候选池只应包含真正相关的角色，不该再拿后面的角色来凑数"
    )


def test_large_library_retrieval_is_relevance_ranked_with_embeddings() -> None:
    """同样场景开向量层：新角色必须排在最前，而不是勉强挤进预算。"""
    characters = [
        character(i, f"角色{i:02d}", f"编号{i:02d}的通用外观特征 蓝色制服 短发") for i in range(49)
    ]
    characters.append(character(49, "漩涡鸣人", "金色刺猬头发 橙色忍者服 脸颊有胡须纹 护额"))
    stub = EmbedStub()

    result = run(retrieve(
        characters,
        Query(description="金色刺猬头发 橙色忍者服 脸颊有胡须纹"),
        options=RetrievalOptions(max_prompt_characters=12),
        embed=stub,
        index=EmbeddingIndex(),
    ))

    assert result.degraded is False
    assert result.selected[0].character.character_id == "char-049"
    # 名字没被描述提到，所以走的是向量/关键词而不是 T1
    assert "名字精确命中" not in result.selected[0].reasons


def test_name_mentioned_in_description_is_exact_hit() -> None:
    characters = [character(0, "阿罗娜", "蓝白长发 发光圆环"), character(1, "普拉娜", "白发 深色圆环")]
    result = run(retrieve(
        characters,
        Query(description="画面中是阿罗娜，站在教室里"),
        options=RetrievalOptions(embedding_enabled=False),
    ))
    assert result.selected[0].character.name == "阿罗娜"
    assert "名字精确命中" in result.selected[0].reasons


# ---------------------------------------------------------------- pinned 保证

def test_reverse_confirmed_character_is_pinned_even_when_scoring_low() -> None:
    """反查源点名的角色无条件入选：它可能长得完全不像画像里的描述。"""
    characters = [
        character(i, f"角色{i:02d}", f"编号{i:02d}的通用外观特征 蓝色制服 短发") for i in range(20)
    ]
    characters.append(Character(character_id="char-target", name="漩涡鸣人", work="火影忍者"))
    stub = EmbedStub()

    result = run(retrieve(
        characters,
        Query(description="一张和画像毫无共同点的图", reverse_names=["漩涡鸣人"]),
        options=RetrievalOptions(max_prompt_characters=5, pinned_reserve=5),
        embed=stub,
        index=EmbeddingIndex(),
        reverse_confidence={"漩涡鸣人": 0.95},
    ))

    assert result.pinned, "反查点名的角色必须进 pinned"
    assert result.selected[0].character.character_id == "char-target"
    assert result.selected[0].pinned is True
    assert len([item for item in result.selected if item.pinned]) == 1


def test_pinned_chars_survive_a_full_budget() -> None:
    """预算被高分角色占满时，pinned 也不能被挤掉。"""
    characters = [character(i, f"角色{i:02d}", "金色刺猬头 橙色忍者服") for i in range(30)]
    characters.append(Character(character_id="char-target", name="漩涡鸣人"))
    stub = EmbedStub()

    result = run(retrieve(
        characters,
        Query(description="金色刺猬头 橙色忍者服", reverse_names=["漩涡鸣人"]),
        options=RetrievalOptions(max_prompt_characters=5, pinned_reserve=5),
        embed=stub,
        index=EmbeddingIndex(),
        reverse_confidence={"漩涡鸣人": 0.9},
    ))

    assert "char-target" in [item.character.character_id for item in result.selected]
    assert len(result.selected) == 5


def test_pinned_beyond_budget_is_truncated_by_confidence() -> None:
    characters = [Character(character_id=f"char-{i}", name=f"角色{i}") for i in range(6)]
    stub = EmbedStub()
    confidence = {f"角色{i}": 0.5 + i * 0.05 for i in range(6)}

    result = run(retrieve(
        characters,
        Query(description="随便什么图", reverse_names=list(confidence)),
        options=RetrievalOptions(max_prompt_characters=3, pinned_reserve=2),
        embed=stub,
        index=EmbeddingIndex(),
        reverse_confidence=confidence,
    ))

    assert len(result.selected) == 3
    # 可信度最高的两个应当入选
    assert [item.character.name for item in result.selected[:2]] == ["角色5", "角色4"]


# ---------------------------------------------------------------- 降级路径

def test_embedding_failure_degrades_to_keywords_without_raising() -> None:
    characters = [
        character(0, "漩涡鸣人", "金色刺猬头 橙色忍者服"),
        character(1, "宇智波佐助", "黑色刺猬头 蓝黑上衣 写轮眼"),
    ]
    stub = EmbedStub(fail=True)

    result = run(retrieve(
        characters,
        Query(description="金色刺猬头 橙色忍者服"),
        options=RetrievalOptions(),
        embed=stub,
        index=EmbeddingIndex(),
    ))

    assert result.degraded is True
    assert result.selected, "降级后仍必须给出有序候选，不能空手而归"
    assert result.selected[0].character.name == "漩涡鸣人"


def test_embedding_absent_marks_degraded() -> None:
    characters = [character(0, "漩涡鸣人", "金色刺猬头")]
    result = run(retrieve(
        characters,
        Query(description="金色刺猬头"),
        options=RetrievalOptions(embedding_enabled=False),
    ))
    assert result.degraded is True
    assert result.selected[0].character.name == "漩涡鸣人"


def test_dimension_mismatch_discards_the_whole_batch() -> None:
    """Host 中途换 embedding 模型 → 维度不一致。整批丢弃，绝不让索引被污染。"""
    characters = [character(0, "漩涡鸣人", "金色刺猬头"), character(1, "佐助", "黑色刺猬头")]

    class Ragged:
        async def __call__(self, texts):
            return [[1.0] * 4 for _ in texts[:-1]] + [[1.0] * 8]

    index = EmbeddingIndex()
    result = run(retrieve(
        characters,
        Query(description="金色刺猬头"),
        options=RetrievalOptions(),
        embed=Ragged(),
        index=index,
    ))
    assert result.degraded is True
    assert len(index) == 0, "维度不一致时必须一个向量都不留"


def test_empty_library_returns_empty_result() -> None:
    result = run(retrieve([], Query(description="任意"), options=RetrievalOptions()))
    assert result.selected == ()
    assert result.total == 0
    assert result.degraded is False


# ---------------------------------------------------------------- 向量缓存

def test_only_changed_character_is_re_embedded() -> None:
    """画像没变的角色不该被反复重算向量，否则每次识图都要重灌整个库。"""
    characters = [character(0, "漩涡鸣人", "金色刺猬头"), character(1, "佐助", "黑色刺猬头")]
    stub = EmbedStub()
    index = EmbeddingIndex()
    query = Query(description="金色刺猬头")
    options = RetrievalOptions()

    run(retrieve(characters, query, options=options, embed=stub, index=index))
    calls_after_first = len(stub.calls)
    assert calls_after_first >= 2, "首次应至少有一次补算 + 一次查询向量"

    run(retrieve(characters, query, options=options, embed=stub, index=index))
    assert len(stub.calls) == calls_after_first + 1, "第二次只剩查询向量那一次调用"

    updated = [
        characters[0],
        Character(
            character_id=characters[1].character_id,
            name=characters[1].name,
            work=characters[1].work,
            appearance_cards=("全新的外观描述",),
            tags=characters[1].tags,
        ),
    ]
    run(retrieve(updated, query, options=options, embed=stub, index=index))
    assert stub.calls[-2] == [updated[1].profile_text], "只应重算画像变了的那个角色"


def test_prune_drops_removed_characters() -> None:
    characters = [character(0, "漩涡鸣人", "金色刺猬头"), character(1, "佐助", "黑色刺猬头")]
    stub = EmbedStub()
    index = EmbeddingIndex()
    run(retrieve(characters, Query(description="金色刺猬头"), embed=stub, index=index))
    assert len(index) == 2

    run(retrieve(characters[:1], Query(description="金色刺猬头"), embed=stub, index=index))
    assert len(index) == 1


# ---------------------------------------------------------------- 纯函数

def test_cosine_bounds_and_edge_cases() -> None:
    assert cosine([], []) == 0.0
    assert cosine([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == 0.0, "负相关要截 0，不能变成负分"


def test_name_appears_in_respects_length_thresholds() -> None:
    assert name_appears_in("画面中是阿罗娜站在教室", ["阿罗娜"])
    assert name_appears_in("hatsune miku singing", ["Hatsune Miku"])
    # 拉丁名两字符太容易误命中，必须被门槛挡住
    assert not name_appears_in("a cv list", ["CV"])
    assert not name_appears_in("随便一句话", [""])


def test_build_reverse_confidence_uses_source_count_not_score_sum() -> None:
    hits = [
        SourceHit("anime_trace", raw_name="阿罗娜", confident=True),
        SourceHit("saucenao", raw_name="阿罗娜", confident=True, score=99.0),
    ]
    confident = build_reverse_confidence(hits)
    single = build_reverse_confidence([hits[0]])

    key = normalize_name("阿罗娜")
    assert confident[key] > single[key], "两源一致的可信度必须高于单源"
    assert confident[key] <= 1.0
    # 分数尺度不可比，所以加第二个源不会因为 score=99 就爆表
    assert confident[key] <= 1.0


def test_build_reverse_confidence_ignores_work_only_sources() -> None:
    hits = [SourceHit("trace_moe", raw_name="", work="火影忍者", confident=True, score=0.95)]
    assert build_reverse_confidence(hits) == {}


def test_cosine_accepts_precomputed_norms() -> None:
    """入库时算好的范数必须与现算结果一致。"""
    assert cosine([1.0, 0.0], [1.0, 0.0], norm_left=1.0, norm_right=1.0) == 1.0
    assert cosine([3.0, 4.0], [3.0, 4.0]) == cosine(
        [3.0, 4.0], [3.0, 4.0], norm_left=5.0, norm_right=5.0
    )


def test_character_derived_fields_are_cached_per_instance() -> None:
    """profile_text / profile_tokens / normalized_profile 是 cached_property：
    检索每轮为每个角色各算一次就够，重复访问必须复用同一对象。"""
    item = character(0, "阿罗娜", "蓝白长发")
    first = item.profile_text
    assert item.profile_text is first, "cached_property 必须复用同一对象"
    assert item.profile_tokens == tokenize(first)
    assert item.normalized_profile == normalize_name(first)
