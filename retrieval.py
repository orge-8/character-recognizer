# -*- coding: utf-8 -*-
"""本地角色库的相关性检索（不依赖 ctx，可脱机单测）。

**这是相对参考实现最关键的改动。** 参考实现的 ``private_catalog(limit=40)`` 做的是
``self._characters[:limit]``——取前 N 个而不是最相关的 N 个；而 ``upsert()`` 又把新条目
追加到列表末尾。两者叠加的后果是：角色库一旦超过 40 个，**最近教的角色永远进不了
视觉提示词**，越是用心补的越识别不出来。

这里改成三层打分 + pinned 保证：

* **T1** 精确命中（名字/别名出现在反查结果或描述里）→ 直接满分
* **T2** 向量余弦（画像文本 vs 查询文本）
* **T3** 中文 bigram 重叠（不依赖任何模型，永远可用）
* **pinned** 被反查源明确点名的角色**无条件入选**，不会被更高分的角色挤掉
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Awaitable, Callable, Mapping, Sequence

try:
    from .models import Character, Query, RetrievalResult, ScoredCharacter
    from .textutil import normalize_name, overlap_score, tokenize
except ImportError:  # pragma: no cover - 取决于加载方式
    from models import Character, Query, RetrievalResult, ScoredCharacter
    from textutil import normalize_name, overlap_score, tokenize

#: 批量取向量：入参是文本列表，返回等长的向量列表；失败时返回 None
EmbedFn = Callable[[Sequence[str]], Awaitable["list[list[float]] | None"]]


@dataclass(frozen=True)
class RetrievalOptions:
    """检索参数。权重按"能区分到什么程度"排，不是等权。"""

    #: 进提示词的候选**上限**。刻意不设下限：预算里只放真正相关的角色，宁可只送两三
    #: 个准的，也不要拿不相关的角色来凑数——塞噪音会让视觉模型更容易认错人。
    max_prompt_characters: int = 12
    pinned_reserve: int = 5
    embedding_enabled: bool = True
    embed_batch_size: int = 32
    weight_embedding: float = 0.55
    weight_keyword: float = 0.35
    boost_reverse_confirmed: float = 0.30
    boost_work_match: float = 0.10
    #: 关键词命中的最低分，低于它视为噪音。
    keyword_min_score: float = 0.12
    #: 向量余弦的最低分，同样用于过滤噪音。
    embed_min_score: float = 0.35


def content_hash(character: Character) -> str:
    """画像文本摘要。只有它变了才需要重算该角色的向量。"""
    return hashlib.sha256(character.profile_text.encode("utf-8")).hexdigest()


class EmbeddingIndex:
    """角色向量的进程内缓存，按 content_hash 失效。"""

    def __init__(self) -> None:
        #: character_id -> (content_hash, 向量, 向量范数)。范数入库时算一次，
        #: 不再在每次识别的每次余弦里重算（O(维度) 的纯 Python 循环）。
        self._vectors: dict[str, tuple[str, tuple[float, ...], float]] = {}

    def get(self, character_id: str) -> tuple[float, ...] | None:
        entry = self._vectors.get(character_id)
        return entry[1] if entry else None

    def get_with_norm(self, character_id: str) -> "tuple[tuple[float, ...], float] | None":
        entry = self._vectors.get(character_id)
        return (entry[1], entry[2]) if entry else None

    def stale(self, characters: Sequence[Character]) -> list[Character]:
        """找出画像已变化或从未算过向量的角色。"""
        pending: list[Character] = []
        for character in characters:
            entry = self._vectors.get(character.character_id)
            if entry is None or entry[0] != content_hash(character):
                pending.append(character)
        return pending

    def put(self, character: Character, vector: Sequence[float]) -> None:
        values = tuple(float(value) for value in vector)
        norm = sum(value * value for value in values) ** 0.5
        self._vectors[character.character_id] = (content_hash(character), values, norm)

    def prune(self, valid_ids: set[str]) -> None:
        for character_id in [key for key in self._vectors if key not in valid_ids]:
            del self._vectors[character_id]

    def dimensions(self) -> set[int]:
        return {len(entry[1]) for entry in self._vectors.values()}

    def clear(self) -> None:
        self._vectors.clear()

    def __len__(self) -> int:
        return len(self._vectors)


def cosine(
    left: Sequence[float],
    right: Sequence[float],
    *,
    norm_left: float | None = None,
    norm_right: float | None = None,
) -> float:
    """余弦相似度，负值截 0（负相关对"是不是这个角色"没有意义）。

    ``norm_left`` / ``norm_right`` 可传预计算的范数：角色向量的范数在入库时已经
    算好（见 ``EmbeddingIndex.put``），逐角色打分时不该再各算一遍。
    """
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    if norm_left is None:
        norm_left = sum(a * a for a in left) ** 0.5
    if norm_right is None:
        norm_right = sum(b * b for b in right) ** 0.5
    if not norm_left or not norm_right:
        return 0.0
    return max(0.0, dot / (norm_left * norm_right))


def name_appears_in(haystack: str, names: Sequence[str]) -> bool:
    """``haystack`` 里是否出现了 ``names`` 中的某个名字（两侧同样归一）。

    长度门槛按字种分：CJK 名字两个字符就有区分度，拉丁名两个字符几乎全是误命中
    （"AI"、"CV"、"OP"），所以要求 4 个字符。
    """
    normalized = normalize_name(haystack)
    if not normalized:
        return False
    return _name_in_normalized(normalized, [normalize_name(name) for name in names])


def _name_in_normalized(normalized_haystack: str, normalized_names: Sequence[str]) -> bool:
    """归一后的包含判定。热路径专用：两侧都已归一，不再逐角色逐 haystack 做 NFKC。"""
    for candidate in normalized_names:
        if not candidate:
            continue
        if candidate == normalized_haystack:
            return True
        threshold = 4 if candidate.isascii() else 2
        if len(candidate) >= threshold and candidate in normalized_haystack:
            return True
    return False


async def retrieve(
    characters: Sequence[Character],
    query: Query,
    *,
    options: RetrievalOptions | None = None,
    embed: EmbedFn | None = None,
    index: EmbeddingIndex | None = None,
    reverse_confidence: Mapping[str, float] | None = None,
) -> RetrievalResult:
    """按相关性挑选要进提示词的角色。

    Args:
        characters: 全量角色库。
        query: 查询（VLM 描述 + 反查名字/作品 + 可选聊天文本）。
        options: 权重与预算。
        embed: 批量取向量的回调。为 None 或失败时自动降级为关键词检索。
        index: 向量缓存，跨次调用复用。
        reverse_confidence: **归一后的名字 → 反查可信度**，命中它的角色会被 pinned。
    """
    options = options or RetrievalOptions()
    reverse_confidence = reverse_confidence or {}
    total = len(characters)
    if not total:
        return RetrievalResult(degraded=False, total=0)

    query_text = query.compose()
    query_tokens = tokenize(query_text)
    normalized_reverse = {normalize_name(name): float(weight) for name, weight in reverse_confidence.items()}
    # haystack 的 NFKC 归一每次检索只做一次，不再逐角色重复（原为 角色数 × haystack 数 次）
    normalized_haystacks = [
        value
        for value in (
            normalize_name(text)
            for text in (*query.reverse_names, *query.evidence, query.description)
        )
        if value
    ]

    # ---- pinned：被反查源点名的角色无条件入选
    weighted_pinned: list[tuple[float, ScoredCharacter]] = []
    for character in characters:
        matched = next(
            (
                key
                for key in (normalize_name(name) for name in character.all_names)
                if key and key in normalized_reverse
            ),
            "",
        )
        if not matched:
            continue
        weight = normalized_reverse[matched]
        weighted_pinned.append((
            weight,
            ScoredCharacter(
                character=character,
                score=1.0,
                pinned=True,
                reasons=(f"反查源点名「{matched}」（可信度 {weight:.2f}）",),
            ),
        ))
    weighted_pinned.sort(key=lambda pair: (-pair[0], pair[1].character.name))
    pinned = [item for _, item in weighted_pinned]

    # ---- 向量层（不可用是常态，必须一等公民对待）
    query_vector: tuple[float, ...] = ()
    if options.embedding_enabled and embed is not None:
        query_vector, degraded = await _prepare_vectors(
            characters, query_text, embed=embed, index=index, batch_size=options.embed_batch_size
        )
    else:
        degraded = True

    pinned_ids = {item.character.character_id for item in pinned}
    active_index = index if query_vector else None
    query_norm = sum(value * value for value in query_vector) ** 0.5 if query_vector else 0.0

    scored: list[ScoredCharacter] = []
    for character in characters:
        if character.character_id in pinned_ids:
            continue
        score, reasons = _score(
            character,
            query=query,
            query_tokens=query_tokens,
            normalized_reverse=normalized_reverse,
            normalized_haystacks=normalized_haystacks,
            query_vector=query_vector,
            query_norm=query_norm,
            index=active_index,
            options=options,
            degraded=degraded,
        )
        if score > 0:
            scored.append(ScoredCharacter(character=character, score=score, reasons=reasons))
    scored.sort(key=lambda item: (-item.score, item.character.name))

    budget = max(1, options.max_prompt_characters)
    reserve = max(0, min(options.pinned_reserve, budget))
    # pinned 最多占用 reserve 个名额，避免反查命中的角色把预算吃光、把其他可能相关的
    # 角色全挤出去；剩余名额由"未被保证的 pinned + 打分候选"按分数竞争。
    # pinned 自身已按可信度排好序，同分时靠稳定排序保序。
    guaranteed = pinned[:reserve]
    pool = [*pinned[len(guaranteed):], *scored]
    pool.sort(key=lambda item: -item.score)
    selected = [*guaranteed, *pool[:max(0, budget - len(guaranteed))]]

    return RetrievalResult(
        selected=tuple(selected),
        pinned=tuple(pinned),
        degraded=degraded,
        total=total,
    )


async def _prepare_vectors(
    characters: Sequence[Character],
    query_text: str,
    *,
    embed: EmbedFn,
    index: EmbeddingIndex | None,
    batch_size: int,
) -> tuple[tuple[float, ...], bool]:
    """取查询向量并补算过期角色向量，返回 ``(query_vector, degraded)``。

    任何一处对不齐都判定"向量层不可用"并**整体降级**，而不是只让部分角色少一项得分——
    否则有向量的角色会系统性压过没向量的角色，而后者往往正是刚加进库的那批，
    那等于把参考实现的"新角色进不来"换了个形式重现。
    """
    index = index if index is not None else EmbeddingIndex()
    stale = index.stale(characters)
    size = max(1, batch_size)

    for offset in range(0, len(stale), size):
        chunk = stale[offset:offset + size]
        vectors = await _safe_embed(embed, [item.profile_text for item in chunk])
        if vectors is None or len(vectors) != len(chunk):
            return (), True
        if len({len(vector) for vector in vectors}) != 1:
            # Host 中途换了 embedding 模型 → 维度不一致。整批丢弃，不污染索引。
            return (), True
        for character, vector in zip(chunk, vectors):
            index.put(character, vector)

    index.prune({character.character_id for character in characters})

    query_vectors = await _safe_embed(embed, [query_text])
    if not query_vectors or len(query_vectors) != 1:
        return (), True
    query_vector = tuple(float(value) for value in query_vectors[0])
    if not query_vector:
        return (), True
    dimensions = index.dimensions()
    if len(dimensions) > 1 or (dimensions and len(query_vector) not in dimensions):
        return (), True
    return query_vector, False


async def _safe_embed(embed: EmbedFn, texts: Sequence[str]) -> list[list[float]] | None:
    if not texts:
        return []
    try:
        result = await embed(texts)
    except Exception:
        return None
    if result is None:
        return None
    try:
        return [list(vector) for vector in result]
    except TypeError:
        return None


def _score(
    character: Character,
    *,
    query: Query,
    query_tokens: frozenset[str],
    normalized_reverse: Mapping[str, float],
    normalized_haystacks: Sequence[str],
    query_vector: tuple[float, ...],
    query_norm: float,
    index: EmbeddingIndex | None,
    options: RetrievalOptions,
    degraded: bool,
) -> tuple[float, tuple[str, ...]]:
    """单角色打分，返回 ``(分数, 命中理由)``。"""
    # 派生数据走 Character 的 cached_property：画像拼接 / NFKC 归一 / 分词
    # 对同一实例只做一次，不再逐角色逐次识别重复。
    normalized_profile = character.normalized_profile

    # T1 精确命中：描述/证据里直接写了这个名字（含别名）
    normalized_names = [normalize_name(name) for name in character.all_names]
    if any(_name_in_normalized(haystack, normalized_names) for haystack in normalized_haystacks):
        return 1.0, ("名字精确命中",)

    # 反查源用的名字能解析到该角色（跨语言写法的情形）
    reverse_hit = 0.0
    for name, weight in normalized_reverse.items():
        if name and name in normalized_profile:
            reverse_hit = max(reverse_hit, weight)
    # 作品名一致也算弱信号（作品跨源匹配比角色名可靠）
    work_hit = bool(character.all_works) and any(
        normalize_name(work) == normalize_name(character.work) for work in query.reverse_works if work
    )

    keyword_score = overlap_score(query_tokens, character.profile_tokens)
    reasons: list[str] = []
    if keyword_score >= options.keyword_min_score:
        reasons.append(f"关键词 {keyword_score:.2f}")

    if degraded or not query_vector or index is None:
        # 降级路径：只剩关键词 + 反查一致性，权重重新分配。
        base = 0.70 * keyword_score + 0.30 * reverse_hit
        if work_hit:
            base += 0.05
        if reverse_hit:
            reasons.append("反查名字可解析到该角色")
        return (min(1.0, base), tuple(reasons)) if base > 0 else (0.0, ())

    entry = index.get_with_norm(character.character_id)
    embed_score = (
        cosine(query_vector, entry[0], norm_left=query_norm, norm_right=entry[1])
        if entry
        else 0.0
    )
    if embed_score >= options.embed_min_score:
        reasons.append(f"向量 {embed_score:.2f}")

    if keyword_score < options.keyword_min_score and embed_score < options.embed_min_score and not reverse_hit:
        return 0.0, ()

    base = options.weight_embedding * embed_score + options.weight_keyword * keyword_score
    if reverse_hit:
        base += options.boost_reverse_confirmed * reverse_hit
        reasons.append("反查名字可解析到该角色")
    if work_hit and embed_score >= 0.5:
        base += options.boost_work_match
        reasons.append("作品一致")
    return min(1.0, base), tuple(reasons)


def build_reverse_confidence(hits: Sequence[object]) -> dict[str, float]:
    """把反查命中转成 ``归一名字 → 可信度``。

    可信度只用"多少个源同意"与"源内是否达标"两个变量，**不跨源求分数和**：
    不同源的分数尺度互不可比（见 fusion.py 的说明）。
    """
    pending: dict[str, dict[str, object]] = {}
    for hit in hits:
        name = str(getattr(hit, "raw_name", "") or "").strip()
        if not name or not bool(getattr(hit, "gives_character", False)):
            continue
        key = normalize_name(name)
        if not key:
            continue
        bucket = pending.setdefault(key, {"sources": set(), "confident": False})
        bucket["sources"].add(str(getattr(hit, "source", "")))  # type: ignore[union-attr]
        if bool(getattr(hit, "confident", False)):
            bucket["confident"] = True

    result: dict[str, float] = {}
    for key, bucket in pending.items():
        weight = 0.5 + 0.25 * len(bucket["sources"])  # type: ignore[arg-type]
        if bucket["confident"]:
            weight += 0.2
        result[key] = min(1.0, weight)
    return result
