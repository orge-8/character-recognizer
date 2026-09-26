# -*- coding: utf-8 -*-
"""数据结构定义（纯 dataclass，不依赖 ctx）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

try:  # Runner 可能按包加载，也可能只把目录塞进 sys.path
    from .textutil import normalize_name, tokenize
except ImportError:  # pragma: no cover - 取决于加载方式
    from textutil import normalize_name, tokenize

# 只有这两个源会给出**角色名**，"角色源"集合决定了 agreement 怎么算。
CHARACTER_SOURCES = frozenset({"anime_trace", "saucenao"})
#: 只给作品、不给角色的源。它永远不能计入角色 agreement，
#: 否则一个纯作品信号会和角色信号"撞车"，凭空制造出冲突。
WORK_ONLY_SOURCES = frozenset({"trace_moe"})

SOURCE_LABELS = {
    "anime_trace": "AnimeTrace",
    "saucenao": "SauceNAO",
    "trace_moe": "trace.moe",
    # 不是外部反查源，而是"本地角色库外观卡检索 + 视觉模型确认"这一对证据的署名，
    # 用来说明标签依据来自哪里（反查源全挂时它是唯一的依据）。
    "local_library": "本地库",
}

# 置信分层（见 fusion.py 的决策表）
TIER_CONFIRMED = "TA"      # 多源确认或单源+VLM 一致 → 可自动贴
TIER_SINGLE = "TB"         # 单源明确 → 默认只注入不贴
TIER_WORK_ONLY = "TC"      # 只有作品 → 只注入作品上下文
TIER_CONFLICT = "TD"       # 多源互相矛盾 → 不贴，标注冲突
TIER_NONE = "TE"           # 无信号
#: 反查源集体失声时的兜底：本地库外观卡检索 + 视觉模型在候选里确认。两条证据都是
#: 用户自己维护的资产、不依赖外部服务可用性，所以在限流窗口里也能贴标签。
TIER_LOCAL = "TL"          # 本地库+VLM 双证 → 可自动贴

UNRECOGNIZED_LABEL = "图片[未识别]"


def _as_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if value is None:
        return ()
    text = str(value).strip()
    return (text,) if text else ()


@dataclass(frozen=True)
class Character:
    """本地角色库中的一个角色。"""

    character_id: str
    name: str
    aliases: tuple[str, ...] = ()
    work: str = ""
    work_aliases: tuple[str, ...] = ()
    persona: str = ""
    tags: tuple[str, ...] = ()
    relationship: str = ""
    appearance_cards: tuple[str, ...] = ()
    source: str = "admin"

    @property
    def all_names(self) -> tuple[str, ...]:
        """参与"精确命中"判定的所有称呼：正名 + 别名。"""
        return (self.name, *self.aliases)

    @property
    def all_works(self) -> tuple[str, ...]:
        return tuple(item for item in (self.work, *self.work_aliases) if item)

    @cached_property
    def profile_text(self) -> str:
        """用于向量与关键词检索的完整画像文本。

        顺序刻意把最能区分身份的内容放前面：截断时丢的是外观卡而不是名字。

        用 ``cached_property`` 而不是方法：检索在每次识别里要为**每个角色**拼这段
        文本、做 NFKC 归一和分词，而 Character 是 frozen 的——派生结果只随实例变。
        ``cached_property`` 直接写实例 ``__dict__``，不经 ``__setattr__``，不破坏
        frozen 语义；库重载会重建实例，缓存自然随之失效。
        """
        parts = [self.name]
        if self.aliases:
            parts.append("别名：" + "、".join(self.aliases))
        if self.work:
            parts.append("作品：" + self.work)
        if self.work_aliases:
            parts.append("作品别名：" + "、".join(self.work_aliases))
        if self.tags:
            parts.append("特征：" + "、".join(self.tags))
        if self.persona:
            parts.append("人设：" + self.persona)
        if self.appearance_cards:
            parts.append("外观：" + "；".join(self.appearance_cards))
        return "\n".join(parts)

    @cached_property
    def profile_tokens(self) -> frozenset[str]:
        """``profile_text`` 的分词结果。检索打分的逐角色重复计算大头。"""
        return tokenize(self.profile_text)

    @cached_property
    def normalized_profile(self) -> str:
        """``profile_text`` 的归一形态。反查名解析与精确命中的逐角色重复计算。"""
        return normalize_name(self.profile_text)

    @classmethod
    def from_dict(cls, raw: Any) -> "Character | None":
        if not isinstance(raw, dict):
            return None
        name = str(raw.get("name") or "").strip()
        if not name:
            return None
        return cls(
            character_id=str(raw.get("id") or "").strip(),
            name=name,
            aliases=_as_tuple(raw.get("aliases")),
            work=str(raw.get("work") or "").strip(),
            work_aliases=_as_tuple(raw.get("work_aliases")),
            persona=str(raw.get("persona") or "").strip(),
            tags=_as_tuple(raw.get("tags")),
            relationship=str(raw.get("relationship") or "").strip(),
            appearance_cards=_as_tuple(raw.get("appearance_cards")),
            source=str(raw.get("source") or "admin").strip() or "admin",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.character_id,
            "name": self.name,
            "aliases": list(self.aliases),
            "work": self.work,
            "work_aliases": list(self.work_aliases),
            "persona": self.persona,
            "tags": list(self.tags),
            "relationship": self.relationship,
            "appearance_cards": list(self.appearance_cards),
            "source": self.source,
        }


@dataclass(frozen=True)
class SourceHit:
    """单个反查源的一条命中。各源分数尺度不同，不可跨源相加。"""

    source: str
    raw_name: str = ""
    work: str = ""
    confident: bool = True
    score: float = 0.0
    detail: str = ""

    @property
    def gives_character(self) -> bool:
        """该命中是否携带角色名信号（trace.moe 恒为 False）。"""
        return self.source in CHARACTER_SOURCES and bool(str(self.raw_name).strip())


@dataclass(frozen=True)
class ClusteredCandidate:
    """把多个源对同一角色的主张聚在一起的结果。"""

    key: str
    display_name: str
    character_id: str = ""
    sources: tuple[str, ...] = ()
    works: tuple[str, ...] = ()
    confident: bool = False

    @property
    def source_count(self) -> int:
        return len(set(self.sources))

    @property
    def source_label(self) -> str:
        return "、".join(SOURCE_LABELS.get(item, item) for item in sorted(set(self.sources)))

    @property
    def gives_label(self) -> bool:
        """只有真的带角色名、且达到可信阈值的候选才能进标签。"""
        return bool(self.display_name) and self.confident


@dataclass(frozen=True)
class FusionResult:
    """融合决策结果。``auto_apply`` 才表示允许把标签写进消息。"""

    tier: str = TIER_NONE
    display_names: tuple[str, ...] = ()
    candidates: tuple[ClusteredCandidate, ...] = ()
    works: tuple[str, ...] = ()
    auto_apply: bool = False
    conflict: bool = False
    degraded: bool = False
    reason: str = ""

    def label(self, relationships: dict[str, str] | None = None, limit: int = 3) -> str:
        """生成 ``图片[角色]（关系）`` 标签。``relationships`` 用 character_id 或归一名索引。"""
        if not self.auto_apply or not self.display_names:
            return UNRECOGNIZED_LABEL
        table = relationships or {}
        rendered: list[str] = []
        seen: set[str] = set()
        for candidate in self.candidates:
            if not candidate.gives_label:
                continue
            plain = candidate.display_name
            if plain.casefold() in seen:
                continue
            seen.add(plain.casefold())
            suffix = table.get(candidate.character_id) or table.get(candidate.key) or ""
            rendered.append(f"{plain}（{suffix}）" if suffix else plain)
            if len(rendered) >= limit:
                break
        if not rendered:
            return UNRECOGNIZED_LABEL
        return f"图片[{'、'.join(rendered)}]"


@dataclass(frozen=True)
class VisionCandidate:
    """视觉模型给出的一个候选角色。"""

    kind: str = "unknown"  # private（本地库内） / unknown
    name: str = ""
    work: str = ""
    evidence: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: Any) -> "VisionCandidate | None":
        if not isinstance(raw, dict):
            return None
        kind = str(raw.get("kind") or "unknown").strip().lower()
        if kind not in {"private", "unknown"}:
            kind = "unknown"
        evidence = raw.get("evidence")
        conflicts = raw.get("conflicts")
        return cls(
            kind=kind,
            name=str(raw.get("name") or "").strip(),
            work=str(raw.get("work") or "").strip(),
            evidence=_as_tuple(evidence),
            conflicts=_as_tuple(conflicts),
        )

    @property
    def usable_name(self) -> str:
        """只有本地库内的候选、且没有冲突特征、且证据足够时才算可用。"""
        if self.kind != "private" or self.conflicts or len(self.evidence) < 2:
            return ""
        return self.name


@dataclass(frozen=True)
class VisionResult:
    """一次视觉请求的结果。"""

    description: str = ""
    is_anime_character: bool = False
    candidates: tuple[VisionCandidate, ...] = ()
    #: 模型原始输出（截断后）。"候选为空"有好几种完全不同的原因，没有原始输出就只能猜；
    #: 而猜错的代价是去改根本没问题的那一环。
    raw: str = ""

    @property
    def candidate_names(self) -> tuple[str, ...]:
        return tuple(item.usable_name for item in self.candidates if item.usable_name)

    @classmethod
    def from_dict(cls, raw: Any) -> "VisionResult | None":
        if not isinstance(raw, dict):
            return None
        description = str(raw.get("description") or "").strip()
        is_anime = raw.get("is_anime_character")
        if not isinstance(is_anime, bool):
            is_anime = False
        raw_candidates = raw.get("candidates")
        candidates = (
            tuple(
                candidate
                for item in raw_candidates
                if (candidate := VisionCandidate.from_dict(item)) is not None
            )
            if isinstance(raw_candidates, list)
            else ()
        )
        # 兼容只给单个候选对象的旧形态
        if not candidates:
            legacy = VisionCandidate.from_dict(raw)
            if legacy is not None and (legacy.name or legacy.evidence):
                candidates = (legacy,)
        description = description[:160]
        if not description and not candidates:
            # 既没描述也没候选 = 零信息。返回 None 让调用方能区分"解析成功但没用"与
            # "解析出了内容"，否则一个空壳对象会被当成有效结果一路传下去。
            return None
        return cls(description=description, is_anime_character=is_anime, candidates=candidates[:10])


@dataclass(frozen=True)
class ScoredCharacter:
    """检索打分后的候选角色。"""

    character: Character
    score: float = 0.0
    pinned: bool = False
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalResult:
    """一次检索的完整结果，``degraded`` 表示向量层不可用。"""

    selected: tuple[ScoredCharacter, ...] = ()
    pinned: tuple[ScoredCharacter, ...] = ()
    degraded: bool = False
    total: int = 0

    @property
    def characters(self) -> tuple[Character, ...]:
        return tuple(item.character for item in self.selected)


@dataclass(frozen=True)
class RecognitionResult:
    """单张图片的识别结论。"""

    description: str = ""
    label: str = UNRECOGNIZED_LABEL
    injection: str = ""
    image_hash: str = ""
    fusion: FusionResult = field(default_factory=FusionResult)
    ok: bool = True


@dataclass(frozen=True)
class Query:
    """送给检索层的查询，各字段分开保留是便于按权重拼装与调试。"""

    description: str = ""
    evidence: tuple[str, ...] = ()
    reverse_names: tuple[str, ...] = ()
    reverse_works: tuple[str, ...] = ()
    chat_text: str = ""

    def compose(self) -> str:
        """拼装成检索用文本。反查名字重复一次以抬高其权重。"""
        parts = [self.description, " ".join(self.evidence)]
        if self.reverse_names:
            joined = " ".join(self.reverse_names)
            parts.extend([joined, joined])
        parts.append(" ".join(self.reverse_works))
        if self.chat_text:
            parts.append(self.chat_text)
        return "\n".join(part for part in parts if part)
