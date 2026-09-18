# -*- coding: utf-8 -*-
"""多源反查结果的融合决策（纯函数，不依赖 ctx，可脱机单测）。

三条不能违背的原则：

1. **不同源的分数不可比，绝不求和。** SauceNAO 的 ``similarity`` 是 booru 检索分，
   AnimeTrace 只有一个 ``not_confident`` 布尔，trace.moe 是帧差相似度 —— 三个不同
   的尺度。跨源只用「有多少个源同意」这一个变量，各源分数只在源内做阈值判断。
2. **trace.moe 永远不能计入角色源。** 它只告诉你"这是哪部动画第几集"，不给角色名。
   把它算进 agreement，会让一个纯作品信号和角色信号"撞车"，凭空造出冲突。
3. **错贴一个人名比不贴更伤。** 所以单源命中默认不自动贴标签，只在多源一致时贴。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

try:
    from .models import (
        CHARACTER_SOURCES,
        SOURCE_LABELS,
        TIER_CONFIRMED,
        TIER_CONFLICT,
        TIER_LOCAL,
        TIER_NONE,
        TIER_SINGLE,
        TIER_WORK_ONLY,
        ClusteredCandidate,
        FusionResult,
        SourceHit,
    )
    from .textutil import clean_display_name, is_plausible_name, normalize_name
except ImportError:  # pragma: no cover - 取决于加载方式
    from models import (
        CHARACTER_SOURCES,
        SOURCE_LABELS,
        TIER_CONFIRMED,
        TIER_CONFLICT,
        TIER_LOCAL,
        TIER_NONE,
        TIER_SINGLE,
        TIER_WORK_ONLY,
        ClusteredCandidate,
        FusionResult,
        SourceHit,
    )
    from textutil import clean_display_name, is_plausible_name, normalize_name

#: 把源返回的名字解析到本地角色；返回 (character_id, 正名) 或 None
Resolver = Callable[[str], "tuple[str, str] | None"]


@dataclass(frozen=True)
class FusionOptions:
    """融合阈值。除 ``auto_apply_single_source`` 外都是待真机调参项。"""

    auto_apply_single_source: bool = False
    require_two_sources_for_auto: bool = True
    conflict_min_confidence: float = 0.7
    #: 反查源全无命中时，是否允许"本地库检索 + 视觉模型确认"这一对证据自动贴标签。
    #: 默认开：它是**双证据**，比单源命中（默认不贴）更严，且不依赖外部源可用性。
    auto_apply_local_confirm: bool = True
    #: SauceNAO 只有相似度可用，需要映射成 confident / weak 两档
    saucenao_confident_similarity: float = 85.0
    saucenao_weak_similarity: float = 60.0


def saucenao_confidence(score: float, options: FusionOptions) -> bool | None:
    """SauceNAO 相似度 → 三态：True 可信 / False 弱 / None 丢弃。"""
    if score >= options.saucenao_confident_similarity:
        return True
    if score >= options.saucenao_weak_similarity:
        return False
    return None


def _cluster_key(raw_name: str, resolve: Resolver) -> tuple[str, str, str]:
    """决定聚类键：能解析到本地角色就用 character_id，否则退回归一名字。

    这是跨语言问题的唯一可行解法 —— ``初音ミク`` / ``初音未来`` / ``Hatsune Miku``
    靠字符串运算永远等价不了，只能靠本地角色库的别名/桥接表当"翻译层"。
    """
    resolved = resolve(raw_name)
    if resolved is not None:
        character_id, canonical = resolved
        return f"id:{character_id}", canonical, character_id
    return f"name:{normalize_name(raw_name)}", clean_display_name(raw_name), ""


def fuse(
    hits: Sequence[SourceHit],
    *,
    resolve: Resolver | None = None,
    options: FusionOptions | None = None,
    vlm_names: Sequence[str] = (),
    degraded: bool = False,
    local_confirmed: Sequence[tuple[str, str]] = (),
) -> FusionResult:
    """把多源命中融合成一个决策。

    Args:
        hits: 各反查源的命中（含只给作品的源）。
        resolve: 名字 → 本地角色的解析函数（通常来自 ``CharacterRepository.resolve``）。
        options: 阈值。
        vlm_names: 视觉模型独立给出的候选角色名，用于"单源 + VLM 一致"升级为确认。
        degraded: 检索层降级中。降级时**禁止自动贴标签**。
        local_confirmed: 已由**调用方**验证过的"本地库检索命中 + 视觉模型确认"候选，
            形如 ``(character_id, 显示名)``。仅在反查源没有任何高置信候选时才被采用，
            是限流期不让本地角色库变成摆设的兜底。判定条件（检索分达标、VLM 确实
            在候选目录里选出了它）留在调用方，因为只有那里拿得到检索分数。
    """
    options = options or FusionOptions()
    resolver: Resolver = resolve or (lambda _name: None)

    works = _collect_works(hits)
    clusters: dict[str, dict[str, object]] = {}

    for hit in hits:
        if not hit.gives_character:
            continue
        display = clean_display_name(hit.raw_name)
        if not is_plausible_name(display):
            continue
        key, name, character_id = _cluster_key(display, resolver)
        if character_id:
            name = clean_display_name(name)
        bucket = clusters.setdefault(
            key,
            {"name": name, "character_id": character_id, "sources": set(), "works": [], "confident": False},
        )
        bucket["sources"].add(hit.source)  # type: ignore[union-attr]
        if hit.work:
            bucket["works"].append(clean_display_name(hit.work))  # type: ignore[union-attr]
        if hit.confident:
            bucket["confident"] = True

    candidates = tuple(
        sorted(
            (
                ClusteredCandidate(
                    key=key,
                    display_name=str(bucket["name"]),
                    character_id=str(bucket["character_id"]),
                    sources=tuple(sorted(bucket["sources"])),  # type: ignore[arg-type]
                    works=tuple(dict.fromkeys(bucket["works"])),  # type: ignore[arg-type]
                    confident=bool(bucket["confident"]),
                )
                for key, bucket in clusters.items()
            ),
            key=lambda item: (-item.source_count, not item.confident, item.display_name),
        )
    )

    confident_candidates = tuple(item for item in candidates if item.confident)

    if not confident_candidates:
        # 低置信候选**不构成一票否决**：它连自己都不确定（否则就进 confident 了），
        # 拿它去否定"本地库 + 视觉模型"这对独立证据没有道理。真正的否决条件是
        # **分歧**——反查源报的是另一个角色（见 _local_confirm_result）。
        rescued = _local_confirm_result(local_confirmed, options, degraded, candidates, works)
        if rescued is not None:
            return rescued
        # 有角色名但全部低置信 / 只有作品信号 —— 只注入上下文，不贴标签。
        return FusionResult(
            tier=TIER_WORK_ONLY if (works or candidates) else TIER_NONE,
            candidates=candidates,
            works=works,
            auto_apply=False,
            degraded=degraded,
            reason="无高置信角色候选" if (works or candidates) else "所有反查源均无命中",
        )

    # 冲突：两个都够可信、但指向不同角色。低置信的分歧不算冲突，采信高置信的那个。
    top = confident_candidates[0]
    conflicting = [item for item in confident_candidates[1:] if item.key != top.key]
    if conflicting:
        return FusionResult(
            tier=TIER_CONFLICT,
            candidates=candidates,
            works=works,
            auto_apply=False,
            conflict=True,
            degraded=degraded,
            reason=(
                f"{top.source_label} 主张「{top.display_name}」，"
                f"{conflicting[0].source_label} 主张「{conflicting[0].display_name}」，双方均达置信阈值"
            ),
        )

    vlm_agrees = _vlm_agrees(top, vlm_names, resolver)

    enough_sources = top.source_count >= (2 if options.require_two_sources_for_auto else 1)
    confirmed = enough_sources or vlm_agrees
    if confirmed:
        tier = TIER_CONFIRMED
        # 检索降级时不能自动贴：降级意味着候选池本身可能不完整。
        auto_apply = not degraded
        reason = (
            f"{top.source_count} 个源一致指向「{top.display_name}」"
            if enough_sources
            else f"单源命中「{top.display_name}」且与视觉模型判断一致"
        )
    else:
        tier = TIER_SINGLE
        auto_apply = bool(options.auto_apply_single_source) and not degraded
        reason = f"仅 {top.source_label} 单源命中「{top.display_name}」"

    return FusionResult(
        tier=tier,
        display_names=(top.display_name,),
        candidates=candidates,
        works=works,
        auto_apply=auto_apply,
        conflict=False,
        degraded=degraded,
        reason=reason,
    )


def _local_confirm_result(
    local_confirmed: Sequence[tuple[str, str]],
    options: FusionOptions,
    degraded: bool,
    candidates: Sequence[ClusteredCandidate],
    works: Sequence[str],
) -> "FusionResult | None":
    """反查源集体失声时的兜底决策。

    免费反查源限流是常态（真机实测 AnimeTrace 会连续返回 429）。没有兜底的话，
    限流窗口里每一张图都是"库里有这个人、却报未识别"——用户攒的角色库在需要它的
    时刻恰好用不上。

    为什么敢自动贴：这里的输入已经是**两条独立证据的交集**（本地库外观卡检索命中，
    且视觉模型在候选目录里独立选中了同一个角色），比"单源命中只注入不贴"更严；
    而且两条证据都来自用户自己维护的数据，不受第三方服务可用性影响。

    检索降级时一律不贴：降级意味着候选池本身可能不完整，"没搜到别人"不等于
    "就是他"。
    """
    if not local_confirmed or degraded or not options.auto_apply_local_confirm:
        return None
    # 分歧否决：反查源（低置信）**明确指向库内另一个角色**时，停下来。
    # 错贴一个人名比不贴更伤，而"库里有两个人都像"正是最该停的信号。
    #
    # 但只对**能解析到库内角色**的名字生效：解析不到的多半是别名没登记
    # （跨语言场景里这是常态——源给日文名、库里只有中文正名），把它当分歧会让
    # 整个兜底在跨语言图上永远失效。何况低置信本身已经表明那个源自己也不确定。
    confirmed_ids = {character_id for character_id, _ in local_confirmed}
    if any(item.character_id and item.character_id not in confirmed_ids for item in candidates):
        return None
    # 标签是从 ``candidates`` 渲染的（``FusionResult.label`` 只认 gives_label 的候选），
    # 所以这里必须造出候选实体——只填 display_names 会渲染成"未识别"。
    confirmed: list[ClusteredCandidate] = []
    seen: set[str] = set()
    for character_id, name in local_confirmed:
        display = clean_display_name(name)
        key = f"id:{character_id}"
        if not display or key in seen:
            continue
        seen.add(key)
        confirmed.append(
            ClusteredCandidate(
                key=key,
                display_name=display,
                character_id=character_id,
                sources=("local_library",),
                confident=True,
            )
        )
    if not confirmed:
        return None
    return FusionResult(
        tier=TIER_LOCAL,
        display_names=tuple(item.display_name for item in confirmed),
        candidates=tuple(candidates) + tuple(confirmed),
        works=works,
        auto_apply=True,
        conflict=False,
        degraded=degraded,
        reason=(
            "反查源无命中，但本地库外观卡检索命中「"
            + "、".join(item.display_name for item in confirmed[:3])
            + "」，且视觉模型在候选目录里确认了它"
        ),
    )


def _vlm_agrees(candidate: ClusteredCandidate, vlm_names: Sequence[str], resolve: Resolver) -> bool:
    """视觉模型是否独立指向同一个角色。

    两条路径都要试：VLM 直接用同一个名字，或 VLM 用了另一个别名（靠库解析到同一
    个 character_id）。只看字面相等会漏掉"视觉模型说日文名、反查源给中文名"的常见情形。
    """
    for raw in vlm_names:
        name = str(raw).strip()
        if not name:
            continue
        if candidate.character_id:
            resolved = resolve(name)
            if resolved is not None and resolved[0] == candidate.character_id:
                return True
        if normalize_name(name) == normalize_name(candidate.display_name):
            return True
    return False


def _collect_works(hits: Iterable[SourceHit]) -> tuple[str, ...]:
    """汇总所有源（**含** trace.moe）声明的作品名，供作品上下文注入用。"""
    values: list[str] = []
    for hit in hits:
        work = clean_display_name(hit.work)
        if work and work.casefold() not in {item.casefold() for item in values}:
            values.append(work)
    return tuple(values)


def describe_hits(hits: Sequence[SourceHit]) -> str:
    """把命中渲染成给 LLM 看的紧凑文本（不含任何二进制）。"""
    lines: list[str] = []
    for hit in hits:
        label = SOURCE_LABELS.get(hit.source, hit.source)
        name = clean_display_name(hit.raw_name) or "（无角色名）"
        parts = [f"{label}: {name}"]
        if hit.work:
            parts.append(f"作品 {clean_display_name(hit.work)}")
        if not hit.confident:
            parts.append("低置信")
        if hit.detail:
            parts.append(hit.detail)
        lines.append("，".join(parts))
    return "\n".join(lines)
