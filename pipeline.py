# -*- coding: utf-8 -*-
"""识别流水线编排：依赖注入后可脱机单测（不碰 ctx）。

条目进来、结论出去。所有跨边界动作（视觉调用、反查、向量、日志、落诊断）都由
``PipelinePorts`` 里的回调承担，所以本模块既看不见 ``ctx``，也看不见 asyncio 之外的
世界。从 plugin.py 拆出（重构路线 A）：编排逻辑与运行时状态解耦。
"""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

try:
    from .fusion import FusionOptions, describe_hits, fuse
    from .imaging import sha256_hex
    from .models import UNRECOGNIZED_LABEL, Query, RecognitionResult, TIER_CONFIRMED
    from .prompts import build_knowledge_block
    from .retrieval import EmbeddingIndex, RetrievalOptions, build_reverse_confidence, retrieve
except ImportError:  # pragma: no cover - Runner 只把目录塞进 sys.path 时走这条
    from fusion import FusionOptions, describe_hits, fuse
    from imaging import sha256_hex
    from models import UNRECOGNIZED_LABEL, Query, RecognitionResult, TIER_CONFIRMED
    from prompts import build_knowledge_block
    from retrieval import EmbeddingIndex, RetrievalOptions, build_reverse_confidence, retrieve


@dataclass
class PipelinePorts:
    """流水线的**外部出入口**：全部是对 ctx / 网络 / 插件状态的封装回调。

    这样编排逻辑里没有任何「宿主能力代理」——纯逻辑，可脱机跑。那份代理必须留在
    plugin.py：``check_plugin.py`` 只扫那一个文件推导能力声明，回调在这里被调用、
    声明在 plugin.py 记账，两边各司其职（``tests/test_repository.py`` 守着这条）。
    """

    #: 图片描述：``(image_bytes) -> description``
    describe: Callable[[bytes], Awaitable[str]]
    #: 候选校验：``(image_bytes, catalog) -> (名字, 说明, 原始输出, 描述)``
    identify: Callable[..., Awaitable[tuple]]
    #: 文本向量：``(texts) -> vectors | None``
    embed: Callable[..., Awaitable[Any]]
    #: 联网反查：``(image_bytes) -> (hits, errors)``
    collect_hits: Callable[[bytes], Awaitable[tuple]]
    #: 日志回调（签名与 logging 一致，支持 ``%s`` 惰性格式化）
    log: Callable[..., None]
    #: 调试日志（对应 ``ctx.logger.debug``）
    debug_log: Callable[..., None]


class RecognitionPipeline:
    """单张图片的完整识别流程编排。

    依赖全部**注入**且多半是闭包（``*_provider``）：现状原本就是每次运行时现读
    ``self.config`` / ``self._cache``，闭包读最新值才能保持语义一致——配置热更后
    ``_cache`` 甚至会被重建，直接捕获实例会拿到已被丢弃的那一个。
    """

    def __init__(
        self,
        ports: PipelinePorts,
        *,
        cache_provider: Callable[[], Any],
        index_provider: Callable[[], EmbeddingIndex],
        repository_provider: Callable[[], Any],
        config_provider: Callable[[], Any],
        generation_provider: Callable[[], int],
        diagnosis_sink: Callable[[dict], None],
    ) -> None:
        self._ports = ports
        self._cache_provider = cache_provider
        self._index_provider = index_provider
        self._repository_provider = repository_provider
        self._config_provider = config_provider
        self._generation_provider = generation_provider
        self._diagnosis_sink = diagnosis_sink

    # -------------------------------------------------- 选项/解析器（配置→参数的纯映射）

    def retrieval_options(self) -> RetrievalOptions:
        cfg = self._config_provider().retrieval
        config = self._config_provider()
        return RetrievalOptions(
            max_prompt_characters=config.library.max_prompt_characters,
            pinned_reserve=config.library.pinned_reserve,
            embedding_enabled=cfg.embedding_enabled,
            embed_batch_size=cfg.embed_batch_size,
            weight_embedding=cfg.weight_embedding,
            weight_keyword=cfg.weight_keyword,
            boost_reverse_confirmed=cfg.boost_reverse_confirmed,
            keyword_min_score=cfg.keyword_min_score,
            embed_min_score=cfg.embed_min_score,
        )

    def fusion_options(self) -> FusionOptions:
        cfg = self._config_provider().fusion
        config = self._config_provider()
        return FusionOptions(
            auto_apply_single_source=cfg.auto_apply_single_source,
            require_two_sources_for_auto=cfg.require_two_sources_for_auto,
            conflict_min_confidence=cfg.conflict_min_confidence,
            auto_apply_local_confirm=cfg.auto_apply_local_confirm,
            saucenao_confident_similarity=config.saucenao.confident_similarity,
            saucenao_weak_similarity=config.saucenao.weak_similarity,
        )

    def resolver(self) -> Callable[[str], Any]:
        """名字 → 本地角色的解析函数，交给融合层做跨语言桥接。

        仓库在解析时就捕获：万一这次调用中途卸载了角色库，half-way 换源会让"同一个名字
        两次解析出不同结论"，那种 bug 没法复现。
        """
        repository = self._repository_provider()

        def resolve(raw_name: str):
            if repository is None:
                return None
            character = repository.resolve(raw_name)
            if character is None:
                return None
            return character.character_id, character.name

        return resolve

    # -------------------------------------------------- 主流程

    async def recognize(self, image_bytes: bytes, *, chat_text: str = "") -> RecognitionResult:
        """单张图片的完整识别流程。"""
        if not image_bytes:
            return RecognitionResult(description="图片未能读取", ok=False)
        config = self._config_provider()
        cache = self._cache_provider()
        index = self._index_provider()
        repository = self._repository_provider()
        digest = sha256_hex(image_bytes)
        cache_key = f"{self._generation_provider()}:{digest}"
        diag: dict[str, Any] = {
            "cache": "未命中",
            "threshold": config.fusion.local_confirm_min_score,
        }
        if config.cache.enabled:
            cached = cache.get(cache_key)
            if cached is not None:
                diag.update(
                    cache="命中，复用上次结论（不重跑链路）",
                    tier=cached.fusion.tier,
                    label=cached.label,
                    reason=cached.fusion.reason,
                )
                self._diagnosis_sink(diag)
                return cached

        # 真值判断会踩坑：空库的 bool() 是 False，语义上会被当成"没有库"。
        characters = tuple(repository.characters) if repository is not None else ()

        # 阶段一：先跑反查，再决定要不要独立的图片描述调用。
        #
        # 反查给出角色名信号且融合尚未确认时，describe 与 identify 两次视觉调用合并成
        # 一次：identify 的输出本就含 description 字段（见 prompts 的 JSON 格式），让它
        # 一趟把"描述 + 候选校验"都做了，省下一整趟 30~50s 的视觉调用。这条路径上检索
        # 查询以反查名为主（Query.compose 里反查名重复加权），且被点名的角色有 pinned
        # 保底，检索失去描述文本的影响可控。
        hits, errors = await self._ports.collect_hits(image_bytes)
        preliminary = fuse(hits, resolve=self.resolver(), options=self.fusion_options())
        merged_vision = (
            config.vision.enabled
            and bool(characters)
            and config.library.enabled
            and any(hit.gives_character for hit in hits)
            and preliminary.tier != TIER_CONFIRMED
        )
        if merged_vision:
            description = ""
        else:
            description = await self._ports.describe(image_bytes)
        diag["characters"] = len(characters)
        diag["reverse"] = f"命中 {len(hits)} 条" if hits else "无命中"
        diag["reverse_errors"] = list(errors)
        if errors:
            self._ports.log("warning", "部分反查源失败：%s", "；".join(errors))
        if config.plugin.debug and hits:
            self._ports.debug_log("反查命中：\n%s", describe_hits(hits))

        # 阶段二：相关性检索挑选候选
        relationships: dict[str, str] = {}
        retrieval = None
        if characters and config.library.enabled:
            retrieval = await retrieve(
                characters,
                Query(
                    description=description,
                    reverse_names=tuple(hit.raw_name for hit in hits if hit.gives_character),
                    reverse_works=tuple(hit.work for hit in hits if hit.work),
                    chat_text=chat_text,
                ),
                options=self.retrieval_options(),
                embed=self._ports.embed,
                index=index,
                reverse_confidence=build_reverse_confidence(hits),
            )
            relationships = {
                item.character.character_id: item.character.relationship
                for item in retrieval.selected
                if item.character.relationship
            }
            diag["retrieval_scores"] = [
                (item.character.name, round(item.score, 3)) for item in retrieval.selected[:3]
            ]
            diag["retrieval"] = (
                "、".join(f"{name} {score:.2f}" for name, score in diag["retrieval_scores"])
                or "没有候选过线"
            )
            if retrieval.degraded:
                diag["retrieval"] += "｜向量不可用，已降级为关键词检索"
        else:
            diag["retrieval"] = "角色库为空，没得挑" if not characters else "库检索被配置关闭"

        # 阶段三：仅在有本地候选且反查尚未定论时，做一次候选校验
        vision_names: "tuple[str, ...]" = ()
        vision_note = "未调用"
        vision_raw = ""
        if retrieval is not None and retrieval.selected and config.vision.enabled:
            if preliminary.tier != TIER_CONFIRMED:
                catalog = [
                    {
                        "id": item.character.character_id,
                        "name": item.character.name,
                        "aliases": list(item.character.aliases),
                        "work": item.character.work,
                        # 给全量卡片：只送前 3 条时，模型可能因为看不到决定性特征而判"库里没有"。
                        "appearance_cards": list(item.character.appearance_cards[:8]),
                    }
                    for item in retrieval.selected
                ]
                vision_names, vision_note, vision_raw, vision_description = await self._ports.identify(
                    image_bytes, catalog
                )
                if merged_vision:
                    # 合并路径：这一次视觉调用同时承担"描述"职责；没产出描述才补一趟
                    description = vision_description
                    if not description:
                        description = await self._ports.describe(image_bytes)
            else:
                vision_note = "未调用（反查源已确认，不需要校验）"
        elif retrieval is not None:
            vision_note = "未调用（检索没有选出候选）"
            if merged_vision:
                # 检索没选出候选时 identify 不会跑，描述得补回来
                description = await self._ports.describe(image_bytes)
        diag["vision"] = vision_note
        diag["vision_names"] = list(vision_names)
        diag["vision_raw"] = vision_raw

        local_confirmed = self.local_confirmed(vision_names, retrieval)
        fusion = fuse(
            hits,
            resolve=self.resolver(),
            options=self.fusion_options(),
            vlm_names=vision_names,
            degraded=bool(retrieval and retrieval.degraded),
            local_confirmed=local_confirmed,
        )
        label = fusion.label(relationships, limit=config.plugin.max_characters_per_image)
        diag["local_confirmed"] = [[item[0], item[1]] for item in local_confirmed]
        diag["reverse_candidates"] = [item.display_name for item in fusion.candidates]
        # 真正会触发分歧否决的，只有能解析到库内角色的那些（库外名字多半是别名没登记）
        confirmed_ids = {item[0] for item in local_confirmed}
        diag["reverse_disputes"] = [
            item.display_name for item in fusion.candidates
            if item.character_id and item.character_id not in confirmed_ids
        ]
        diag["tier"] = fusion.tier
        diag["label"] = label
        diag["reason"] = fusion.reason
        diag["description"] = bool(description)
        self._diagnosis_sink(diag)
        injection = ""
        if config.injection.enabled and (retrieval is not None or fusion.works):
            injection = build_knowledge_block(
                # 注意取 .characters（Character 元组）而不是 .selected（ScoredCharacter 元组），
                # 后者没有 name/work 这些字段，传错会得到空知识块且不报错。
                characters=retrieval.characters if retrieval else (),
                fusion=fusion,
                options=config.injection,
                max_chars=config.library.knowledge_block_max_chars,
            )

        result = RecognitionResult(
            description=description or "",
            label=label,
            injection=injection,
            image_hash=digest,
            fusion=fusion,
            ok=True,
        )
        if config.cache.enabled:
            ttl = (
                config.cache.ttl_seconds
                if label != UNRECOGNIZED_LABEL
                else config.cache.negative_ttl_seconds
            )
            if ttl > 0:
                cache.put(cache_key, result, ttl_seconds=ttl)
        return result

    def local_confirmed(
        self, vision_names: "tuple[str, ...]", retrieval: Any
    ) -> "list[tuple[str, str]]":
        """筛出「本地库检索命中 **且** 视觉模型确认」的角色，交给融合层做限流兜底。

        两个条件缺一不可：
        1. 该角色在检索里排进了候选，且分数不低于 ``fusion.local_confirm_min_score``；
        2. 视觉模型返回的名字能解析到**候选里的那个角色**——模型只看得到候选目录，
           所以它报的名字落不进候选，说明这次回答不可用，不能拿来当证据。

        这条通路的现实意义：免费反查源（AnimeTrace）会整段整段地限流，那期间反查
        恒为空；没有它，用户自己攒的角色库在限流窗口里等于不存在。
        """
        if not vision_names or retrieval is None or not retrieval.selected:
            return []
        config = self._config_provider()
        resolve = self.resolver()
        scored = {item.character.character_id: item for item in retrieval.selected}
        threshold = config.fusion.local_confirm_min_score
        confirmed: "list[tuple[str, str]]" = []
        for raw in vision_names:
            resolved = resolve(str(raw).strip())
            if resolved is None:
                continue
            character_id, canonical = resolved
            candidate = scored.get(character_id)
            if candidate is None or candidate.score < threshold:
                continue
            if all(item[0] != character_id for item in confirmed):
                confirmed.append((character_id, canonical))
        return confirmed
