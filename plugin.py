# -*- coding: utf-8 -*-
"""角色识别插件：让 MaiBot 认出聊天图片里的二次元角色，并且真的"认识"她。

相对参考实现（QlzqQlzq/character-knowledge-plugin）修掉的四件事：

1. 角色库进提示词的候选由**相关性检索**挑，不再取列表前 40 个——后者的后果是
   库超过 40 个角色后新教的角色永远进不了提示词。
2. 除 ``图片[角色名]`` 之外，额外把角色的人设/作品/关系注入模型请求，模型不再
   "知道名字但一无所知"。
3. 反查融合层按多源设计，并对"仅作品源"与"角色源"分开计数，避免凭空制造冲突。
4. 角色库损坏时降级为空库并留证，不再 raise 把 on_load 打挂。

**本文件是唯一允许出现 ``self.ctx`` 的地方**：``check_plugin.py`` 只扫描本文件推导
能力名，其它模块里写 ctx 会造成"静态检查全绿、真机拒绝授权"。纯逻辑一律写在
textutil / models / fusion / retrieval / imaging / runtime / sources / vision / prompts 里。
"""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence
from urllib.parse import urlsplit

from maibot_sdk import Command, Field, HookHandler, MaiBotPlugin, PluginConfigBase, Tool
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder, ToolParameterInfo, ToolParamType

try:
    from .fusion import FusionOptions, describe_hits, fuse
    from .imaging import (
        download_image,
        extract_image_payload,
        host_resolves_to_public,
        sha256_hex,
    )
    from .inject import apply_injection, apply_rewrite
    from .models import (
        UNRECOGNIZED_LABEL,
        Character,
        Query,
        RecognitionResult,
        SourceHit,
        TIER_CONFIRMED,
        TIER_LOCAL,
    )
    from .prompts import build_capability_hint, build_knowledge_block
    from .repository import CharacterRepository
    from .retrieval import EmbeddingIndex, RetrievalOptions, build_reverse_confidence, retrieve
    from .runtime import CircuitBreaker, TTLCache
    from .sources import IMPLEMENTED_SOURCES, SourceConfig, probe_endpoint, run_source
    from .textutil import CORE_APPEARANCE_CATEGORIES, group_appearance_cards
    from .vision import (
        VisionError,
        build_appearance_cards,
        describe_image,
        compress_appearance_cards,
        describe_vision_dropouts,
        identify_image,
        rescue_unlabeled_candidates,
    )
except ImportError:  # pragma: no cover - Runner 只把目录塞进 sys.path 时走这条
    from fusion import FusionOptions, describe_hits, fuse
    from imaging import (
        download_image,
        extract_image_payload,
        host_resolves_to_public,
        sha256_hex,
    )
    from inject import apply_injection, apply_rewrite
    from models import (
        UNRECOGNIZED_LABEL,
        Character,
        Query,
        RecognitionResult,
        SourceHit,
        TIER_CONFIRMED,
        TIER_LOCAL,
    )
    from prompts import build_capability_hint, build_knowledge_block
    from repository import CharacterRepository
    from retrieval import EmbeddingIndex, RetrievalOptions, build_reverse_confidence, retrieve
    from runtime import CircuitBreaker, TTLCache
    from sources import IMPLEMENTED_SOURCES, SourceConfig, probe_endpoint, run_source
    from textutil import CORE_APPEARANCE_CATEGORIES, group_appearance_cards
    from vision import (
        VisionError,
        build_appearance_cards,
        describe_image,
        compress_appearance_cards,
        describe_vision_dropouts,
        identify_image,
        rescue_unlabeled_candidates,
    )

#: 注入标记。同时用于幂等判定与对用户的说明，改它要同步 README。
INJECT_MARKER = "【角色识别】"
#: 识图工具对外暴露的名字，供能力提示引用。
TOOL_NAMES = ("recognize_image", "query_character", "search_character_library", "reverse_lookup_image")

#: 角色登记的空闲窗口：**每收到一批图就重置**。用"空闲"而不是"绝对期限"，是因为
#: 连续发图的间隔不该把整轮登记判成超时——否则图越多越容易半路挂掉。
CHARACTER_ADD_IDLE_SECONDS = 300.0

#: 每个会话保留的最近图片数。``/识图修正`` 不引用消息时用的就是这段历史——
#: 只留最后一张的话，"不引用"就等于"只能补一张卡"，跟引用没区别。
RECENT_IMAGE_LIMIT = 8
#: 这段历史的有效时间窗。没有它会出事：十分钟前发的图会在下一次命令里被当成"刚发的"。
RECENT_IMAGE_SECONDS = 900.0
#: 保留图片记忆的会话数上限（总字节数另有上限，见 ``_latest_max_bytes``）。
RECENT_IMAGE_SESSIONS = 64

#: LLM 能力调用的 RPC 超时。Host 的 ``cap.call`` 默认只有 **30s**，而真机视觉模型实测
#: 要 54.9s——SDK 的 ``ctx.llm.generate`` 转发时**没有传** ``timeout_ms``，于是每次都在
#: 30s 被切断（``RPCError: [E_TIMEOUT] 请求 cap.call 超时 (30000ms)``）。底层
#: ``ctx.call_capability`` 是支持覆盖的，所以这里自己带一个够大的值。
LLM_RPC_TIMEOUT_MS = 180_000

#: 整理外观卡默认用哪个 Host 任务。
#:
#: **不能沿用视觉任务**：整理是纯文本活，挂到视觉大模型上实测单次 54.9~56.8s，而 Host 的
#: ``cap.call`` RPC 硬超时是 **30s**——插件侧把超时调到 90s 也没用，因为先断的是 RPC 那一层。
#: 换通用小任务才有活路。
COMPRESS_DEFAULT_TASK = "utils"

#: 检索分的实测波动区间（2026-09-18 真机：同一张图、同一份卡片，分数在 0.34~0.40 之间漂，
#: 因为 description 由视觉模型每次现生成、措辞一变向量就变）。阈值落进这个区间会表现为
#: "同一张图时贴时不贴"——比设错更难查，因为它看起来像随机故障。所以状态里要喊出来。
SCORE_JITTER_BAND = (0.30, 0.42)


@dataclass
class PendingAddition:
    """等待图片的角色登记。

    图片是**连续收集**的：每来一批就合并进外观卡，直到管理员显式结束或空闲超时。
    只收一张的话，给一个角色建卡就得反复敲 /识图修正，而每次还得再引用一遍图片——
    真机上攒出一张完整的外观卡要来回七八次。
    """

    created_at: float
    name: str
    work: str = ""
    relationship: str = ""
    images: int = 0
    cards: int = 0

    def touched(self) -> None:
        """收到新图后把空闲计时推后。"""
        self.created_at = time.monotonic()

    def expired(self, now: float) -> bool:
        return now - self.created_at > CHARACTER_ADD_IDLE_SECONDS


# ══════════════════════════════════════════════════════════════ 配置模型


class PluginSectionConfig(PluginConfigBase):
    """插件基础配置。"""

    __ui_label__ = "插件"
    __ui_icon__ = "scan-face"
    __ui_order__ = 0

    enabled: bool = Field(default=True, description="是否启用图片角色识别")
    config_version: str = Field(default="1.0.0", description="配置版本")
    debug: bool = Field(default=False, description="输出详细诊断日志")
    max_images_per_message: int = Field(default=4, ge=1, le=20, description="单条消息最多识别的图片数")
    max_characters_per_image: int = Field(default=3, ge=1, le=10, description="单图最多保留的角色数")
    max_concurrency: int = Field(default=1, ge=1, le=4, description="视觉与反查的最大并发")
    image_timeout_seconds: float = Field(
        default=50.0,
        ge=3.0,
        le=170.0,
        description=(
            "单张图片的识别超时。要覆盖得住慢模型——真机实测视觉模型单次 27~53s，"
            "25s 会让一半识别直接失败。受 message_timeout_seconds 总预算约束"
        ),
    )
    card_timeout_seconds: float = Field(
        default=90.0,
        ge=3.0,
        le=170.0,
        description=(
            "建卡 / 补卡时单张图的抽取超时。比识别宽松：那时用户已经预期要等，"
            "没必要和实时识别共用同一个短超时"
        ),
    )
    message_timeout_seconds: float = Field(default=110.0, ge=10.0, le=115.0, description="单条消息的识别总预算")


class VisionSectionConfig(PluginConfigBase):
    """视觉识别配置。"""

    __ui_label__ = "视觉识别"
    __ui_icon__ = "eye"
    __ui_order__ = 1

    enabled: bool = Field(default=True, description="是否启用视觉识别")
    provider: Literal["host", "gemini", "openai"] = Field(
        default="host", description="host 复用 MaiBot 已配置的视觉任务，无需再填密钥"
    )
    task_name: str = Field(default="vlm", description="Host 模型任务名（仅 host 通道使用）")
    model_name: str = Field(default="", description="具体模型名，留空则用任务默认")
    api_key: str = Field(default="", description="直连通道的密钥（host 通道留空）")
    base_url: str = Field(default="", description="直连通道的接口地址（host 通道留空）")
    max_tokens: int = Field(default=700, ge=64, le=4096, description="视觉请求的最大输出 token")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="视觉请求温度")
    max_upload_bytes: int = Field(default=4194304, ge=65536, description="发给视觉服务的图片体积上限")


class LibrarySectionConfig(PluginConfigBase):
    """本地角色库配置。"""

    __ui_label__ = "角色库"
    __ui_icon__ = "book-user"
    __ui_order__ = 2

    enabled: bool = Field(default=True, description="是否启用本地角色库")
    file_name: str = Field(default="characters.json", description="角色库文件名（位于插件数据目录）")
    admin_ids: list[str] = Field(default_factory=list, description="允许管理角色库的账号，如 qq:123456")
    allow_unrestricted_admin: bool = Field(
        default=False, description="任何人都可管理角色库（仅本地测试用，服务器上不要开）"
    )
    import_from: str = Field(default="", description="从旧插件的 characters.json 绝对路径导入（仅在库不存在时执行）")
    max_prompt_characters: int = Field(default=12, ge=1, le=60, description="进视觉提示词的角色数上限")
    pinned_reserve: int = Field(default=5, ge=0, le=30, description="为反查命中的角色预留的候选位数")
    knowledge_block_max_chars: int = Field(default=1200, ge=120, le=6000, description="注入模型请求的知识块字符上限")
    compress_on_finish: bool = Field(
        default=True,
        description=(
            "结束角色登记时用 LLM 把同类措辞整理一遍（保留造型差异与细节；失败时保持原样）。"
            "只在 host 视觉通道下生效"
        ),
    )
    compress_task_name: str = Field(
        default=COMPRESS_DEFAULT_TASK,
        description=(
            "整理外观卡用的 Host 模型任务名。**别填成视觉任务**：整理是纯文本活，挂到视觉"
            f"大模型上实测单次 54.9~56.8s，而 Host 的 cap.call RPC 硬超时只有 30s，必死。"
            f"默认 {COMPRESS_DEFAULT_TASK}（通用小任务）；填 vlm 可以退回旧行为"
        ),
    )


class RetrievalSectionConfig(PluginConfigBase):
    """检索与向量配置。"""

    __ui_label__ = "检索"
    __ui_icon__ = "search"
    __ui_order__ = 3

    embedding_enabled: bool = Field(default=True, description="启用向量检索；失败时自动降级为关键词检索")
    embed_task_name: str = Field(default="embedding", description="Host 的 embedding 任务名")
    embed_model_name: str = Field(default="", description="embedding 具体模型名，留空用任务默认")
    embed_batch_size: int = Field(default=32, ge=1, le=256, description="批量取向量的分块大小")
    embed_timeout_seconds: float = Field(default=15.0, ge=1.0, le=120.0, description="单次取向量的超时")
    embed_cooldown_seconds: float = Field(default=900.0, ge=0.0, description="向量不可用后的冷却秒数")
    weight_embedding: float = Field(default=0.55, ge=0.0, le=1.0, description="向量分权重")
    weight_keyword: float = Field(default=0.35, ge=0.0, le=1.0, description="关键词分权重")
    boost_reverse_confirmed: float = Field(default=0.30, ge=0.0, le=1.0, description="反查可解析到该角色时的加成")
    keyword_min_score: float = Field(default=0.12, ge=0.0, le=1.0, description="关键词命中过滤阈值")
    embed_min_score: float = Field(default=0.35, ge=0.0, le=1.0, description="向量命中过滤阈值")


class AnimeTraceSectionConfig(PluginConfigBase):
    """AnimeTrace 源配置。"""

    __ui_label__ = "AnimeTrace"
    __ui_icon__ = "image-search"
    __ui_order__ = 4

    enabled: bool = Field(default=True, description="启用 AnimeTrace 反查（免费、无需密钥）")
    url: str = Field(default="https://api.animetrace.com", description="接口地址")
    timeout_seconds: float = Field(default=15.0, ge=1.0, le=120.0, description="请求超时")
    max_upload_bytes: int = Field(default=900000, ge=65536, description="上传体积上限，超限自动压缩")
    max_candidates: int = Field(default=3, ge=1, le=10, description="最多保留的候选数")


class SauceNaoSectionConfig(PluginConfigBase):
    """SauceNAO 源配置（尚未接入 adapter，仅用于探测）。"""

    __ui_label__ = "SauceNAO"
    __ui_icon__ = "flask"
    __ui_order__ = 5

    enabled: bool = Field(default=False, description="启用 SauceNAO 反查（需自行注册免费密钥）")
    url: str = Field(default="https://saucenao.com", description="接口地址")
    api_key: str = Field(default="", description="SauceNAO API Key")
    timeout_seconds: float = Field(default=20.0, ge=1.0, le=120.0, description="请求超时")
    max_upload_bytes: int = Field(default=2000000, ge=65536, description="上传体积上限")
    max_candidates: int = Field(default=3, ge=1, le=10, description="最多保留的候选数")
    confident_similarity: float = Field(default=85.0, ge=0.0, le=100.0, description="可信阈值")
    weak_similarity: float = Field(default=60.0, ge=0.0, le=100.0, description="弱匹配下限")


class TraceMoeSectionConfig(PluginConfigBase):
    """trace.moe 源配置（只给作品，不给角色名；默认关闭）。"""

    __ui_label__ = "trace.moe"
    __ui_icon__ = "film"
    __ui_order__ = 6

    enabled: bool = Field(default=False, description="启用 trace.moe 作品反查（仅识别动画截图出处）")
    url: str = Field(default="https://api.trace.moe", description="接口地址")
    timeout_seconds: float = Field(default=20.0, ge=1.0, le=120.0, description="请求超时")
    max_upload_bytes: int = Field(default=4000000, ge=65536, description="上传体积上限")
    max_candidates: int = Field(default=3, ge=1, le=10, description="最多保留的候选数")


class FusionSectionConfig(PluginConfigBase):
    """融合决策配置。"""

    __ui_label__ = "融合"
    __ui_icon__ = "git-merge"
    __ui_order__ = 7

    auto_apply_single_source: bool = Field(
        default=False, description="单源命中是否自动贴标签（默认否：错贴一个人名比不贴更伤）"
    )
    require_two_sources_for_auto: bool = Field(default=True, description="自动贴标签是否要求至少两个源一致")
    conflict_min_confidence: float = Field(default=0.7, ge=0.0, le=1.0, description="判定冲突所需的最低置信度")
    auto_apply_local_confirm: bool = Field(
        default=True,
        description=(
            "反查源全无命中时，允许「本地库外观卡检索 + 视觉模型确认」这对双证据自动贴标签。"
            "关掉后，只要反查源限流/挂掉，库里的角色也会一律报未识别"
        ),
    )
    local_confirm_min_score: float = Field(
        default=0.25, ge=0.0, le=2.0,
        description=(
            "本地库候选参与上述双证据判定所需的最低检索分。该分是 0.55×向量 + 0.35×关键词的"
            "加权和，默认 0.25 对应「两条入选线都刚好过」（0.55×0.35 + 0.35×0.12 ≈ 0.23）。"
            "注意别把它设成 0.35：description 由视觉模型每次现生成、措辞一变向量就变，"
            "实测同一张图在 0.34~0.40 之间浮动，阈值卡在波动带里等于掷骰子"
        ),
    )


class InjectionSectionConfig(PluginConfigBase):
    """角色知识注入配置。"""

    __ui_label__ = "知识注入"
    __ui_icon__ = "book"
    __ui_order__ = 8

    enabled: bool = Field(default=True, description="是否注入角色知识")
    inject_into_replyer: bool = Field(default=True, description="注入回复模型请求")
    inject_into_planner: bool = Field(default=True, description="注入 Planner 请求")
    inject_persona: bool = Field(default=True, description="注入角色设定")
    inject_work: bool = Field(default=True, description="注入所属作品")
    inject_aliases: bool = Field(default=False, description="注入别名")
    inject_appearance: bool = Field(default=False, description="注入外观卡")

    max_chars: int = Field(default=1200, ge=120, le=6000, description="注入内容字符上限")


class CacheSectionConfig(PluginConfigBase):
    """缓存配置。"""

    __ui_label__ = "缓存"
    __ui_icon__ = "archive"
    __ui_order__ = 9

    enabled: bool = Field(default=True, description="是否启用识别结果缓存")
    max_entries: int = Field(default=256, ge=16, le=4096, description="缓存条目上限")
    ttl_seconds: float = Field(default=3600.0, ge=60.0, description="缓存有效期")
    negative_ttl_seconds: float = Field(default=300.0, ge=0.0, description="未识别结果的缓存有效期")


class CircuitSectionConfig(PluginConfigBase):
    """熔断配置。"""

    __ui_label__ = "熔断"
    __ui_icon__ = "shield"
    __ui_order__ = 10

    failures: int = Field(default=2, ge=1, le=10, description="连续失败多少次后暂停该服务")
    cooldown_seconds: float = Field(default=60.0, ge=1.0, description="暂停秒数，到期自动试探")


class CharacterRecognizerConfig(PluginConfigBase):
    """角色识别插件配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    vision: VisionSectionConfig = Field(default_factory=VisionSectionConfig)
    library: LibrarySectionConfig = Field(default_factory=LibrarySectionConfig)
    retrieval: RetrievalSectionConfig = Field(default_factory=RetrievalSectionConfig)
    anime_trace: AnimeTraceSectionConfig = Field(default_factory=AnimeTraceSectionConfig)
    saucenao: SauceNaoSectionConfig = Field(default_factory=SauceNaoSectionConfig)
    trace_moe: TraceMoeSectionConfig = Field(default_factory=TraceMoeSectionConfig)
    fusion: FusionSectionConfig = Field(default_factory=FusionSectionConfig)
    injection: InjectionSectionConfig = Field(default_factory=InjectionSectionConfig)
    cache: CacheSectionConfig = Field(default_factory=CacheSectionConfig)
    circuit: CircuitSectionConfig = Field(default_factory=CircuitSectionConfig)


# ══════════════════════════════════════════════════════════════ 插件主体


class CharacterRecognizerPlugin(MaiBotPlugin):
    """识别聊天图片中的二次元角色并注入角色知识。"""

    config_model = CharacterRecognizerConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # 必须最先调用：漏掉这一句会让整个 Runner 崩溃，而不是只让本插件失败。
        super().__init__(*args, **kwargs)
        self._repository: CharacterRepository | None = None
        self._cache = TTLCache()
        self._breaker = CircuitBreaker()
        self._index = EmbeddingIndex()
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(1)
        self._tasks: set[asyncio.Task] = set()
        #: 视觉超时的建议每次加载只提示一次——反复刷同一条建议会淹掉别的日志。
        self._vision_timeout_hinted = False
        #: 会话 → [(图片, 标签, 收到时刻)]，保留最近若干张（见 RECENT_IMAGE_LIMIT）。
        self._latest_images: dict[str, list[tuple[bytes, str, float]]] = {}
        self._latest_bytes = 0
        self._latest_max_bytes = 32 * 1024 * 1024
        self._pending_additions: dict[str, PendingAddition] = {}
        self._generation = 0
        self._embed_unavailable_until = 0.0
        self._embed_single_fallback_logged = False
        self._hook_payload_logged = False
        self._logged_injection_keys = False
        #: 最近一次识图的链路诊断，供 /识别探测 展开。""未识别""是个黑盒结论——
        #: 不记录岔路口，用户只能反复换图试，而真正的原因可能在检索、VLM 或限流。
        self._last_diagnosis: "dict[str, Any] | None" = None

    # ---------------------------------------------------------- 生命周期

    async def on_load(self) -> None:
        """插件加载：准备目录、加载角色库。任何单点失败都不应阻止插件启动。"""
        self._rebuild_runtime()
        self._repository = CharacterRepository(self._characters_path(), log=self._log)
        if self.config.library.import_from:
            self._repository.import_from(self.config.library.import_from)
        self._repository.load()
        try:
            self._repository.ensure_exists()
        except Exception as exc:  # 防御：数据目录不可写也要能跑
            self._log("error", f"角色库初始化失败：{exc}")
        self.ctx.logger.info(
            "角色识别插件已加载：角色库 %s 个角色，AnimeTrace=%s，视觉通道=%s",
            len(self._repository),
            "开" if self.config.anime_trace.enabled else "关",
            self.config.vision.provider if self.config.vision.enabled else "关",
        )

    async def on_unload(self) -> None:
        """插件卸载：取消后台任务并释放全部缓存，避免卸载后仍有 task 在跑。"""
        for task in list(self._tasks):
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._cache.clear()
        self._index.clear()
        self._breaker.reset()
        self._latest_images.clear()
        self._pending_additions.clear()
        self.ctx.logger.info("角色识别插件已卸载，后台任务与缓存已清理")

    async def on_config_update(self, scope: str, config_data: dict, version: str) -> None:
        """配置热重载：作废在途识别、清缓存、重载角色库并重建向量索引。"""
        del config_data, version
        self._generation += 1
        self._rebuild_runtime()
        self._cache.clear()
        self._index.clear()
        self._breaker.reset()
        self._embed_unavailable_until = 0.0
        if self._repository is not None:
            self._repository.load()
        self.ctx.logger.info("配置已更新（scope=%s），缓存与向量索引已失效", scope)

    # ---------------------------------------------------------- 基础设施适配

    def _rebuild_runtime(self) -> None:
        cache_cfg = self.config.cache
        self._cache = TTLCache(max_entries=cache_cfg.max_entries, ttl_seconds=cache_cfg.ttl_seconds)
        self._breaker = CircuitBreaker(
            failures=self.config.circuit.failures, cooldown_seconds=self.config.circuit.cooldown_seconds
        )
        self._semaphore = asyncio.Semaphore(max(1, self.config.plugin.max_concurrency))

    def _log(self, level: str, message: str, *args: Any) -> None:
        """给纯模块用的日志回调（纯模块不许碰 ctx）。

        签名与 ``logging`` 一致，所以 ``self._log("warning", "失败：%s", err)`` 这种
        惰性格式化写法也能用——不必在每个调用点先手动拼字符串。
        """
        logger = self.ctx.logger
        target = {"error": logger.error, "warning": logger.warning}.get(level, logger.info)
        if args:
            target(message, *args)
        else:
            target(message)

    def _characters_path(self) -> Path:
        name = str(self.config.library.file_name or "characters.json").strip() or "characters.json"
        # 用户输入不直接当路径用：只取文件名部分，杜绝 ../ 逃逸。
        return Path(self.ctx.paths.data_dir) / Path(name).name

    def _stream_id(self, kwargs: dict[str, Any]) -> str:
        for key in ("stream_id", "chat_id", "session_id", "stream"):
            value = kwargs.get(key)
            if value:
                return str(value)
        message = kwargs.get("message")
        if isinstance(message, dict):
            return str(message.get("session_id") or message.get("stream_id") or "")
        return ""

    @staticmethod
    def _normalize_account(value: Any) -> str:
        """账号归一化。**配置侧与运行时侧必须走同一个函数**，否则会出现"填了管理员却被拒"。"""
        text = str(value or "").strip().casefold()
        if ":" in text:
            text = text.split(":")[-1].strip()
        return text

    def _is_admin(self, kwargs: dict[str, Any]) -> bool:
        if bool(kwargs.get("is_local_operator")):
            return True
        if self.config.library.allow_unrestricted_admin:
            return True
        admins = {self._normalize_account(item) for item in (self.config.library.admin_ids or [])}
        admins.discard("")
        if not admins:
            return False
        actor = self._normalize_account(kwargs.get("user_id"))
        if not actor:
            message = kwargs.get("message")
            if isinstance(message, dict):
                info = message.get("message_info")
                if isinstance(info, dict):
                    user_info = info.get("user_info")
                    if isinstance(user_info, dict):
                        actor = self._normalize_account(user_info.get("user_id"))
        return bool(actor) and actor in admins

    async def _reply(self, stream_id: str, text: str) -> bool:
        """命令回执。命令的返回值不会自动发到群里，必须显式发送。"""
        if not stream_id or not text:
            return False
        try:
            return bool(await self.ctx.send.text(text, stream_id))
        except Exception as exc:
            self.ctx.logger.error("回执发送失败：%s", exc, exc_info=True)
            return False

    async def _generate(self, **kwargs: Any) -> dict:
        """适配 Host 的 llm.generate（vision.py 只认这个形状）。

        **显式带上 RPC 超时**。SDK 的 ``ctx.llm.generate`` 在转发到 ``cap.call`` 时没有传
        ``timeout_ms``，于是固定吃 30 秒的默认值：真机视觉模型实测 54.9s，必然在 30s 被
        切断。而底层 ``ctx.call_capability`` 本身支持覆盖——绕开那层封装即可。

        返回形状不变：两条路都过 ``_normalize_capability_result``，``ctx.llm.generate``
        只额外补了个 ``model`` 别名，而我们全程只读 ``success`` / ``response``。
        """
        timeout_ms = int(kwargs.pop("rpc_timeout_ms", LLM_RPC_TIMEOUT_MS))
        call_capability = getattr(self.ctx, "call_capability", None)
        if callable(call_capability):
            return await call_capability("llm.generate", timeout_ms=timeout_ms, **kwargs)
        # 老 SDK 没有这个方法：退回原封装（会吃 30s 默认超时，但至少还能跑）
        return await self.ctx.llm.generate(**kwargs)

    async def _embed_texts(self, texts: "list[str]") -> "list[list[float]] | None":
        """适配 Host 的 llm.embed，兼容单条与批量两种返回形态。

        注意 ``task_name`` 与 ``model_name`` 是不同参数，且**只能非空才传**——传空串会
        让 Host 去找一个不存在的模型并直接失败。

        批量形态解析不出来时会**逐条重试**：有些实现只吃单条 ``text=``，若直接判定
        "向量层不可用"，用户会看到整个向量检索静默降级，却不知道为什么。逐条重试的
        调用次数由 ``_prepare_vectors`` 的分块大小兜住，是有界的。
        """
        cfg = self.config.retrieval
        if not cfg.embedding_enabled:
            return None
        if time.monotonic() < self._embed_unavailable_until:
            return None
        kwargs: dict[str, Any] = {}
        if cfg.embed_task_name:
            kwargs["task_name"] = cfg.embed_task_name
        if cfg.embed_model_name:
            kwargs["model_name"] = cfg.embed_model_name

        try:
            result = await asyncio.wait_for(
                self.ctx.llm.embed(texts=list(texts), **kwargs), timeout=cfg.embed_timeout_seconds
            )
        except Exception as exc:
            self._mark_embed_unavailable(f"调用失败：{type(exc).__name__}: {exc}")
            return None
        vectors = self._extract_vectors(result, expected=len(texts))
        if vectors is not None:
            return vectors

        if len(texts) > 1:
            vectors = await self._embed_individually(texts, kwargs, cfg.embed_timeout_seconds)
            if vectors is not None:
                if not self._embed_single_fallback_logged:
                    self._embed_single_fallback_logged = True
                    self.ctx.logger.info(
                        "Host 的批量嵌入返回结构不可用，已改用逐条嵌入（结果一样，首次建索引会慢一些）"
                    )
                return vectors

        self._mark_embed_unavailable("返回结构无法解析（embedding/results 都没有向量）")
        return None

    async def _embed_individually(
        self, texts: "list[str]", kwargs: dict[str, Any], timeout_seconds: float
    ) -> "list[list[float]] | None":
        """逐条取向量。任一条失败就整体放弃（保持"要么全有要么全无"的一致性）。"""
        vectors: list[list[float]] = []
        for text in texts:
            try:
                result = await asyncio.wait_for(
                    self.ctx.llm.embed(text=text, **kwargs), timeout=timeout_seconds
                )
            except Exception:
                return None
            extracted = self._extract_vectors(result, expected=1)
            if not extracted:
                return None
            vectors.append(extracted[0])
        return vectors

    def _mark_embed_unavailable(self, reason: str) -> None:
        cooldown = float(self.config.retrieval.embed_cooldown_seconds or 0.0)
        self._embed_unavailable_until = time.monotonic() + cooldown
        self.ctx.logger.warning(
            "向量检索不可用（%s）；已降级为关键词检索，%.0f 秒后重试。"
            "若 MaiBot 未配置 embedding 任务，可忽略此警告或在配置里关闭向量检索。",
            reason, cooldown,
        )

    @staticmethod
    def _extract_vectors(result: Any, *, expected: int) -> "list[list[float]] | None":
        """从 Host 返回值里抠出向量列表，兼容单条 ``embedding`` 与批量 ``results``。"""
        if not isinstance(result, dict):
            return None
        single = result.get("embedding")
        if expected == 1 and isinstance(single, list) and single and isinstance(single[0], (int, float)):
            return [[float(value) for value in single]]
        for key in ("results", "embeddings", "data", "vectors"):
            values = result.get(key)
            if not isinstance(values, list) or len(values) != expected:
                continue
            vectors: list[list[float]] = []
            for item in values:
                vector = item.get("embedding") if isinstance(item, dict) else item
                if not isinstance(vector, list) or not vector:
                    return None
                vectors.append([float(value) for value in vector])
            return vectors
        if expected == 1 and isinstance(single, list) and single and isinstance(single[0], list):
            return [[float(value) for value in single[0]]]
        return None

    # ---------------------------------------------------------- 反查源

    def _source_configs(self) -> "dict[str, SourceConfig]":
        plugin = self.config.plugin
        return {
            "anime_trace": SourceConfig(
                name="anime_trace",
                enabled=self.config.anime_trace.enabled,
                url=self.config.anime_trace.url,
                timeout_seconds=self.config.anime_trace.timeout_seconds,
                max_upload_bytes=self.config.anime_trace.max_upload_bytes,
                max_candidates=self.config.anime_trace.max_candidates,
            ),
            "saucenao": SourceConfig(
                name="saucenao",
                enabled=self.config.saucenao.enabled,
                url=self.config.saucenao.url,
                timeout_seconds=self.config.saucenao.timeout_seconds,
                max_upload_bytes=self.config.saucenao.max_upload_bytes,
                api_key=self.config.saucenao.api_key,
                max_candidates=self.config.saucenao.max_candidates,
                confident_similarity=self.config.saucenao.confident_similarity,
                weak_similarity=self.config.saucenao.weak_similarity,
            ),
            "trace_moe": SourceConfig(
                name="trace_moe",
                enabled=self.config.trace_moe.enabled,
                url=self.config.trace_moe.url,
                timeout_seconds=self.config.trace_moe.timeout_seconds,
                max_upload_bytes=self.config.trace_moe.max_upload_bytes,
                max_candidates=self.config.trace_moe.max_candidates,
            ),
        }

    async def _collect_hits(self, image_bytes: bytes) -> "tuple[tuple[SourceHit, ...], list[str]]":
        """并发跑所有已启用的源，返回 (命中, 错误说明列表)。"""
        configs = [cfg for cfg in self._source_configs().values() if cfg.enabled]
        if not configs:
            return (), []
        results = await asyncio.gather(
            *(run_source(cfg, image_bytes, breaker=self._breaker) for cfg in configs),
            return_exceptions=True,
        )
        hits: list[SourceHit] = []
        errors: list[str] = []
        for config, outcome in zip(configs, results):
            if isinstance(outcome, BaseException):
                errors.append(f"{config.name}: {outcome}")
                continue
            group, error = outcome
            hits.extend(group)
            if error:
                errors.append(f"{config.name}: {error}")
            elif not config.enabled:
                continue
        return tuple(hits), errors

    # ---------------------------------------------------------- 视觉

    async def _describe(self, image_bytes: bytes) -> str:
        cfg = self.config.vision
        if not cfg.enabled:
            return ""
        if cfg.provider != "host":
            self._log("warning", "图片描述目前只走 host 通道，直连通道请用识别路径")
            return ""
        try:
            return await describe_image(
                self._generate,
                image_bytes=image_bytes,
                task_name=cfg.task_name,
                model_name=cfg.model_name,
                timeout_seconds=self.config.plugin.image_timeout_seconds,
                max_upload_bytes=cfg.max_upload_bytes,
            )
        except VisionError as exc:
            hint = self._vision_timeout_hint() if "超时" in str(exc) else ""
            self._log("warning", f"图片描述失败：{exc}{hint}")
            return ""

    async def _identify_candidates(
        self, image_bytes: bytes, catalog: "list[dict]"
    ) -> "tuple[tuple[str, ...], str, str]":
        """用带本地库目录的提示词做一次候选校验。

        返回 ``(候选名, 说明, 模型原始输出)``。后两项是给排障用的——**"没有可用候选"
        下面藏着四种完全不同的原因**（模型没给候选 / 判定不在库内 / 带了冲突特征 /
        证据不足），处理方式各不相同。只报一句"未确认"，用户会去改没错的那一环。
        """
        cfg = self.config.vision
        if not cfg.enabled:
            return (), "视觉通道已关闭", ""
        if not catalog:
            return (), "没有可校验的本地候选", ""
        try:
            result = await identify_image(
                provider=cfg.provider,
                image_bytes=image_bytes,
                catalog=catalog,
                max_candidates=self.config.plugin.max_characters_per_image,
                generate=self._generate,
                task_name=cfg.task_name,
                model_name=cfg.model_name,
                api_key=cfg.api_key,
                base_url=cfg.base_url,
                timeout_seconds=self.config.plugin.image_timeout_seconds,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                max_upload_bytes=cfg.max_upload_bytes,
            )
        except VisionError as exc:
            hint = self._vision_timeout_hint() if "超时" in str(exc) else ""
            self._log("warning", f"候选校验失败：{exc}{hint}")
            return (), f"调用失败：{exc}", ""
        if result is None:
            return (), "模型输出不是可解析的 JSON", ""
        # 模型的格式自觉不可靠（真机出现过"证据逐字抄了库内画像，却把 kind 标成 unknown"），
        # 所以再用确定性比对兜一次。
        result, rescue_note = rescue_unlabeled_candidates(result, catalog)
        names = tuple(result.candidate_names)
        if names:
            note = "确认了候选" + (f"；{rescue_note}" if rescue_note else "")
        else:
            note = f"无可用候选（{describe_vision_dropouts(result)}）"
        return names, note, result.raw

    # ---------------------------------------------------------- 识别主流程

    def _retrieval_options(self) -> RetrievalOptions:
        cfg = self.config.retrieval
        return RetrievalOptions(
            max_prompt_characters=self.config.library.max_prompt_characters,
            pinned_reserve=self.config.library.pinned_reserve,
            embedding_enabled=cfg.embedding_enabled,
            embed_batch_size=cfg.embed_batch_size,
            weight_embedding=cfg.weight_embedding,
            weight_keyword=cfg.weight_keyword,
            boost_reverse_confirmed=cfg.boost_reverse_confirmed,
            keyword_min_score=cfg.keyword_min_score,
            embed_min_score=cfg.embed_min_score,
        )

    def _fusion_options(self) -> FusionOptions:
        cfg = self.config.fusion
        return FusionOptions(
            auto_apply_single_source=cfg.auto_apply_single_source,
            require_two_sources_for_auto=cfg.require_two_sources_for_auto,
            conflict_min_confidence=cfg.conflict_min_confidence,
            auto_apply_local_confirm=cfg.auto_apply_local_confirm,
            saucenao_confident_similarity=self.config.saucenao.confident_similarity,
            saucenao_weak_similarity=self.config.saucenao.weak_similarity,
        )

    def _resolver(self):
        """名字 → 本地角色的解析函数，交给融合层做跨语言桥接。"""
        repository = self._repository

        def resolve(raw_name: str):
            if repository is None:
                return None
            character = repository.resolve(raw_name)
            if character is None:
                return None
            return character.character_id, character.name

        return resolve

    async def _recognize(self, image_bytes: bytes, *, chat_text: str = "") -> RecognitionResult:
        """单张图片的完整识别流程。"""
        if not image_bytes:
            return RecognitionResult(description="图片未能读取", ok=False)
        digest = sha256_hex(image_bytes)
        cache_key = f"{self._generation}:{digest}"
        diag: dict[str, Any] = {"cache": "未命中", "threshold": self.config.fusion.local_confirm_min_score}
        if self.config.cache.enabled:
            cached = self._cache.get(cache_key)
            if cached is not None:
                diag.update(
                    cache="命中，复用上次结论（不重跑链路）",
                    tier=cached.fusion.tier,
                    label=cached.label,
                    reason=cached.fusion.reason,
                )
                self._last_diagnosis = diag
                return cached

        # 真值判断会踩坑：空库的 bool() 是 False，语义上会被当成"没有库"。
        characters = tuple(self._repository.characters) if self._repository is not None else ()

        # 阶段一：反查源与图片描述并行，互不依赖
        (hits, errors), description = await asyncio.gather(
            self._collect_hits(image_bytes),
            self._describe(image_bytes),
        )
        diag["characters"] = len(characters)
        diag["reverse"] = f"命中 {len(hits)} 条" if hits else "无命中"
        diag["reverse_errors"] = list(errors)
        diag["description"] = bool(description)
        if errors:
            self._log("warning", "部分反查源失败：%s", "；".join(errors))
        if self.config.plugin.debug and hits:
            self.ctx.logger.debug("反查命中：\n%s", describe_hits(hits))

        # 阶段二：相关性检索挑选候选
        relationships: dict[str, str] = {}
        retrieval = None
        if characters and self.config.library.enabled:
            retrieval = await retrieve(
                characters,
                Query(
                    description=description,
                    reverse_names=tuple(hit.raw_name for hit in hits if hit.gives_character),
                    reverse_works=tuple(hit.work for hit in hits if hit.work),
                    chat_text=chat_text,
                ),
                options=self._retrieval_options(),
                embed=self._embed_texts,
                index=self._index,
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
        if retrieval is not None and retrieval.selected and self.config.vision.enabled:
            preliminary = fuse(hits, resolve=self._resolver(), options=self._fusion_options())
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
                vision_names, vision_note, vision_raw = await self._identify_candidates(image_bytes, catalog)
            else:
                vision_note = "未调用（反查源已确认，不需要校验）"
        elif retrieval is not None:
            vision_note = "未调用（检索没有选出候选）"
        diag["vision"] = vision_note
        diag["vision_names"] = list(vision_names)
        diag["vision_raw"] = vision_raw

        local_confirmed = self._local_confirmed(vision_names, retrieval)
        fusion = fuse(
            hits,
            resolve=self._resolver(),
            options=self._fusion_options(),
            vlm_names=vision_names,
            degraded=bool(retrieval and retrieval.degraded),
            local_confirmed=local_confirmed,
        )
        label = fusion.label(relationships, limit=self.config.plugin.max_characters_per_image)
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
        self._last_diagnosis = diag
        injection = ""
        if self.config.injection.enabled and (retrieval is not None or fusion.works):
            injection = build_knowledge_block(
                # 注意取 .characters（Character 元组）而不是 .selected（ScoredCharacter 元组），
                # 后者没有 name/work 这些字段，传错会得到空知识块且不报错。
                characters=retrieval.characters if retrieval else (),
                fusion=fusion,
                options=self.config.injection,
                max_chars=self.config.library.knowledge_block_max_chars,
            )

        result = RecognitionResult(
            description=description or "",
            label=label,
            injection=injection,
            image_hash=digest,
            fusion=fusion,
            ok=True,
        )
        if self.config.cache.enabled:
            ttl = self.config.cache.ttl_seconds if label != UNRECOGNIZED_LABEL else self.config.cache.negative_ttl_seconds
            if ttl > 0:
                self._cache.put(cache_key, result, ttl_seconds=ttl)
        return result

    def _local_confirmed(
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
        resolve = self._resolver()
        scored = {item.character.character_id: item for item in retrieval.selected}
        threshold = self.config.fusion.local_confirm_min_score
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

    async def _recognize_limited(self, image_bytes: bytes, generation: int, chat_text: str) -> "RecognitionResult | None":
        async with self._semaphore:
            result = await self._recognize(image_bytes, chat_text=chat_text)
        if generation != self._generation:
            # 配置在识别期间被改过，结论已经不能代表当前配置。
            return None
        return result

    def _track_task(self, task: asyncio.Task) -> None:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # ---------------------------------------------------------- 消息处理

    @staticmethod
    def _is_management_command(text: str) -> bool:
        """管理命令自身不参与识图，否则命令的回执会被识别结果挤掉。"""
        head = str(text or "").strip()
        for prefix in ("/", "／"):
            if head.startswith(prefix):
                head = head[len(prefix):].strip()
                break
        else:
            return False
        return head.split(" ", 1)[0].strip() in {
            "角色添加", "取消角色添加", "结束角色添加", "完成角色添加", "角色添加完成",
            "识图修正", "角色列表", "角色库", "查看角色", "查看人设",
            "设置人设", "设置作品", "设置关系", "添加别名", "删除别名", "删除角色", "重建索引",
            "识别状态", "识别探测",
        }

    async def _resolve_image(self, component: dict) -> "tuple[bytes | None, str]":
        """取图片数据：先 base64，再退回 URL 下载。"""
        data, note = extract_image_payload(component)
        if data:
            return data, note
        url = str(component.get("url") or "").strip()
        if url.startswith(("http://", "https://")):
            # 这里先判一次公网，只为**把拒绝原因说清楚**：真正的拦截在 download_image 里
            # 逐跳做。判两次的代价是多一次 DNS，换来日志能分清"被 SSRF 防护拦下"和
            # "网络失败"——两者对用户的含义完全不同。
            if not await asyncio.to_thread(host_resolves_to_public, urlsplit(url).hostname or ""):
                return None, f"URL 指向非公网地址，已按 SSRF 防护跳过：{url[:80]}"
            downloaded = await asyncio.to_thread(download_image, url, timeout_seconds=15.0)
            if downloaded:
                return downloaded, "url 下载"
        return None, note

    async def _message_components(self, message_id: str) -> "list[dict]":
        """取引用消息的组件列表（不同版本/适配器的字段名不一致，逐个试）。"""
        if not message_id:
            return []
        try:
            payload = await self.ctx.message.get_by_id(message_id)
        except Exception as exc:
            self._log("warning", f"读取引用消息失败：{exc}")
            return []
        if not isinstance(payload, dict):
            return []
        for key in ("raw_message", "message", "segments", "content"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict) and isinstance(value.get("raw_message"), list):
                return [item for item in value["raw_message"] if isinstance(item, dict)]
        return []

    async def _message_images(self, message_id: str, limit: int = 4) -> "list[bytes]":
        """按消息 ID 取引用消息里的图片，最多 ``limit`` 张。

        带 ``limit`` 是为了让只要一张的调用方（工具）不必把整条消息的图全下载下来——
        ``_message_image`` 就是 ``limit=1`` 的用法。
        """
        images: "list[bytes]" = []
        for component in await self._message_components(message_id):
            if len(images) >= max(1, limit):
                break
            if component.get("type") != "image":
                continue
            data, _ = await self._resolve_image(component)
            if data:
                images.append(data)
        return images

    async def _message_image(self, message_id: str) -> bytes | None:
        """按消息 ID 取引用消息里的**第一张**图（供返回单个识别结论的工具用）。"""
        images = await self._message_images(message_id, limit=1)
        return images[0] if images else None

    def _pending_key(self, payload: dict[str, Any]) -> str:
        """待处理项与"最近图片"的归属键：``会话:用户``。

        命令载荷（``cmd_*``）与入站消息载荷（hook）结构不同：前者把触发者放在顶层
        ``user_id``，后者埋在 ``message_info.user_info`` 里。**两条路径必须算出同一个键**，
        否则 ``/角色添加`` 登记的待处理项永远等不到那张图。
        """
        message = payload.get("message") if isinstance(payload.get("message"), dict) else payload
        session = str(message.get("session_id") or message.get("stream_id") or "")
        info = message.get("message_info") if isinstance(message, dict) else None
        user = ""
        if isinstance(info, dict):
            user_info = info.get("user_info")
            if isinstance(user_info, dict):
                user = str(user_info.get("user_id") or "")
        if not user:
            user = str(message.get("user_id") or message.get("sender_id") or "")
        return f"{session}:{user}"

    def _remember_image(self, key: str, image_bytes: bytes, label: str) -> None:
        """记住这个会话刚识别过的图片，**保留一小段历史**。

        为什么不是只留最后一张：``/识图修正`` 不引用消息时用的是"刚发的那几张"，
        只留一张就永远只能补一张卡——那跟要求引用没区别。每条带时间戳，读的时候按
        时间窗过滤（见 ``_recent_images``）。

        三重上限：单会话张数、会话数、总字节数。字节数是必须的——一张图动辄几 MB，
        不限就是几百 MB 常驻内存。dict 保序，所以"最早插入的"就是最该淘汰的。
        """
        buffer = self._latest_images.setdefault(key, [])
        buffer.append((image_bytes, label, time.monotonic()))
        self._latest_bytes += len(image_bytes)
        while len(buffer) > RECENT_IMAGE_LIMIT:
            data, _, _ = buffer.pop(0)
            self._latest_bytes -= len(data)
        while len(self._latest_images) > 1 and (
            len(self._latest_images) > RECENT_IMAGE_SESSIONS
            or self._latest_bytes > self._latest_max_bytes
        ):
            oldest = next(iter(self._latest_images))
            if oldest == key:  # 只剩当前会话时不再淘汰，否则刚存进来的当场被扔
                break
            dropped = self._latest_images.pop(oldest)
            self._latest_bytes -= sum(len(item[0]) for item in dropped)
        self._latest_bytes = max(0, self._latest_bytes)

    def _recent_images(
        self, key: str, limit: int = 1, window: float = RECENT_IMAGE_SECONDS
    ) -> "list[bytes]":
        """取该会话最近 ``window`` 秒内的图片，最多 ``limit`` 张（新的在后）。

        时间窗是必要的：没有它，十分钟前发的图会在下一次 ``/识图修正`` 里被当成
        "刚发的"用上，而用户完全看不出这些卡是从哪张图来的。
        """
        now = time.monotonic()
        fresh = [item for item in self._latest_images.get(key, ()) if now - item[2] <= window]
        return [item[0] for item in fresh[-max(1, limit):]]

    def _latest_in_session(self, kwargs: dict, limit: int = 1) -> "list[bytes]":
        """按会话取最近发过的图（新→旧），最多 ``limit`` 张。

        工具的载荷里不一定带得出精确的 ``user_id``，所以这里按 ``stream_id:`` 前缀扫，
        而不是用 ``_pending_key`` 拼精确键。
        """
        prefix = f"{self._stream_id(kwargs)}:"
        images: "list[bytes]" = []
        for key in reversed(list(self._latest_images)):
            if not key.startswith(prefix):
                continue
            images.extend(reversed(self._recent_images(key, limit=limit)))
            if len(images) >= limit:
                break
        return images[:limit]

    async def _build_replacements(
        self, message: dict, images: "list[dict]", chat_text: str
    ) -> "list[list[dict]]":
        """并发识别图片，返回与**图片**一一对应的替换片段。

        入参必须是过滤后的图片列表，不能是完整的 ``raw_message``：替换片段是按图片顺序
        消费的，若按全部组件建表，前面插一个文字组件就会让第一张图错位拿到文字组件的
        结果——表现为图片被替换掉、甚至整张图消失。
        """
        limit = self.config.plugin.max_images_per_message
        generation = self._generation
        tasks: list[asyncio.Task] = []
        for component in images[:limit]:
            image_bytes, note = await self._resolve_image(component)
            if image_bytes is None:
                self._log("warning", "图片载荷不可用：%s", note)
                tasks.append(asyncio.create_task(self._noop_result()))
                continue
            tasks.append(asyncio.create_task(self._recognize_limited(image_bytes, generation, chat_text)))

        done, pending = await asyncio.wait(tasks, timeout=self.config.plugin.message_timeout_seconds)
        for task in pending:
            self._log("warning", "单条消息识别超预算，未完成的图片将原样交给 MaiBot")
            self._track_task(task)

        key = self._pending_key(message)
        replacements: "list[list[dict]]" = []
        for index, component in enumerate(images):
            if index >= limit:
                replacements.append([component])
                continue
            task = tasks[index]
            if task not in done:
                replacements.append([component])
                continue
            try:
                result = task.result()
            except Exception as exc:
                self._log("warning", "单张图片识别失败，保留原图：%s", exc)
                replacements.append([component])
                continue
            if result is None or not result.ok:
                replacements.append([component])
                continue
            image_bytes, _ = await self._resolve_image(component)
            if image_bytes:
                self._remember_image(key, image_bytes, result.label)
            if result.description:
                replacements.append([{"type": "text", "data": f"[图片：{result.description}] {result.label}"}])
            elif result.label != UNRECOGNIZED_LABEL:
                replacements.append([component, {"type": "text", "data": f"[角色识别：{result.label}]"}])
            else:
                replacements.append([component])
        return replacements

    @staticmethod
    async def _noop_result() -> "RecognitionResult | None":
        return None

    def _injection_for(self, session_id: str) -> str:
        """按会话取"最近一次识别到的东西"来构造注入内容。

        注入发生在模型请求前，那时已经没有消息载荷可用了，所以依赖识别阶段留下的
        结果缓存：把同一会话最近识别出的角色资料带去。
        """
        if not self.config.injection.enabled or self._repository is None:
            return ""
        marker = INJECT_MARKER
        hint = build_capability_hint(TOOL_NAMES, marker=marker)
        recent: "list[str]" = []
        for key, items in self._latest_images.items():
            if key.startswith(f"{session_id}:"):
                recent.extend(item[1] for item in items)
        if not recent:
            return hint if self.config.injection.inject_into_planner else ""
        characters: list[Character] = []
        seen: set[str] = set()
        for label in recent[-3:]:
            for name in _names_from_label(label):
                character = self._repository.find_name(name)
                if character is not None and character.character_id not in seen:
                    seen.add(character.character_id)
                    characters.append(character)
        if not characters:
            return hint if self.config.injection.inject_into_planner else ""
        block = build_knowledge_block(
            characters=characters,
            options=self.config.injection,
            max_chars=self.config.library.knowledge_block_max_chars,
        )
        return f"{block}\n{hint}" if hint else block

    # ---------------------------------------------------------- Hook 组件

    @HookHandler(
        "chat.receive.before_process",
        name="recognize_incoming_images",
        description="在消息进入 MaiBot 前识别图片角色，把图片替换为描述与角色标签。",
        mode=HookMode.BLOCKING,
        order=HookOrder.EARLY,
        timeout_ms=120000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def recognize_incoming_images(self, message: Any = None, **kwargs: Any) -> "dict[str, Any] | None":
        if not self.config.plugin.enabled:
            return None
        if not isinstance(message, dict):
            return None
        if not self._hook_payload_logged:
            # 不同版本/适配器给的消息字段不完全一致，首次触发留一条字段清单便于排障。
            self._hook_payload_logged = True
            self.ctx.logger.info("入站消息字段：%s", sorted(message))
        components = message.get("raw_message")
        if not isinstance(components, list):
            return None
        images = [
            item for item in components
            if isinstance(item, dict) and item.get("type") == "image"
        ]
        if not images:
            return None

        key = self._pending_key(message)
        pending = self._pending_additions.get(key)
        if pending is not None:
            if pending.expired(time.monotonic()):
                self._pending_additions.pop(key, None)
                stream_id = self._stream_id({"message": message})
                await self._reply(stream_id, "角色登记已超时结束，这批图片按普通图片处理。")
            else:
                # 持续收集：这批图全部用于建卡/补卡，不收手就一直收。
                await self._collect_character_images(message, images, pending)
                return None

        if self._is_management_command(str(message.get("processed_plain_text") or "")):
            return None

        replacements = await self._build_replacements(message, images, str(message.get("processed_plain_text") or ""))
        modified = apply_rewrite(message, replacements)
        if modified is None:
            return None
        return {"action": "continue", "modified_kwargs": {"message": modified}}

    @HookHandler(
        "maisaka.replyer.before_model_request",
        name="inject_character_context",
        description="把最近识别到的角色资料注入回复模型请求。",
        mode=HookMode.BLOCKING,
        order=HookOrder.EARLY,
        timeout_ms=3000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def inject_character_context(self, **kwargs: Any) -> "dict[str, Any] | None":
        if not self.config.injection.enabled or not self.config.injection.inject_into_replyer:
            return None
        if not self._logged_injection_keys:
            self._logged_injection_keys = True
            self.ctx.logger.info("模型请求注入载荷字段：%s", sorted(kwargs))
        session_id = self._stream_id(kwargs)
        text = self._injection_for(session_id)
        if not text:
            return None
        payload = dict(kwargs)
        status = apply_injection(payload, text, INJECT_MARKER)
        if self.config.plugin.debug:
            self.ctx.logger.debug("注入结果：%s", status)
        if "已注入" not in status:
            return None
        return {"action": "continue", "modified_kwargs": payload}

    @HookHandler(
        "maisaka.planner.before_request",
        name="inject_character_context_planner",
        description="把角色资料与可用工具提示注入 Planner 请求。",
        mode=HookMode.BLOCKING,
        order=HookOrder.EARLY,
        timeout_ms=3000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def inject_character_context_planner(self, **kwargs: Any) -> "dict[str, Any] | None":
        if not self.config.injection.enabled or not self.config.injection.inject_into_planner:
            return None
        session_id = self._stream_id(kwargs)
        text = self._injection_for(session_id)
        if not text:
            return None
        payload = dict(kwargs)
        status = apply_injection(payload, text, INJECT_MARKER)
        if "已注入" not in status:
            return None
        return {"action": "continue", "modified_kwargs": payload}

    # ---------------------------------------------------------- Tool 组件

    @Tool(
        "recognize_image",
        description=(
            "识别聊天里的一张图片里的二次元角色，返回角色名、所属作品与反查出处。"
            "当用户问「这张图里是谁」「这个角色出自哪」，或你想确认某个角色身份时调用。"
            "参数 image_ref 填被引用消息的 ID；留空则使用当前会话最近发过的图片。"
        ),
        parameters=[
            ToolParameterInfo(
                name="image_ref",
                param_type=ToolParamType.STRING,
                description="要识别的消息 ID（通常是被引用消息的 ID），留空表示用最近一张图",
                required=False,
            ),
        ],
    )
    async def recognize_image(self, image_ref: str = "", **kwargs: Any) -> "dict[str, str]":
        if not self.config.plugin.enabled:
            return {"content": "角色识别功能未启用"}
        image_bytes = await self._message_image(image_ref)
        if image_bytes is None:
            recent = self._latest_in_session(kwargs, limit=1)
            if not recent:
                return {"content": "没有找到可识别的图片。请引用一张带图的消息，或先让对方发一张。"}
            image_bytes = recent[0]
        try:
            result = await self._recognize(image_bytes, chat_text="")
        except Exception as exc:
            self.ctx.logger.error("识图失败：%s", exc, exc_info=True)
            return {"content": "识别失败，稍后再试。"}
        return {"content": _describe_result(result)}

    @Tool(
        "query_character",
        description=(
            "按名字或别名查询本机角色库里的角色资料（设定、作品、关系、别名、外观）。"
            "当用户提到某个角色名、或你已经知道角色名需要她的设定时调用。"
            "参数 name：角色名或别名。参数 field 指定只看某一项，默认 auto 返回全部。"
        ),
        parameters=[
            ToolParameterInfo(
                name="name", param_type=ToolParamType.STRING, description="角色名或别名", required=True
            ),
            ToolParameterInfo(
                name="field",
                param_type=ToolParamType.STRING,
                description="要查看的字段：auto / persona / work / aliases / appearance / relationship",
                required=False,
                enum_values=["auto", "persona", "work", "aliases", "appearance", "relationship"],
            ),
        ],
    )
    async def query_character(self, name: str = "", field: str = "auto", **kwargs: Any) -> "dict[str, str]":
        del kwargs
        if self._repository is None or not name.strip():
            return {"content": "请提供要查询的角色名。"}
        character = self._repository.find_name(name)
        if character is None:
            candidates = self._repository.search(name, limit=5)
            if candidates:
                names = "、".join(item.name for item in candidates)
                return {"content": f"角色库里没有「{name.strip()}」。相近的有：{names}"}
            return {"content": f"角色库里没有「{name.strip()}」。可以让管理员用 /角色添加 收录。"}
        return {"content": _render_character(character, field)}

    @Tool(
        "search_character_library",
        description=(
            "按外貌或特征描述检索本机角色库，返回最相近的几个角色。"
            "适合用户描述得含糊（比如「白发戴光环的那个」）而你不知道确切名字时使用。"
            "参数 query：特征描述。参数 limit：返回条数上限，0 表示用默认值。"
        ),
        parameters=[
            ToolParameterInfo(
                name="query", param_type=ToolParamType.STRING, description="外貌或特征描述", required=True
            ),
            ToolParameterInfo(
                name="limit", param_type=ToolParamType.INTEGER, description="返回条数上限", required=False
            ),
        ],
    )
    async def search_character_library(self, query: str = "", limit: int = 0, **kwargs: Any) -> "dict[str, str]":
        del kwargs
        if self._repository is None or not query.strip():
            return {"content": "请提供要检索的特征描述。"}
        characters = tuple(self._repository.characters)
        if not characters:
            return {"content": "本机角色库还是空的，暂时无法检索。"}
        size = limit if isinstance(limit, int) and limit > 0 else 5
        options = self._retrieval_options()
        options = RetrievalOptions(
            max_prompt_characters=size,
            pinned_reserve=0,
            embedding_enabled=options.embedding_enabled,
            embed_batch_size=options.embed_batch_size,
            weight_embedding=options.weight_embedding,
            weight_keyword=options.weight_keyword,
            boost_reverse_confirmed=options.boost_reverse_confirmed,
            keyword_min_score=options.keyword_min_score,
            embed_min_score=options.embed_min_score,
        )
        result = await retrieve(
            characters,
            Query(description=query),
            options=options,
            embed=self._embed_texts,
            index=self._index,
        )
        if not result.selected:
            return {"content": f"角色库里没有与「{query.strip()}」明显相关的角色。"}
        lines = [
            f"{index}. {item.character.name}"
            + (f"（{item.character.work}）" if item.character.work else "")
            + (f"｜{'、'.join(item.reasons)}" if item.reasons else "")
            for index, item in enumerate(result.selected, start=1)
        ]
        suffix = "（向量检索不可用，已降级为关键词匹配）" if result.degraded else ""
        return {"content": f"最相近的角色：\n" + "\n".join(lines) + suffix}

    @Tool(
        "reverse_lookup_image",
        description=(
            "对聊天里的图片做联网反查，找出它的出处与可能的角色（不依赖本机角色库）。"
            "当用户想知道图片出自哪部作品、画师是谁，或本机角色库查不到时调用。"
            "参数 image_ref：消息 ID，留空表示用当前会话最近一张图。"
        ),
        parameters=[
            ToolParameterInfo(
                name="image_ref",
                param_type=ToolParamType.STRING,
                description="要反查的消息 ID，留空表示用最近一张图",
                required=False,
            ),
        ],
    )
    async def reverse_lookup_image(self, image_ref: str = "", **kwargs: Any) -> "dict[str, str]":
        image_bytes = await self._message_image(image_ref)
        if image_bytes is None:
            recent = self._latest_in_session(kwargs, limit=1)
            if not recent:
                return {"content": "没有找到可反查的图片。"}
            image_bytes = recent[0]
        hits, errors = await self._collect_hits(image_bytes)
        if not hits:
            detail = f"（{'；'.join(errors)}）" if errors else ""
            return {"content": f"没有反查到出处{detail}"}
        lines = []
        for hit in hits:
            parts = [f"{hit.source}：{hit.raw_name or '未给出角色名'}"]
            if hit.work:
                parts.append(f"作品 {hit.work}")
            parts.append("置信" if hit.confident else "低置信")
            lines.append("，".join(parts))
        return {"content": "反查结果：\n" + "\n".join(lines)}

    # ---------------------------------------------------------- Command 组件

    @Command("status", description="查看角色识别插件状态", pattern=r"^\s*[/／]\s*(?:识别状态|状态|status)\s*$")
    async def cmd_status(self, **kwargs: Any) -> "tuple[bool, str, int]":
        stream_id = self._stream_id(kwargs)
        if not self._is_admin(kwargs):
            reply = self._admin_hint()
            return True, reply, 2 if await self._reply(stream_id, reply) else 0
        lines = ["【角色识别】运行状态", _status_lines(
            self.config,
            self._repository,
            self._source_configs(),
            len(self._cache),
            len(self._index),
            sum(len(items) for items in self._latest_images.values()),
            self._embed_available(),
        )]
        models = await self._available_models()
        lines.append(f"可用模型任务：{models}")
        reply = "\n".join(lines)
        return True, reply, 2 if await self._reply(stream_id, reply) else 0

    @Command(
        "probe",
        description="用真实图片探测各反查源与向量是否可用",
        pattern=r"^\s*[/／]\s*(?:识别探测|探测|probe)\s*$",
    )
    async def cmd_probe(self, **kwargs: Any) -> "tuple[bool, str, int]":
        stream_id = self._stream_id(kwargs)
        if not self._is_admin(kwargs):
            reply = self._admin_hint()
            return True, reply, 2 if await self._reply(stream_id, reply) else 0
        recent = self._recent_images(self._pending_key(kwargs), limit=1)
        image_bytes = recent[0] if recent else None
        if image_bytes is None:
            reply = "【角色识别】请先在本会话发一张图，再用本命令探测。"
            return True, reply, 2 if await self._reply(stream_id, reply) else 0
        hits, errors = await self._collect_hits(image_bytes)
        enabled = [cfg.name for cfg in self._source_configs().values() if cfg.enabled]
        lines = ["【角色识别】探测结果", f"已接入的源：{'、'.join(IMPLEMENTED_SOURCES) or '无'}"]
        if hits:
            lines.append("命中角色：")
            for hit in hits:
                lines.append(f"  · [{hit.source}] {hit.raw_name}"
                             + (f"｜{hit.work}" if hit.work else "")
                             + ("（低置信）" if not hit.confident else ""))
            known = [
                hit.raw_name for hit in hits
                if self._repository is not None and self._repository.find_name(hit.raw_name)
            ]
            lines.append(f"其中能命中本地库别名：{len(known)}/{len(hits)}"
                         + (f"（{'、'.join(known)}）" if known else ""))
            if not known:
                lines.append("  一个都对不上本地库：库为空，或源用的写法没进别名。"
                             "用 /添加别名 把反查源常用的写法补进去")
        elif not enabled:
            lines.append("没有启用任何反查源：开启 anime_trace 或 saucenao 后重跑本命令")
        else:
            lines.append(f"无命中：{'、'.join(enabled)} 都正常查了，没有源报出角色")
        if errors:
            # 源故障必须单独成段。混进"无命中"里，用户会以为图里没角色，
            # 于是反复重发同一张图——对限流来说这是最糟的应对。
            lines.append("源故障（不是图里没角色，是这一次没查成）：")
            lines.extend(f"  · {_render_source_error(item)}" for item in errors)
        for name in ("saucenao", "trace_moe"):
            config = self._source_configs()[name]
            if not config.enabled:
                lines.append(f"{name}：未启用（如需评估，开启后重跑本命令）")
                continue
            summary = await probe_endpoint(config, image_bytes)
            lines.append(f"{name}：{_render_probe(summary)}")
        # 反查源状态只说明"外部源行不行"，不能解释"为什么这张图没认出来"——
        # 后者要看识别链路自己的岔路口。
        lines.append("最近一次识图诊断：")
        lines.append(_render_diagnosis(self._last_diagnosis))
        reply = "\n".join(lines)
        return True, reply, 2 if await self._reply(stream_id, reply) else 0

    @Command(
        "character_add",
        description="登记一个新角色，下一条图片会成为她的外观卡",
        pattern=(
            r"^\s*[/／]\s*(?:角色添加|添加角色)\s+(?P<name>\S+)"
            r"(?:\s+(?P<work>\S+))?"
            r"(?:\s+(?P<relationship>\S+))?"
            r"(?:\s+\[图片[：:].*)?\s*$"
        ),
    )
    async def cmd_character_add(self, **kwargs: Any) -> "tuple[bool, str, int]":
        stream_id = self._stream_id(kwargs)
        if not self._is_admin(kwargs):
            reply = self._admin_hint()
            return True, reply, 2 if await self._reply(stream_id, reply) else 0
        groups = kwargs.get("matched_groups") if isinstance(kwargs.get("matched_groups"), dict) else {}
        name = str(groups.get("name") or kwargs.get("name") or "").strip()
        work = str(groups.get("work") or kwargs.get("work") or "").strip()
        relationship = str(groups.get("relationship") or kwargs.get("relationship") or "").strip()
        if not name:
            reply = "用法：/角色添加 角色名 [作品] [关系]"
            return True, reply, 2 if await self._reply(stream_id, reply) else 0
        self._pending_additions[self._pending_key(kwargs)] = PendingAddition(
            created_at=time.monotonic(), name=name, work=work, relationship=relationship
        )
        reply = (
            f"开始登记「{name}」。接下来**连续发图**吧——每张（或每批）都会并进外观卡，"
            "不同服装、不同形态的图各发一张最有用（同一造型的不同角度基本是重复信息）。"
            "发完用 /结束角色添加 收工；每批图后 5 分钟内有效，/取消角色添加 可退出。"
        )
        return True, reply, 2 if await self._reply(stream_id, reply) else 0

    @Command(
        "character_add_done",
        description="结束角色登记，汇总这批图建出的外观卡",
        pattern=r"^\s*[/／]\s*(?:结束角色添加|完成角色添加|角色添加完成)\s*$",
    )
    async def cmd_character_add_done(self, **kwargs: Any) -> "tuple[bool, str, int]":
        stream_id = self._stream_id(kwargs)
        if not self._is_admin(kwargs):
            reply = self._admin_hint()
            return True, reply, 2 if await self._reply(stream_id, reply) else 0
        pending = self._pending_additions.pop(self._pending_key(kwargs), None)
        if pending is None:
            reply = "当前没有进行中的角色登记。用 /角色添加 角色名 开始。"
            return True, reply, 2 if await self._reply(stream_id, reply) else 0
        character = self._repository.find_name(pending.name) if self._repository is not None else None
        if character is None:
            reply = f"「{pending.name}」没建起来（这批图都没抽出可用外观卡），可以重新 /角色添加。"
            return True, reply, 2 if await self._reply(stream_id, reply) else 0
        cards = list(character.appearance_cards)
        reply = (
            f"已结束登记，共收了 {pending.images} 张图。\n"
            + _render_appearance_cards(cards, title=f"「{character.name}」")
        )
        if self.config.library.compress_on_finish:
            # 整理要跑几十秒（模型 30~55s），不能卡着回执不放。丢后台，完成后另发一条。
            # 注意这**不解决** cap.call 的 30s RPC 超时——那个限制在宿主侧，放后台一样会断；
            # 后台化省的是"你要干等 50 秒"，顺带让整理失败不占着命令的返回路径。
            self._track_task(asyncio.create_task(
                self._compress_and_report(stream_id, character.name, cards)
            ))
            reply += "\n（外观卡整理已在后台进行，完成后会另发一条）"
        reply += "\n要补卡可再 /角色添加 同名，或发图后 /识图修正"
        return True, reply, 2 if await self._reply(stream_id, reply) else 0

    @Command(
        "character_add_cancel",
        description="取消进行中的角色登记",
        pattern=r"^\s*[/／]\s*(?:取消角色添加|取消添加)\s*$",
    )
    async def cmd_character_add_cancel(self, **kwargs: Any) -> "tuple[bool, str, int]":
        stream_id = self._stream_id(kwargs)
        if not self._is_admin(kwargs):
            reply = self._admin_hint()
            return True, reply, 2 if await self._reply(stream_id, reply) else 0
        existed = self._pending_additions.pop(self._pending_key(kwargs), None)
        reply = "已取消，本次登记的图不入库。" if existed else "当前没有进行中的角色登记。"
        return True, reply, 2 if await self._reply(stream_id, reply) else 0

    @Command(
        "character_correct",
        description="用被引用的图片（可含多张）补充某个角色的外观卡",
        pattern=r"^\s*[/／]\s*(?:识图修正|修正角色)\s+(?P<name>\S+)\s*$",
    )
    async def cmd_character_correct(self, **kwargs: Any) -> "tuple[bool, str, int]":
        stream_id = self._stream_id(kwargs)
        if not self._is_admin(kwargs):
            reply = self._admin_hint()
            return True, reply, 2 if await self._reply(stream_id, reply) else 0
        groups = kwargs.get("matched_groups") if isinstance(kwargs.get("matched_groups"), dict) else {}
        name = str(groups.get("name") or kwargs.get("name") or "").strip()
        if self._repository is None or not name.strip():
            reply = "用法：/识图修正 角色名 —— 先引用一条带图的消息（有多张就一起用），或先发一张图"
            return True, reply, 2 if await self._reply(stream_id, reply) else 0
        if self._repository.find_name(name) is None:
            reply = f"角色库里没有「{name}」，先用 /角色添加 收录。"
            return True, reply, 2 if await self._reply(stream_id, reply) else 0
        images: "list[bytes]" = []
        reference = str(kwargs.get("reply_to") or "")
        message = kwargs.get("message")
        if not reference and isinstance(message, dict):
            reference = str(message.get("reply_to") or "")
        if reference:
            # 被引用的消息里可能有多张图（同一角色的不同服装常常合成一条发），全都用上。
            images = await self._message_images(
                reference, limit=self.config.plugin.max_images_per_message
            )
        if not images:
            # 不引用也能用：直接吃这个会话刚发的那几张（按时间窗过滤，老图不算数）。
            images = self._recent_images(
                self._pending_key(kwargs), limit=self.config.plugin.max_images_per_message
            )
        if not images:
            reply = "没有找到可用的图片：先发一张（或几张）图，再执行本命令。"
            return True, reply, 2 if await self._reply(stream_id, reply) else 0
        existing = self._known_cards(name)
        results = await asyncio.gather(
            *(self._cards_from_bytes(data, existing=existing) for data in images),
            return_exceptions=True,
        )
        cards: list[str] = []
        failed = 0
        for outcome in results:
            if isinstance(outcome, BaseException) or not outcome:
                failed += 1
                continue
            cards.extend(outcome)
        if not cards:
            reply = f"这 {len(images)} 张图都没抽出可用的外观卡，换更清晰的图再试。"
            return True, reply, 2 if await self._reply(stream_id, reply) else 0
        try:
            updated = await self._write(lambda: self._append_cards(name, cards))
        except Exception as exc:
            reply = f"修正失败：{exc}"
            return True, reply, 2 if await self._reply(stream_id, reply) else 0
        self._index.clear()
        reply = (
            f"从 {len(images)} 张图里为「{updated.name}」补充了 {len(cards)} 条"
            + (f"（{failed} 张没抽到卡）" if failed else "")
            + "。\n"
            + _render_appearance_cards(list(updated.appearance_cards), title=f"「{updated.name}」")
        )
        return True, reply, 2 if await self._reply(stream_id, reply) else 0

    @Command(
        "character_list",
        description="查看角色库",
        pattern=r"^\s*[/／]\s*(?:角色列表|角色库|list)\s*$",
    )
    async def cmd_character_list(self, **kwargs: Any) -> "tuple[bool, str, int]":
        stream_id = self._stream_id(kwargs)
        if not self._is_admin(kwargs):
            reply = self._admin_hint()
            return True, reply, 2 if await self._reply(stream_id, reply) else 0
        if self._repository is None or not len(self._repository):
            reply = "角色库是空的。用 /角色添加 角色名 开始收录。"
            return True, reply, 2 if await self._reply(stream_id, reply) else 0
        characters = self._repository.list_characters()
        lines = [f"【角色识别】角色库共 {len(characters)} 个："]
        for index, character in enumerate(characters[:100], start=1):
            extra = f"（{character.work}）" if character.work else ""
            alias = f" 别名：{'、'.join(character.aliases)}" if character.aliases else ""
            lines.append(f"{index}. {character.name}{extra}{alias}")
        if len(characters) > 100:
            lines.append(f"… 仅显示前 100 个，共 {len(characters)} 个")
        reply = "\n".join(lines)
        return True, reply, 2 if await self._reply(stream_id, reply) else 0

    @Command(
        "character_view",
        description="查看某个角色的资料",
        pattern=r"^\s*[/／]\s*(?:查看角色|查看人设)\s+(?P<name>\S+)\s*$",
    )
    async def cmd_character_view(self, **kwargs: Any) -> "tuple[bool, str, int]":
        stream_id = self._stream_id(kwargs)
        if not self._is_admin(kwargs):
            reply = self._admin_hint()
            return True, reply, 2 if await self._reply(stream_id, reply) else 0
        groups = kwargs.get("matched_groups") if isinstance(kwargs.get("matched_groups"), dict) else {}
        name = str(groups.get("name") or kwargs.get("name") or "").strip()
        character = self._repository.find_name(name) if self._repository is not None else None
        reply = _render_character(character, "auto") if character else f"角色库里没有「{name}」。"
        return True, reply, 2 if await self._reply(stream_id, reply) else 0

    @Command(
        "character_persona",
        description="设置角色设定",
        pattern=r"^\s*[/／]\s*设置人设\s+(?P<name>\S+)\s+(?P<value>.+?)\s*$",
    )
    async def cmd_character_persona(self, **kwargs: Any) -> "tuple[bool, str, int]":
        return await self._field_command(kwargs, "persona", "设定")

    @Command(
        "character_work",
        description="设置角色所属作品",
        pattern=r"^\s*[/／]\s*设置作品\s+(?P<name>\S+)\s+(?P<value>.+?)\s*$",
    )
    async def cmd_character_work(self, **kwargs: Any) -> "tuple[bool, str, int]":
        return await self._field_command(kwargs, "work", "作品")

    @Command(
        "character_relationship",
        description="设置角色与 bot 的关系",
        pattern=r"^\s*[/／]\s*设置关系\s+(?P<name>\S+)\s+(?P<value>.+?)\s*$",
    )
    async def cmd_character_relationship(self, **kwargs: Any) -> "tuple[bool, str, int]":
        return await self._field_command(kwargs, "relationship", "关系")

    @Command(
        "character_alias_add",
        description="为角色添加别名",
        pattern=r"^\s*[/／]\s*添加别名\s+(?P<name>\S+)\s+(?P<value>.+?)\s*$",
    )
    async def cmd_character_alias_add(self, **kwargs: Any) -> "tuple[bool, str, int]":
        return await self._field_command(kwargs, "alias_add", "别名")

    @Command(
        "character_alias_delete",
        description="删除角色的别名",
        pattern=r"^\s*[/／]\s*删除别名\s+(?P<name>\S+)\s+(?P<value>.+?)\s*$",
    )
    async def cmd_character_alias_delete(self, **kwargs: Any) -> "tuple[bool, str, int]":
        return await self._field_command(kwargs, "alias_delete", "别名")

    @Command(
        "character_delete",
        description="删除角色",
        pattern=r"^\s*[/／]\s*(?:删除角色)\s+(?P<name>\S+)\s*$",
    )
    async def cmd_character_delete(self, **kwargs: Any) -> "tuple[bool, str, int]":
        stream_id = self._stream_id(kwargs)
        if not self._is_admin(kwargs):
            reply = self._admin_hint()
            return True, reply, 2 if await self._reply(stream_id, reply) else 0
        groups = kwargs.get("matched_groups") if isinstance(kwargs.get("matched_groups"), dict) else {}
        name = str(groups.get("name") or kwargs.get("name") or "").strip()
        try:
            removed = await self._write(
                lambda: self._repository.delete(name) if self._repository is not None else None
            )
        except ValueError as exc:
            reply = str(exc)
            return True, reply, 2 if await self._reply(stream_id, reply) else 0
        self._index.clear()
        reply = f"已删除角色「{removed.name}」。" if removed else "删除失败。"
        return True, reply, 2 if await self._reply(stream_id, reply) else 0

    @Command(
        "rebuild_index",
        description="清空并重建角色库向量索引",
        pattern=r"^\s*[/／]\s*(?:重建索引|刷新索引)\s*$",
    )
    async def cmd_rebuild_index(self, **kwargs: Any) -> "tuple[bool, str, int]":
        stream_id = self._stream_id(kwargs)
        if not self._is_admin(kwargs):
            reply = self._admin_hint()
            return True, reply, 2 if await self._reply(stream_id, reply) else 0
        self._index.clear()
        self._cache.clear()
        self._embed_unavailable_until = 0.0
        if self._repository is not None:
            self._repository.load()
        count = len(self._repository) if self._repository is not None else 0
        reply = f"已清空索引与缓存。角色库现有 {count} 个角色，下次识图会重建向量。"
        return True, reply, 2 if await self._reply(stream_id, reply) else 0

    # ---------------------------------------------------------- 内部工具

    def _admin_hint(self) -> str:
        count = len({self._normalize_account(item) for item in (self.config.library.admin_ids or [])} - {""})
        if not count:
            return "该命令仅管理员可用，但当前还没有配置管理员。请在插件配置的 admin_ids 里填入管理账号。"
        return f"该命令仅管理员可用（当前已配置 {count} 个管理员）。"

    async def _available_models(self) -> str:
        try:
            result = await self.ctx.llm.get_available_models()
        except Exception as exc:
            return f"读取失败（{type(exc).__name__}）"
        if isinstance(result, dict):
            values = result.get("models") or result.get("data") or []
        else:
            values = result or []
        if isinstance(values, list) and values:
            names = [str(item.get("name") if isinstance(item, dict) else item) for item in values]
            return "、".join(names[:20])
        return "未返回任何模型"

    def _embed_available(self) -> bool:
        return self.config.retrieval.embedding_enabled and time.monotonic() >= self._embed_unavailable_until

    async def _write(self, action):
        """串行化所有角色库写操作（并发命令会互相踩）。"""
        async with self._lock:
            return await asyncio.to_thread(action)

    def _append_cards(self, name: str, cards: "list[str]"):
        if self._repository is None:
            raise ValueError("角色库未初始化")
        return self._repository.append_appearance_cards(name, cards)

    async def _compress_and_report(self, stream_id: str, name: str, cards: "list[str]") -> None:
        """后台整理外观卡，跑完补一条消息。

        整理要几十秒，``/结束角色添加`` 的回执不该陪着等。失败也发一条**简短**通知：
        不发的话用户会默认整理成功了。
        """
        merged, note = await self._compress_cards(name, cards)
        if merged is None:
            await self._reply(stream_id, f"「{name}」的外观卡未整理：{note}")
            return
        await self._reply(
            stream_id,
            f"「{name}」的外观卡整理完成：{len(cards)} → {len(merged)} 条"
            "（只合并了同类重复的说法，造型差异与细节保留）。\n"
            + _render_appearance_cards(merged, title=f"「{name}」"),
        )

    async def _cards_from_component(
        self, component: dict, existing: "Sequence[str]" = ()
    ) -> "list[str]":
        """从一张消息图片组件里抽外观卡：先解出图片字节，再交给 ``_cards_from_bytes``。

        这一层存在的意义只是"组件 → 字节"的适配（base64 / url 两条来源）。拆成两层是
        为了让 ``/识图修正`` 那条已经拿到字节的路径不必再伪造一个组件出来。
        """
        image_bytes, note = await self._resolve_image(component)
        if image_bytes is None:
            self._log("warning", "图片载荷不可用：%s", note)
            return []
        return await self._cards_from_bytes(image_bytes, existing=existing)

    async def _cards_from_bytes(
        self, image_bytes: bytes, existing: "Sequence[str]" = ()
    ) -> "list[str]":
        """从图片字节抽外观卡。

        失败返回空列表而不抛——批量处理时单张失败不该拖垮整批。

        ``existing`` 是库里已有的卡片，会一并进提示词让模型只补新特征：看不见已有内容时，
        第二张图必然把"浅蓝色长发"再写一遍，攒到第十张就全是变体。
        """
        if not image_bytes:
            return []
        try:
            cards = await build_appearance_cards(
                provider=self.config.vision.provider,
                image_bytes=image_bytes,
                generate=self._generate,
                task_name=self.config.vision.task_name,
                model_name=self.config.vision.model_name,
                api_key=self.config.vision.api_key,
                base_url=self.config.vision.base_url,
                # 建卡用更宽松的超时：用户此时已预期要等，不该和实时识别共用短超时。
                timeout_seconds=self.config.plugin.card_timeout_seconds,
                max_tokens=self.config.vision.max_tokens,
                max_upload_bytes=self.config.vision.max_upload_bytes,
                existing_cards=existing,
            )
        except Exception as exc:
            self._log("warning", "外观卡抽取失败：%s", exc)
            return []
        return [str(card) for card in cards]

    def _vision_timeout_hint(self) -> str:
        """视觉请求超时时给一次可操作建议。

        "超时"本身对用户没有指导意义：他不知道该改哪、改成多少。而这件事的根因几乎总是
        模型太慢（真机实测 50s 级），处理方式只有两条——换快模型，或调大超时。
        只提示一次，免得刷屏。
        """
        if self._vision_timeout_hinted:
            return ""
        self._vision_timeout_hinted = True
        return (
            f"→ 当前单张超时 {self.config.plugin.image_timeout_seconds:.0f}s；"
            "该模型单次常超过它。处理方式：换更快的视觉模型，或调大 plugin.image_timeout_seconds"
        )

    def _known_cards(self, name: str) -> "list[str]":
        """取某个角色库里已有的外观卡（给抽卡提示词用）。库不可用时返回空。"""
        if self._repository is None or not str(name or "").strip():
            return []
        character = self._repository.find_name(name)
        return list(character.appearance_cards) if character is not None else []

    async def _compress_cards(
        self, name: str, cards: "list[str]"
    ) -> "tuple[list[str] | None, str]":
        """结束登记时把同类措辞整理一遍。返回 ``(整理后的卡片, 说明)``。

        卡片为 None 表示保持原样，说明写清原因。原因不该被吞掉：**超时、跳过、模型不听话
        的处理方式完全不同**（调超时/换模型/改提示词），只说一句"未生效"等于没说——
        真机第一次跑就是这么瞎的（实际是 56.8s 超了 45s 的限）。

        写回走 ``set_appearance_cards``（**替换**）而不是 ``append``：整理的结果就是要顶掉旧的
        那一堆，再走合并等于白压。失败一律保持原样——绝不能把用户攒的卡片弄丢。
        """
        if not self.config.library.compress_on_finish:
            return None, "整理已关闭"
        if self._repository is None:
            return None, "角色库未初始化"
        merged, note = await compress_appearance_cards(
            cards=cards,
            provider=self.config.vision.provider,
            generate=self._generate,
            # 整理是纯文本活，**不能沿用视觉任务的模型**（实测 54.9~56.8s，而 Host 的
            # cap.call RPC 硬超时 30s）。配置留空就走 COMPRESS_DEFAULT_TASK。
            task_name=self.config.library.compress_task_name or COMPRESS_DEFAULT_TASK,
            model_name="",  # 任务名已经决定了模型；再传视觉模型名会把小任务压回大模型
        )
        if merged is None:
            self._log("info", "外观卡整理未采用：%s", note)
            if "超时" in note or "E_TIMEOUT" in note:
                return None, (
                    f"{note}；该任务上的模型太慢（Host 的 cap.call 硬超时 30s），"
                    f"可在 library.compress_task_name 换更快的小任务，或关掉 compress_on_finish"
                )
            return None, note
        # 整理是慢活：这期间用户完全可能又补了卡（整理在后台跑时尤其如此）。
        # 快照对不上就放弃——**拿旧结果覆盖新卡片是不可接受的**。
        if self._known_cards(name) != cards:
            self._log("info", "外观卡在整理期间被改动，丢弃本次整理结果")
            return None, "整理期间外观卡被改动，结果已丢弃（现有卡片不受影响）"
        try:
            updated = await self._write(
                lambda: self._repository.set_appearance_cards(name, merged)
            )
        except (ValueError, RuntimeError) as exc:
            self._log("warning", "外观卡整理写回失败，保持原样：%s", exc)
            return None, f"写回失败：{exc}"
        self._index.clear()
        return list(updated.appearance_cards), ""

    async def _collect_character_images(
        self, message: dict, images: "list[dict]", pending: PendingAddition
    ) -> None:
        """把这一批图全部收进登记中的角色卡。

        单张图也走这条路径（只有一条代码路径，少一处会分叉的地方）。连续发图时每批都
        合并进同一份外观卡——``upsert`` 本身是**幂等合并**（按 card 去重、上限 15 条、
        ``work or existing.work`` 保留旧值），所以不需要区分"首次创建"与"后续补充"。
        """
        stream_id = self._stream_id({"message": message})
        if self._repository is None:
            await self._reply(stream_id, "角色库未初始化，无法创建角色。")
            return
        batch = images[: self.config.plugin.max_images_per_message]
        existing = self._known_cards(pending.name)
        results = await asyncio.gather(
            *(self._cards_from_component(component, existing=existing) for component in batch),
            return_exceptions=True,
        )
        cards: list[str] = []
        failed = 0
        for outcome in results:
            if isinstance(outcome, BaseException) or not outcome:
                failed += 1
                continue
            cards.extend(outcome)
        if not cards:
            await self._reply(
                stream_id,
                f"这批 {len(batch)} 张图都没抽到可用的外观卡，换更清晰的图再试。"
                "（登记还在，发图继续；/取消角色添加 可退出）",
            )
            return
        try:
            saved = await self._write(
                lambda: self._repository.upsert(
                    name=pending.name,
                    work=pending.work,
                    relationship=pending.relationship,
                    appearance_cards=cards,
                )
            )
        except ValueError as exc:
            self._pending_additions.pop(self._pending_key(message), None)
            await self._reply(stream_id, f"角色未收录：{exc}（登记已结束）")
            return
        self._index.clear()
        pending.touched()
        pending.images += len(batch)
        stored = list(saved.appearance_cards)
        # 上限会**静默**吃掉卡片：已有的加本批若超过 15 条，最早的那些直接消失。
        # 不报出来，用户只会看到"我明明又发了几张，卡怎么没变多"。
        dropped = max(0, pending.cards + len(cards) - len(stored))
        pending.cards = len(stored)
        reply = (f"「{saved.name}」已累积 {pending.images} 张图、{pending.cards} 条外观卡"
                 f"（{_appearance_summary(stored)}，本批 +{len(cards)}）")
        if failed:
            reply += f"，其中 {failed} 张没抽到卡"
        if dropped:
            reply += f"，另有 {dropped} 条未入库（与已有重复，或超出 15 条上限）"
        reply += "。继续发图，发完用 /结束角色添加 收工。"
        await self._reply(stream_id, reply)

    async def _field_command(self, kwargs: dict, field: str, label: str) -> "tuple[bool, str, int]":
        """设置类命令的公共实现。"""
        stream_id = self._stream_id(kwargs)
        if not self._is_admin(kwargs):
            reply = self._admin_hint()
            return True, reply, 2 if await self._reply(stream_id, reply) else 0
        groups = kwargs.get("matched_groups") if isinstance(kwargs.get("matched_groups"), dict) else {}
        name = str(groups.get("name") or kwargs.get("name") or "").strip()
        value = str(groups.get("value") or kwargs.get("value") or "").strip()
        if self._repository is None or not name or not value:
            reply = f"用法：/{label}相关命令 角色名 值"
            return True, reply, 2 if await self._reply(stream_id, reply) else 0
        actions = {
            "persona": lambda: self._repository.set_persona(name, value),
            "work": lambda: self._repository.set_work(name, value),
            "relationship": lambda: self._repository.set_relationship(name, value),
            "alias_add": lambda: self._repository.add_alias(name, value),
            "alias_delete": lambda: self._repository.remove_alias(name, value),
        }
        try:
            await self._write(actions[field])
        except ValueError as exc:
            reply = str(exc)
            return True, reply, 2 if await self._reply(stream_id, reply) else 0
        self._index.clear()
        reply = f"已更新「{name}」的{label}。"
        return True, reply, 2 if await self._reply(stream_id, reply) else 0


# ══════════════════════════════════════════════════════════════ 模块级辅助


def _names_from_label(label: str) -> "list[str]":
    """从 ``图片[角色A（关系）、角色B]`` 里抠出角色名。

    前缀长度按 ``len(prefix)`` 取，别写死数字：这里曾经写成 ``text[4:]``，而 ``"图片["``
    只有 3 个字符，于是 ``"图片[未识别]"`` 被切出 ``"识别"``——它不等于 ``"未识别"``，
    就被当成一个角色名去查库了。查不到所以没出事，但"未识别"这条路径等于从未被覆盖过。
    """
    prefix = "图片["
    text = str(label or "")
    if not text.startswith(prefix) or "]" not in text:
        return []
    body = text[len(prefix):text.index("]")]
    if body == "未识别":
        return []
    names: list[str] = []
    for item in body.split("、"):
        name = item.split("（", 1)[0].strip()
        if name:
            names.append(name)
    return names


def _appearance_summary(cards: "Sequence[str]") -> str:
    """一行类别计数，如「发色发型 2｜眼睛 1｜服装 3」。没有卡片时返回空串。"""
    groups = group_appearance_cards(cards)
    order = [*CORE_APPEARANCE_CATEGORIES, "配饰", "其他"]
    return "｜".join(f"{name} {len(groups[name])}" for name in order if name in groups)


def _fallback_line(config: Any) -> str:
    """渲染"本地库兜底"这一行，顺手做一次阈值体检。

    为什么要在状态里喊：这个阈值是从"0.55×向量 + 0.35×关键词"的加权和上取的，而那个和会
    随 description 的措辞漂（实测 0.34~0.40）。**落在这个区间里的阈值是最坏的一种配置**
    ——不是一直失败（那会被发现），而是时灵时不灵，让人以为是随机故障。所以只要落在区间
    内就显式警告，并给出建议值。
    """
    line = "本地库兜底：" + ("开" if config.auto_apply_local_confirm else "关")
    line += f"（检索分阈值 {config.local_confirm_min_score}）"
    low, high = SCORE_JITTER_BAND
    if config.auto_apply_local_confirm and low <= config.local_confirm_min_score <= high:
        line += (f"\n  ⚠ 该阈值落在检索分的实测波动区间 {low}~{high} 内："
                 "同一张图会时贴时不贴。建议改成 0（判据回到「进了候选池 + 视觉模型确认」这对双证）")
    return line


def _render_appearance_cards(cards: "Sequence[str]", title: str = "") -> str:
    """把外观卡按类别分组渲染，并点名缺了哪一类。

    平铺成一列时，用户看不出"三类齐不齐"——而那三类恰恰是插件核对候选时用的判据
    （见 prompts 的「三类核对」）。所以分组与缺口提示不是排版装饰，是把内部判据外显：
    缺「眼睛」就补一张能给到眼睛的图，而不是漫无目的地继续发图。
    """
    groups = group_appearance_cards(cards)
    head = f"{title}外观卡" if title else "外观卡"
    total = sum(len(items) for items in groups.values())
    if not groups:
        return f"{head} 0 条（还没有）"
    order = [*CORE_APPEARANCE_CATEGORIES, "配饰", "其他"]
    present = [name for name in order if name in groups]
    lines = [f"{head} {total} 条（{_appearance_summary(cards)}）"]
    for name in present:
        lines.append(f"▸ {name}")
        lines.extend(f"  · {card}" for card in groups[name])
    missing = [name for name in CORE_APPEARANCE_CATEGORIES if name not in groups]
    if missing:
        lines.append(f"⚠ 缺「{'、'.join(missing)}」：再补一张能看到这些特征的图，核对时会稳很多")
    return "\n".join(lines)


def _render_character(character: Character, field: str = "auto") -> str:
    """把角色渲染成给用户或模型看的多行文本。"""
    if field == "persona" and character.persona:
        return character.persona
    if field == "work" and character.work:
        return character.work
    if field == "relationship" and character.relationship:
        return character.relationship
    if field == "aliases" and character.aliases:
        return "、".join(character.aliases)
    if field == "appearance" and character.appearance_cards:
        return _render_appearance_cards(character.appearance_cards)
    lines = [f"{character.name}"]
    if character.aliases:
        lines.append(f"别名：{'、'.join(character.aliases)}")
    if character.work:
        lines.append(f"作品：{character.work}")
    if character.relationship:
        lines.append(f"关系：{character.relationship}")
    if character.persona:
        lines.append(f"设定：{character.persona}")
    if character.appearance_cards:
        lines.append(_render_appearance_cards(character.appearance_cards))
    if not character.persona and not character.work:
        lines.append("（还没填设定：可用 /设置人设、/设置作品 补充）")
    return "\n".join(lines)


def _describe_result(result: RecognitionResult) -> str:
    """把识别结论渲染成工具返回文本。"""
    parts: list[str] = []
    if result.label != UNRECOGNIZED_LABEL:
        parts.append(f"识别结果：{result.label}")
    elif result.fusion.candidates:
        names = "、".join(item.display_name for item in result.fusion.candidates[:3])
        parts.append(f"没能确认角色（候选：{names}，仅单源命中，未采信）")
    else:
        parts.append("没能识别出角色")
    if result.fusion.reason:
        parts.append(f"依据：{result.fusion.reason}")
    if result.fusion.works:
        parts.append(f"可能出自：{'、'.join(result.fusion.works[:3])}")
    if result.description:
        parts.append(f"图片内容：{result.description}")
    if result.fusion.conflict:
        parts.append("注意：多个反查源给出了互相矛盾的结果，请如实说明不确定。")
    return "\n".join(parts)


def _render_source_error(detail: str) -> str:
    """把一句源错误翻译成"下一步该做什么"。

    最关键的是把**限流**和**无命中**分开：用户看到"无命中"会理解成"这张图里没有
    角色"，于是反复重发同一张图——而 429 恰恰是"越试越糟"的失败。所以限流必须显式
    点出来，并给出等待建议。纯函数，可离线单测。
    """
    text = str(detail or "").strip()
    if "被限流" in text or "429" in text:
        return f"{text} → 反查源限流，等一两分钟再试；连续失败会自动熔断，冷却后恢复"
    if "熔断" in text:
        return f"{text} → 该源已暂停，冷却结束会自动恢复"
    if "超时" in text:
        return f"{text} → 网络超时，可稍后重试"
    return text


def _diagnose_local_confirm(diag: dict) -> str:
    """说清楚兜底通路这次为什么没救回这张图。

    "未识别"最容易被读成"库里没有这个人"，但实际的堵点有四个，处理方式完全不同：
    反查源没查成 / 检索没选中 / 检索分不够 / 视觉模型没确认。把这四个岔路口分开报，
    用户才知道该去调阈值、去补外观卡，还是去修模型配置。
    """
    if diag.get("tier") == TIER_LOCAL:
        return "已触发（这次就是靠它贴上的标签）"
    label = str(diag.get("label") or "")
    if label and label != UNRECOGNIZED_LABEL:
        return "未使用（反查源自己给出了结论）"
    if diag.get("local_confirmed"):
        return "条件已满足（结论见上方档位）"
    names = [str(item) for item in (diag.get("vision_names") or ())]
    if not names:
        # 把视觉模型那一步自己报的原因带出来（模型没给候选 / 判为库外 / 有冲突特征 / 证据不足），
        # 否则用户只会看到"未确认"，然后去改没错的那一环。
        return f"未触发：{diag.get('vision', '视觉模型没有确认任何候选')}（双证缺一半）"
    scores = {str(name): float(score) for name, score in (diag.get("retrieval_scores") or ())}
    threshold = float(diag.get("threshold") or 0.0)
    # 只有**低于**阈值的才算被阈值挡住。分数过线却报"没过阈值"，会把人骗去调一个
    # 本来就没问题的参数。
    below = [f"{name} {scores[name]:.2f}" for name in names if name in scores and scores[name] < threshold]
    if below:
        return (
            f"未触发：模型确认了 {'、'.join(below)}，但检索分没过阈值 "
            f"{threshold} → 可下调 fusion.local_confirm_min_score"
        )
    missing = [name for name in names if name not in scores]
    if missing:
        return f"未触发：模型确认的 {'、'.join(missing)} 不在检索候选里（检索没把它选出来）"
    disputes = [str(item) for item in (diag.get("reverse_disputes") or ())]
    if disputes:
        return (
            f"未触发：模型确认了 {'、'.join(names)}，但反查源指向库内另一个角色 "
            f"{'、'.join(disputes)}，分歧时不贴（错贴一个人名比不贴更伤）"
        )
    return "未触发：双证均已满足但结论未生效，请把本段诊断回贴给开发者"


def _render_diagnosis(diag: "dict | None") -> str:
    """把最近一次识图的链路诊断渲染成可读的几行。"""
    if not diag:
        return "  还没有识图记录：先在本会话发一张图，再回来按一次。"
    names = "、".join(str(item) for item in (diag.get("vision_names") or ()))
    lines = [
        f"  缓存：{diag.get('cache', '未知')}",
        f"  反查源：{diag.get('reverse', '未跑')}"
        + ("" if diag.get("description", True) else "｜图片描述为空（视觉通道没跑通）"),
        f"  角色库：{diag.get('characters', 0)} 个角色｜本地检索：{diag.get('retrieval', '未跑')}",
        f"  视觉候选校验：{diag.get('vision', '未跑')}" + (f"（{names}）" if names else ""),
        f"  判定档位：{diag.get('tier', '?')}｜标签：{diag.get('label', '')}",
    ]
    for item in diag.get("reverse_errors") or ():
        lines.append(f"    · {_render_source_error(str(item))}")
    raw = str(diag.get("vision_raw") or "").strip().replace("\n", " ")
    if raw:
        lines.append(f"  视觉模型原始输出：{raw[:300]}" + ("…" if len(raw) > 300 else ""))
    if diag.get("reason"):
        lines.append(f"  依据：{diag['reason']}")
    lines.append("  本地库兜底通路：" + _diagnose_local_confirm(diag))
    return "\n".join(lines)


def _render_probe(summary: dict) -> str:
    if not summary.get("ok"):
        return f"探测失败（{summary.get('error')}）"
    parts = [f"顶层字段 {summary.get('top_level_keys')}"]
    for key in ("results_count", "data_count", "result_count"):
        if key in summary:
            parts.append(f"{key.replace('_count', '')}={summary[key]}")
    if "characters_populated_of_10" in summary:
        parts.append(f"前 10 条里 characters 非空的：{summary['characters_populated_of_10']}")
    if "has_anilist" in summary:
        parts.append(f"含 anilist 元数据：{summary['has_anilist']}")
    return "；".join(str(item) for item in parts)


def _status_lines(config: Any, repository: Any, source_configs: dict, cache_size: int,
                  index_size: int, image_count: int, embed_available: bool) -> str:
    """状态摘要。降级状态必须外显，否则用户会以为向量检索在工作。

    ``repository`` 的真假判断必须用 ``is None``，**不能用真值测试**：
    ``CharacterRepository`` 定义了 ``__len__``，空库的 ``bool()`` 就是 ``False``，
    于是"库是空的"会被报成"未初始化"。这两件事的排查方向完全相反——前者去
    ``/角色添加`` 建卡就行，后者才要怀疑数据目录不可写。用真值测试等于把用户
    引到错误的方向上，还会白翻一遍启动日志。
    """
    if repository is None:
        library_line = "角色库：未初始化（数据目录不可用，见启动日志）"
    else:
        library_line = f"角色库：{len(repository)} 个角色（{repository.path.name}）"
        if not len(repository):
            library_line += "｜库为空：先 /角色添加 建第一张卡"
    lines = [
        library_line,
        f"识别：{'开' if config.plugin.enabled else '关'}｜"
        f"视觉：{config.vision.provider if config.vision.enabled else '关'}｜"
        f"知识注入：{'开' if config.injection.enabled else '关'}",
        f"候选上限：{config.library.max_prompt_characters}｜反查预留：{config.library.pinned_reserve}",
        "反查源：" + "、".join(
            f"{name}{'开' if cfg.enabled else '关'}" for name, cfg in source_configs.items()
        ),
        "单源自动贴标签：" + ("开" if config.fusion.auto_apply_single_source else "关（更安全）"),
        _fallback_line(config.fusion),
        f"缓存：{cache_size} 条｜向量索引：{index_size} 条｜最近图片记忆：{image_count} 条",
        "向量检索：" + ("可用" if embed_available else "不可用，已降级为关键词检索"),
    ]
    return "\n".join(lines)


def create_plugin() -> CharacterRecognizerPlugin:
    """创建插件实例。"""
    return CharacterRecognizerPlugin()
