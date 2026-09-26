# -*- coding: utf-8 -*-
"""反查源 adapter（不依赖 ctx）。

**v1 只实现 AnimeTrace**——它是唯一已被验证可用、免费、无需 API Key 的源。
SauceNAO 与 trace.moe 的价值还没被真图验证过（SauceNAO 的 ``characters`` 字段实测
可能三到五成是空的），所以先不写它们的完整 adapter，改由 ``probe_endpoint()`` 发一次
原始请求、把响应形态报回来，用真实数据决定值不值得写。这样不会把代码写在无效数据源上。

接口按"多源"设计：``run_source()`` 用名字查 adapter，后补源不用改融合层。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from .imaging import prepare_upload_async
    from .models import SourceHit
    from .textutil import clean_display_name, is_plausible_name
except ImportError:  # pragma: no cover - 取决于加载方式
    from imaging import prepare_upload_async
    from models import SourceHit
    from textutil import clean_display_name, is_plausible_name

MAX_RESPONSE_BYTES = 2_000_000
#: AnimeTrace 对 multipart 体积有硬限制（超限 413），留出余量。
ANIME_TRACE_UPLOAD_BYTES = 900_000


class SourceError(RuntimeError):
    """反查源调用失败（网络、HTTP 状态、响应结构）。

    ``transient`` 区分"这个源暂时不行"和"这次请求本身就不该发"：只有前者计入熔断。
    图片压不下来（例如环境里没有 Pillow）属于后者——若也计入熔断，两次之后整个源就被
    停掉，用户会以为服务挂了，其实是本地缺个依赖。

    ``rate_limited`` 单独标出限流（HTTP 429）。它和"无命中"是**完全不同的结论**：
    无命中说明源正常查了、图里确实没认出角色；限流说明这一次根本没查成。两者混在
    一句"无命中"里报给用户，用户会以为图里没角色，于是反复重发同一张图——只会把
    限流窗口越推越长。所以要能一路传到展示层。
    """

    def __init__(
        self, message: str, *, transient: bool = True, rate_limited: bool = False
    ) -> None:
        super().__init__(message)
        self.transient = transient
        self.rate_limited = rate_limited


@dataclass(frozen=True)
class SourceConfig:
    """单个反查源的配置。由 plugin.py 从插件配置映射过来。"""

    name: str
    enabled: bool = False
    url: str = ""
    timeout_seconds: float = 15.0
    max_upload_bytes: int = ANIME_TRACE_UPLOAD_BYTES
    api_key: str = ""
    max_candidates: int = 3
    confident_similarity: float = 85.0
    weak_similarity: float = 60.0


def _multipart(image_bytes: bytes, mime_type: str, *, field: str = "file") -> tuple[bytes, str]:
    boundary = f"----MaiBotCharacter{uuid.uuid4().hex}"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="image"\r\n'
        f"Content-Type: {mime_type}\r\n\r\n"
    ).encode("ascii")
    tail = f"\r\n--{boundary}--\r\n".encode("ascii")
    return head + image_bytes + tail, boundary


def _rate_limit_reason(detail: str) -> str:
    """从限流响应里抠一句人话。

    服务商的 429 正文经常是"多语言 + 机器码"的 JSON（AnimeTrace 就是
    ``{"zh_message": "...", "msg": "...", "detail": "Too Many Requests"}``），
    整段塞给用户没有信息量；正文被截断时 JSON 还可能解析不出来，所以要能退回原文。
    纯函数，可离线单测。
    """
    text = str(detail or "").strip()
    try:
        payload: Any = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, dict):
        for key in ("zh_message", "message", "msg", "detail"):
            value = str(payload.get(key) or "").strip()
            if value:
                return value[:80]
    return text[:80] or "服务未给出原因"


def _post(
    url: str,
    body: bytes,
    content_type: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    request = Request(url, data=body, method="POST", headers={"Content-Type": content_type})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        detail = exc.read(501).decode("utf-8", errors="replace")[:300]
        if exc.code == 429:
            raise SourceError(
                f"被限流：{_rate_limit_reason(detail)}（HTTP 429）",
                rate_limited=True,
            ) from exc
        raise SourceError(f"HTTP {exc.code}：{detail}") from exc
    except URLError as exc:
        raise SourceError(f"连接失败：{exc.reason}") from exc
    except TimeoutError as exc:
        raise SourceError("请求超时") from exc
    if len(raw) > MAX_RESPONSE_BYTES:
        raise SourceError(f"响应超过 {MAX_RESPONSE_BYTES} 字节")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceError(f"响应不是合法 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise SourceError("响应顶层不是对象")
    return payload


# ---------------------------------------------------------------- AnimeTrace

def parse_anime_trace(payload: dict[str, Any], *, max_candidates: int = 3) -> tuple[SourceHit, ...]:
    """解析 AnimeTrace 响应。纯函数，可离线单测。

    注意 ``not_confident`` 是**条目级**布尔，不是每个角色各有一个。
    """
    entries = payload.get("data")
    if not isinstance(entries, list):
        return ()
    hits: list[SourceHit] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        not_confident = bool(entry.get("not_confident"))
        characters = entry.get("character")
        if not isinstance(characters, list):
            continue
        for character in characters:
            if not isinstance(character, dict):
                continue
            name = clean_display_name(character.get("character"))
            work = clean_display_name(character.get("work"))
            if not is_plausible_name(name):
                continue
            key = (name.casefold(), work.casefold())
            if key in seen:
                continue
            seen.add(key)
            hits.append(
                SourceHit(
                    source="anime_trace",
                    raw_name=name,
                    work=work,
                    confident=not not_confident,
                    # AnimeTrace 不给数值分数，只有这个布尔。别伪造一个"分数"出来。
                    score=0.0,
                    detail="服务标注低置信" if not_confident else "服务报告置信",
                )
            )
    # 高置信优先，低置信只作为补充信息。
    hits.sort(key=lambda item: item.confident, reverse=True)
    return tuple(hits[:max_candidates])


async def search_anime_trace(image_bytes: bytes, config: SourceConfig) -> tuple[SourceHit, ...]:
    try:
        upload, mime_type = await prepare_upload_async(image_bytes, max_bytes=config.max_upload_bytes)
    except ValueError as exc:
        raise SourceError(f"图片无法压到上传上限：{exc}", transient=False) from exc
    body, boundary = _multipart(upload, mime_type)
    url = config.url.rstrip("/") + "/v1/search"
    payload = await asyncio.to_thread(
        _post, url, body, f"multipart/form-data; boundary={boundary}", config.timeout_seconds
    )
    return parse_anime_trace(payload, max_candidates=config.max_candidates)


# ---------------------------------------------------------------- 源注册表

Adapter = Callable[[bytes, SourceConfig], Awaitable[Any]]

ADAPTERS: dict[str, Adapter] = {
    "anime_trace": search_anime_trace,
}

#: 已实现 adapter 的源，用于 /识别探测 里区分"已接入"与"仅探测"
IMPLEMENTED_SOURCES = tuple(ADAPTERS)


async def run_source(
    config: SourceConfig,
    image_bytes: bytes,
    *,
    breaker: Any = None,
) -> tuple[tuple[SourceHit, ...], str]:
    """跑一个源，返回 ``(命中, 错误说明)``。错误说明为空表示成功。

    熔断打开时直接跳过，不去耗满超时。任何异常都被吸收成一句可读错误——单个源挂掉
    不该让整条识图链路失败。
    """
    adapter = ADAPTERS.get(config.name)
    if adapter is None:
        return (), f"未实现的源：{config.name}"
    if not config.enabled:
        return (), ""
    if breaker is not None and not breaker.allow(config.name):
        return (), f"熔断中（{breaker.remaining_cooldown(config.name):.0f}s 后重试）"
    try:
        hits = await adapter(image_bytes, config)
    except SourceError as exc:
        if breaker is not None and exc.transient:
            breaker.record_failure(config.name)
        return (), str(exc)
    except Exception as exc:  # 防御：adapter 内部未预料的异常
        if breaker is not None:
            breaker.record_failure(config.name)
        return (), f"{type(exc).__name__}: {exc}"
    if breaker is not None:
        breaker.record_success(config.name)
    return tuple(hits), ""


# ---------------------------------------------------------------- 原始探测（用于先验证再写）

async def _probe_request(config: SourceConfig, image_bytes: bytes) -> tuple[str, bytes, str]:
    """按各源已知的请求形态构造一次原始探测请求。"""
    if config.name == "saucenao":
        upload, mime_type = await prepare_upload_async(image_bytes, max_bytes=config.max_upload_bytes)
        body, boundary = _multipart(upload, mime_type)
        # SauceNAO 需要把 key / output_type 放在 query string 上
        query = f"key={config.api_key}&output_type=2&numres={max(1, config.max_candidates)}&db=999"
        return f"{config.url.rstrip('/')}/search.php?{query}", body, f"multipart/form-data; boundary={boundary}"
    if config.name == "trace_moe":
        upload, mime_type = await prepare_upload_async(image_bytes, max_bytes=config.max_upload_bytes)
        body, boundary = _multipart(upload, mime_type, field="image")
        return f"{config.url.rstrip('/')}/search?anilistInfo=1", body, f"multipart/form-data; boundary={boundary}"
    raise SourceError(f"没有 {config.name} 的探测请求构造器")


async def probe_endpoint(config: SourceConfig, image_bytes: bytes) -> dict[str, Any]:
    """对一个尚未接入的源发一次原始请求，回报**响应形态**而不是解析结果。

    目的是用一张真实图片回答"这个源值不值得写 adapter"——比如 SauceNAO 的
    ``characters`` 字段到底有多稀疏。所以这里刻意只报顶层字段名与条目数。
    """
    try:
        url, body, content_type = await _probe_request(config, image_bytes)
        payload = await asyncio.to_thread(_post, url, body, content_type, config.timeout_seconds)
    except SourceError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    summary: dict[str, Any] = {"ok": True, "top_level_keys": sorted(payload)[:12]}
    for key in ("results", "data", "result"):
        value = payload.get(key)
        if isinstance(value, list):
            summary[f"{key}_count"] = len(value)
            if value and isinstance(value[0], dict):
                summary[f"{key}_item_keys"] = sorted(value[0])[:15]
    if config.name == "saucenao" and isinstance(payload.get("results"), list):
        populated = 0
        for entry in payload["results"][:10]:
            data = entry.get("data") if isinstance(entry, dict) else None
            if isinstance(data, dict) and str(data.get("characters") or "").strip():
                populated += 1
        summary["characters_populated_of_10"] = populated
    if config.name == "trace_moe" and isinstance(payload.get("result"), list):
        first = payload["result"][0] if payload["result"] else None
        if isinstance(first, dict):
            summary["sample_fields"] = sorted(first)[:12]
            summary["has_anilist"] = "anilist" in first
    return summary


def collect_hits(*groups: "tuple[SourceHit, ...]") -> tuple[SourceHit, ...]:
    """合并多个源的命中，保持源内顺序。"""
    merged: list[SourceHit] = []
    for group in groups:
        merged.extend(group)
    return tuple(merged)
