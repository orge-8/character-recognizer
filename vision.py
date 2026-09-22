# -*- coding: utf-8 -*-
"""视觉识别通道（不依赖 ctx）。

三条通道共用一个"识别 + 从本地库匹配"的提示词：

* ``host``   —— 复用 MaiBot 自己配置的视觉任务，**不需要在插件里再填一份密钥**。
* ``gemini`` / ``openai`` —— 插件自带密钥的直连通道，适合想用更便宜的视觉模型时。

``ctx.llm.generate`` 没有超时参数，所以所有调用都在这里用 ``asyncio.wait_for`` 包住；
超时会被计入熔断，避免一个卡死的模型把每条带图消息都拖满。
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from dataclasses import replace
from typing import Any, Awaitable, Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from .imaging import prepare_upload, sniff_mime_type
    from .models import VisionCandidate, VisionResult
    from .prompts import (
        HOST_DESCRIPTION_PROMPT,
        build_appearance_prompt,
        build_compress_prompt,
        build_identify_prompt,
    )
    from .textutil import CORE_APPEARANCE_CATEGORIES, classify_appearance_card
except ImportError:  # pragma: no cover - 取决于加载方式
    from imaging import prepare_upload, sniff_mime_type
    from models import VisionCandidate, VisionResult
    from prompts import (
        HOST_DESCRIPTION_PROMPT,
        build_appearance_prompt,
        build_compress_prompt,
        build_identify_prompt,
    )
    from textutil import CORE_APPEARANCE_CATEGORIES, classify_appearance_card

#: 注入进来的 Host LLM 调用（plugin.py 负责把 ctx.llm.generate 包成这个形状）
GenerateFn = Callable[..., Awaitable[dict]]

MAX_RESPONSE_BYTES = 2_000_000
DEFAULT_UPLOAD_BYTES = 4_194_304

#: 外观卡条数上限（与 repository 的裁剪保持一致）。
APPEARANCE_CARD_LIMIT = 15
#: 少于这么多条就不值得为"整理"多花一次调用。
COMPRESS_MIN_CARDS = 8


def _covered_kinds(cards: "Sequence[str]") -> "set[str]":
    """这组卡片覆盖了哪些**核心类别**（发色发型 / 眼睛 / 服装 / 配饰）。

    只看核心类别："其他"是兜底桶，算进来会让覆盖数虚高，反而误杀正常整理。
    """
    tracked = set(CORE_APPEARANCE_CATEGORIES) | {"配饰"}
    return {classify_appearance_card(card) for card in cards} & tracked


class VisionError(RuntimeError):
    """视觉通道调用失败。"""


def _image_content(image_bytes: bytes) -> dict[str, Any]:
    return {
        "type": "image",
        "image_format": sniff_mime_type(image_bytes).removeprefix("image/"),
        "image_base64": base64.b64encode(image_bytes).decode("ascii"),
    }


async def generate_with_host(
    generate: GenerateFn,
    *,
    prompt: str,
    image_bytes: bytes,
    task_name: str,
    model_name: str,
    timeout_seconds: float,
    max_tokens: int,
    temperature: float,
    max_upload_bytes: int = DEFAULT_UPLOAD_BYTES,
) -> str:
    """经 Host 的视觉任务跑一次请求，返回文本。

    ``task_name`` 与 ``model_name`` 是**两个不同的参数**（Host 1.2.5 起）：前者是任务
    别名（如 ``vlm``），后者是具体模型名。两者都只在非空时才传，否则会去查一个不存在
    的模型并直接失败。
    """
    upload, _ = prepare_upload(image_bytes, max_bytes=max_upload_bytes)
    kwargs: dict[str, Any] = {
        "prompt": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            _image_content(upload),
        ]}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if task_name:
        kwargs["task_name"] = task_name
    if model_name:
        kwargs["model_name"] = model_name
    try:
        response = await asyncio.wait_for(generate(**kwargs), timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise VisionError(f"视觉请求超时（{timeout_seconds:.0f}s）") from exc
    except Exception as exc:
        raise VisionError(f"视觉请求失败：{type(exc).__name__}: {exc}") from exc
    if not isinstance(response, dict) or not response.get("success", False):
        detail = ""
        if isinstance(response, dict):
            detail = str(response.get("error") or response.get("message") or "")
        raise VisionError(f"视觉任务未成功返回{('：' + detail) if detail else ''}")
    return str(response.get("response") or "").strip()


async def describe_image(
    generate: GenerateFn,
    *,
    image_bytes: bytes,
    task_name: str,
    model_name: str,
    timeout_seconds: float,
    max_tokens: int = 220,
    temperature: float = 0.0,
    max_upload_bytes: int = DEFAULT_UPLOAD_BYTES,
) -> str:
    """只取通用图片描述（沿用 MaiBot 内置提示词口径）。"""
    text = await generate_with_host(
        generate,
        prompt=HOST_DESCRIPTION_PROMPT,
        image_bytes=image_bytes,
        task_name=task_name,
        model_name=model_name,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        temperature=temperature,
        max_upload_bytes=max_upload_bytes,
    )
    return _strip_code_fence(text)[:160]


async def identify_image(
    *,
    provider: str,
    image_bytes: bytes,
    catalog: Sequence[dict],
    max_candidates: int = 3,
    generate: GenerateFn | None = None,
    task_name: str = "vlm",
    model_name: str = "",
    api_key: str = "",
    base_url: str = "",
    timeout_seconds: float = 25.0,
    max_tokens: int = 700,
    temperature: float = 0.0,
    max_upload_bytes: int = DEFAULT_UPLOAD_BYTES,
) -> VisionResult | None:
    """识别图片并从本地库匹配候选。解析失败返回 None（不抛）。"""
    prompt = build_identify_prompt(catalog, max_candidates)
    if provider == "host":
        if generate is None:
            raise VisionError("host 通道需要注入 Host 生成函数")
        content = await generate_with_host(
            generate,
            prompt=prompt,
            image_bytes=image_bytes,
            task_name=task_name,
            model_name=model_name,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            temperature=temperature,
            max_upload_bytes=max_upload_bytes,
        )
    elif provider in {"gemini", "openai"}:
        content = await _call_direct(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            prompt=prompt,
            image_bytes=image_bytes,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            temperature=temperature,
            max_upload_bytes=max_upload_bytes,
        )
    else:
        raise VisionError(f"不支持的视觉提供方：{provider}")
    return parse_vision_result(content)


# ------------------------------------------------------ 区分性证据（服装 / 饰品）

#: 服装与饰品是"换个角色就不同"的特征载体。
#:
#: 真机实测（09-18 09:20）：一张蓝发金瞳的**女仆装**角色，模型的证据只有
#: 「浅蓝色长发」「金色瞳孔」两条**共有**特征（库里有 10 个蓝发角色时它们同样吻合），
#: 服装明显不同却把 conflicts 留空，断言这是库内那个穿蓝制服白百褶裙的角色。
#: 发色瞳色吻合说明不了任何事，所以贴标签这条路上必须要求至少一条区分性特征。
#: 词表对应 ``build_appearance_prompt`` 要求的四类里的后两类。
DISTINCTIVE_MARKERS = (
    # 服装
    "衣", "服", "裙", "裤", "装", "鞋", "袜", "帽", "领", "结", "带", "袍", "披",
    # 饰品
    "饰", "环", "冠", "耳", "项链", "挂", "徽", "链", "簪", "戒",
)


def has_distinctive_evidence(texts: Sequence[str]) -> bool:
    """证据里是否包含服装 / 饰品这类"区分性"特征。

    只靠发色、瞳色认人，等于把库里所有同色系的角色混成一个人。纯函数，可脱机单测。
    """
    return any(marker in str(text) for text in texts for marker in DISTINCTIVE_MARKERS)


# ------------------------------------------------------------ 未标注候选的救援

#: 一条 evidence 与一张外观卡达到这个重合度，就算"说的是同一条特征"。
CARD_MATCH_SIMILARITY = 0.5
#: 至少对上这么多条卡片，才认为模型其实指向了库内角色。
CARD_MATCH_MIN_HITS = 2

_PUNCTUATION = re.compile(r"[\s，。、；：,.;:·\-—~～！？!?（）()\[\]【】「」『』“”\"'·]+")


def _bigrams(text: str) -> "set[str]":
    """去标点后的 2-gram 集合。中文用字符 2-gram 比按词切更稳，且不需要分词依赖。"""
    clean = _PUNCTUATION.sub("", str(text or ""))
    if len(clean) < 2:
        return {clean} if clean else set()
    return {clean[index:index + 2] for index in range(len(clean) - 1)}


def _similarity(left: str, right: str) -> float:
    a, b = _bigrams(left), _bigrams(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def rescue_unlabeled_candidates(
    result: VisionResult,
    catalog: Sequence[dict],
    *,
    min_hits: int = CARD_MATCH_MIN_HITS,
    min_similarity: float = CARD_MATCH_SIMILARITY,
) -> "tuple[VisionResult, str]":
    """把"逐条抄了库内画像、却忘了填名字"的候选救回来。

    真机实测（2026-09-18）：模型把某个角色的 5 条外观卡**逐字**抄成 evidence，
    ``conflicts`` 也是空数组——它分明认出来了，却填了 ``kind=unknown``、``name=""``。
    整条候选就因为这**一个字段没填**而作废，表现是"图片与画像高度吻合，却永远贴不上"。

    所以这里不再把判定权全交给模型的格式自觉：evidence 是模型自己抄回来的库内文本，
    拿它跟 catalog 的外观卡做确定性比对，就能把这种情况判回库内角色。

    仍然要求 ``conflicts`` 为空——模型说"有矛盾"就不能救；命中条数也要够，
    免得拿一两条泛泛的特征（"长发""少女"）乱认人。

    纯函数，可脱机单测。返回 ``(新结果, 说明)``，说明为空表示没有需要救援的候选。
    """
    known = {
        str(item.get("name") or "").strip(): [str(card) for card in (item.get("appearance_cards") or ())]
        for item in catalog
        if str(item.get("name") or "").strip()
    }
    if not known:
        return result, ""
    changed: list[VisionCandidate] = []
    rescued: list[str] = []
    for candidate in result.candidates:
        if candidate.usable_name or candidate.conflicts:
            changed.append(candidate)
            continue
        # 情形一：名字逐字就在库里，只是 kind 没标对。
        if candidate.name and candidate.name in known:
            changed.append(replace(candidate, kind="private"))
            rescued.append(f"{candidate.name}（名字逐字在库，kind 标错）")
            continue
        # 情形二：名字空着，但 evidence 逐条对得上某个角色的外观卡。
        if len(candidate.evidence) < min_hits:
            changed.append(candidate)
            continue
        best_name, best_hits = "", 0
        for name, cards in known.items():
            hits = sum(
                1 for text in candidate.evidence
                if any(_similarity(text, card) >= min_similarity for card in cards)
            )
            if hits > best_hits:
                best_name, best_hits = name, hits
        if best_name and best_hits >= min_hits:
            changed.append(replace(candidate, kind="private", name=best_name))
            rescued.append(f"{best_name}（{best_hits} 条特征对上外观卡）")
        else:
            changed.append(candidate)
    if not rescued:
        return result, ""
    return replace(result, candidates=tuple(changed)), "按外观卡逐条比对救回：" + "、".join(rescued)


async def build_appearance_cards(
    *,
    provider: str,
    image_bytes: bytes,
    generate: GenerateFn | None = None,
    task_name: str = "vlm",
    model_name: str = "",
    api_key: str = "",
    base_url: str = "",
    timeout_seconds: float = 25.0,
    max_tokens: int = 700,
    max_upload_bytes: int = DEFAULT_UPLOAD_BYTES,
    existing_cards: "Sequence[str]" = (),
) -> list[str]:
    """从一张图抽取稳定外观卡（管理员建卡用）。

    ``existing_cards`` 是库里已有的卡片，会一并进提示词，让模型只补新特征——不带它的话，
    第二张图必然把"浅蓝色长发"再写一遍，攒到第十张就全是变体。
    """
    prompt = build_appearance_prompt(existing_cards)
    if provider == "host":
        if generate is None:
            raise VisionError("host 通道需要注入 Host 生成函数")
        content = await generate_with_host(
            generate, prompt=prompt, image_bytes=image_bytes, task_name=task_name,
            model_name=model_name, timeout_seconds=timeout_seconds, max_tokens=max_tokens,
            temperature=0.0, max_upload_bytes=max_upload_bytes,
        )
    elif provider in {"gemini", "openai"}:
        content = await _call_direct(
            provider=provider, api_key=api_key, base_url=base_url, model_name=model_name,
            prompt=prompt, image_bytes=image_bytes, timeout_seconds=timeout_seconds,
            max_tokens=max_tokens, temperature=0.0, max_upload_bytes=max_upload_bytes,
        )
    else:
        raise VisionError(f"不支持的视觉提供方：{provider}")

    payload = json.loads(_extract_json_object(content))
    values = payload.get("appearance_cards") if isinstance(payload, dict) else None
    if not isinstance(values, list):
        raise VisionError("外观卡响应缺少 appearance_cards")
    cards = [str(item).strip() for item in values if str(item).strip()]
    if len(cards) < 2:
        raise VisionError("外观卡不足两条，无法建卡")
    return cards[:5]


async def _generate_text_with_host(
    generate: GenerateFn,
    *,
    prompt: str,
    task_name: str,
    model_name: str,
    timeout_seconds: float,
    max_tokens: int,
    temperature: float = 0.0,
) -> str:
    """纯文本的 Host 调用（外观卡整理这类不需要图片的场景）。

    不复用 ``generate_with_host``：那个函数从签名到实现都绑着图片载荷，为了纯文本调用
    去给它加分支，反而会让"带图/不带图"两条路都变难读。
    """
    kwargs: dict[str, Any] = {
        "prompt": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if task_name:
        kwargs["task_name"] = task_name
    if model_name:
        kwargs["model_name"] = model_name
    try:
        response = await asyncio.wait_for(generate(**kwargs), timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise VisionError(f"整理请求超时（{timeout_seconds:.0f}s）") from exc
    except Exception as exc:
        raise VisionError(f"整理请求失败：{type(exc).__name__}: {exc}") from exc
    if not isinstance(response, dict) or not response.get("success", False):
        detail = ""
        if isinstance(response, dict):
            detail = str(response.get("error") or response.get("message") or "")
        raise VisionError(f"整理任务未成功返回{('：' + detail) if detail else ''}")
    return str(response.get("response") or "").strip()


async def compress_appearance_cards(
    *,
    cards: "Sequence[str]",
    provider: str,
    generate: GenerateFn | None = None,
    task_name: str = "vlm",
    model_name: str = "",
    timeout_seconds: float = 90.0,
    max_tokens: int = 900,
) -> "tuple[list[str] | None, str]":
    """把外观卡整理成更少、不重复的一组。返回 ``(整理结果, 说明)``。

    结果是"整理"而不是"重写"：失败的代价必须只是白跑一次，绝不能把用户攒的卡片弄丢，
    所以这里**不抛异常**——调用方拿到 None 就保持原样。

    说明那一项写清**为什么没采用**。这很重要：真机第一次跑时超时了（模型耗时 56.8s，
    而当时超时设的是 45s），但日志只报"未生效"，看不出是超时、跳过还是模型不听话——
    三种情况的处理方式完全不同（调超时 / 换模型 / 改提示词）。

    默认超时给到 90s：整理是纯文本任务但输出有十多条中文，遇到慢模型（实测 56.8s）
    45s 会直接掐掉。

    只走 ``host`` 通道：直连通道（gemini/openai）的密钥与模型是给视觉准备的，为了一个
    可选增强去各写一份纯文本实现不划算，跳过即可（调用方照常保留原卡片）。
    """
    values = [str(card).strip() for card in cards if str(card).strip()]
    if len(values) < COMPRESS_MIN_CARDS:
        return None, f"卡片不足 {COMPRESS_MIN_CARDS} 条，跳过"
    if provider != "host":
        return None, "仅 host 通道提供整理，跳过"
    if generate is None:
        return None, "host 生成函数不可用，跳过"
    try:
        content = await _generate_text_with_host(
            generate,
            prompt=build_compress_prompt(values),
            task_name=task_name,
            model_name=model_name,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
        )
    except VisionError as exc:
        return None, str(exc)
    try:
        payload = json.loads(_extract_json_object(content))
    except (ValueError, json.JSONDecodeError):
        return None, "模型输出无法解析为 JSON"
    raw = payload.get("appearance_cards") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return None, "模型输出缺少 appearance_cards 字段"
    merged = [str(item).strip() for item in raw if str(item).strip()]
    # 结果明显不合理时宁可不采用：条数太少说明它把造型差异也合掉了，没减少说明没干活。
    if len(merged) < 3:
        return None, f"整理后只剩 {len(merged)} 条，疑似把造型差异也合掉了，未采用"
    if len(merged) >= len(values):
        return None, f"整理后条数没减少（{len(values)} → {len(merged)}），未采用"
    # 跨类别合并是**静默的灾难**：真机把 12 条压成 3 条"每张图一条"的大杂烩，每条同时讲
    # 发色/眼睛/服装，整库只剩"眼睛"一类，连缺口提示都反过来误报缺两类。
    # 这里用**类别覆盖数**兜底——确定性判据，不依赖模型守规矩。
    kinds_before, kinds_after = _covered_kinds(values), _covered_kinds(merged)
    if len(kinds_after) < len(kinds_before):
        return None, (
            f"整理后类别覆盖从 {len(kinds_before)} 类降到 {len(kinds_after)} 类"
            f"（剩：{'、'.join(sorted(kinds_after)) or '无法归类'}），"
            "疑似把多类特征拼进了同一条，未采用"
        )
    return merged[:APPEARANCE_CARD_LIMIT], ""


def parse_vision_result(content: str) -> VisionResult | None:
    """从模型输出里解析识别结果。任何格式问题都返回 None 而不是抛。"""
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        payload = json.loads(_extract_json_object(content))
    except (ValueError, json.JSONDecodeError):
        return None
    result = VisionResult.from_dict(payload)
    if result is None:
        return None
    # 留下原始输出：候选被丢弃时，"为什么丢"必须能回看模型到底写了什么。
    return replace(result, raw=str(content).strip()[:800])


def describe_vision_dropouts(result: VisionResult) -> str:
    """说明"模型没给出可用候选"是哪一类。

    ``VisionCandidate.usable_name`` 是三重条件的合取，任何一条不满足都返回空串——
    于是四种**处理方式完全不同**的原因被压成同一句"没有匹配"：

    * 模型压根没给候选 → 该看模型输出与提示词；
    * 给了但 ``kind`` 不是 private → 模型认为图里不是库里的人；
    * 给了但带了 conflicts → 该核对画像与图片差异，可能需要补卡；
    * 给了但 evidence 不足两条 → 图片信息量或提示词问题。

    合并成一句的代价是：用户会去改没错的那一环。所以这里逐条拆开报。

    纯函数，可脱机单测。
    """
    if not result.candidates:
        return "模型没有给出任何候选"
    reasons: list[str] = []
    for item in result.candidates:
        label = item.name or "（无名）"
        if item.kind != "private":
            reasons.append(f"「{label}」未被认定为库内角色（kind={item.kind}）")
        elif item.conflicts:
            reasons.append(f"「{label}」被判有冲突特征：{'；'.join(item.conflicts[:2])}")
        elif len(item.evidence) < 2:
            reasons.append(f"「{label}」可见证据不足 2 条（给了 {len(item.evidence)} 条）")
        else:
            reasons.append(f"「{label}」可用名称为空")
    return "；".join(reasons)


def _strip_code_fence(content: str) -> str:
    text = str(content or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _extract_json_object(content: str) -> str:
    """从可能带 ``` 包裹或前后废话的输出里抠出 JSON 对象。"""
    text = _strip_code_fence(content)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("模型未返回 JSON 对象")
    return text[start:end + 1]


async def _call_direct(
    *,
    provider: str,
    api_key: str,
    base_url: str,
    model_name: str,
    prompt: str,
    image_bytes: bytes,
    timeout_seconds: float,
    max_tokens: int,
    temperature: float,
    max_upload_bytes: int,
) -> str:
    upload, _ = prepare_upload(image_bytes, max_bytes=max_upload_bytes)
    if provider == "gemini":
        url = base_url.rstrip("/") + f"/models/{model_name}:generateContent?key={api_key}"
        payload = {
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
            "contents": [{"role": "user", "parts": [
                {"text": prompt},
                {"inlineData": {
                    "mimeType": sniff_mime_type(upload),
                    "data": base64.b64encode(upload).decode("ascii"),
                }},
            ]}],
        }
        response = await asyncio.to_thread(_post_json, url, {}, payload, timeout_seconds)
        candidates = response.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise VisionError("Gemini 未返回 candidates")
        content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            raise VisionError("Gemini 未返回内容部分")
        return "".join(str(part.get("text") or "") for part in parts if isinstance(part, dict))

    data_url = f"data:{sniff_mime_type(upload)};base64,{base64.b64encode(upload).decode('ascii')}"
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
        ]}],
    }
    response = await asyncio.to_thread(
        _post_json, url, {"Authorization": f"Bearer {api_key}"}, payload, timeout_seconds
    )
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise VisionError("OpenAI 兼容接口未返回 choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise VisionError("OpenAI 兼容接口未返回文本内容")
    return message["content"]


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, method="POST",
                      headers={"Content-Type": "application/json", **headers})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        detail = exc.read(501).decode("utf-8", errors="replace")[:300]
        raise VisionError(f"视觉接口 HTTP {exc.code}：{detail}") from exc
    except URLError as exc:
        raise VisionError(f"视觉接口连接失败：{exc.reason}") from exc
    except TimeoutError as exc:
        raise VisionError("视觉接口超时") from exc
    if len(raw) > MAX_RESPONSE_BYTES:
        raise VisionError(f"视觉接口响应超过 {MAX_RESPONSE_BYTES} 字节")
    try:
        result = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VisionError(f"视觉接口响应不是合法 JSON：{exc}") from exc
    if not isinstance(result, dict):
        raise VisionError("视觉接口响应顶层不是对象")
    return result
