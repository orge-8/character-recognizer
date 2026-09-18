# -*- coding: utf-8 -*-
"""入站消息改写与模型请求注入（不依赖 ctx）。

两个容易踩的点，都在这里收口：

1. ``maisaka.replyer.before_model_request`` 收的是 **``items``（Context Item 快照）**，
   不是 ``messages``。返回 ``{"messages": [...]}`` 会被完全忽略。
2. 该 hook 每次 attempt/retry 都会触发，所以注入必须**幂等**——否则重试几次就叠几遍。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Sequence

#: 支持的 item 快照 schema 版本。收到更高版本宁可跳过，也不注入畸形 item。
SUPPORTED_ITEM_SCHEMA_VERSION = 1


def new_item_id() -> str:
    return uuid.uuid4().hex


def build_system_item(text: str, *, item_id: str | None = None) -> dict[str, Any]:
    """构造一条 SystemMessageItem 快照。

    ``meta`` 里 ``logical_turn_id`` 这个键**必须存在**（可以为 null），
    缺键会被 Host 侧校验拒掉。
    """
    return {
        "item_type": "SystemMessageItem",
        "meta": {
            "item_id": item_id or new_item_id(),
            "logical_turn_id": None,
            "timestamp": datetime.now().isoformat(),
        },
        "parts": [{"type": "text", "text": text}],
    }


def _text_of(item: Any) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return ""
    chunks: list[str] = []
    parts = item.get("parts")
    if isinstance(parts, list):
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "text":
                chunks.append(str(part.get("text") or ""))
    if not chunks:
        content = item.get("content")
        if isinstance(content, str):
            chunks.append(content)
    return "".join(chunks)


def has_marker(container: Any, marker: str) -> bool:
    """判断注入内容是否已经在里面了（幂等检查）。"""
    if not marker:
        return False
    if isinstance(container, str):
        return marker in container
    if isinstance(container, (list, tuple)):
        return any(has_marker(item, marker) for item in container)
    if isinstance(container, dict):
        if marker in str(container.get("text") or ""):
            return True
        if marker in str(container.get("content") or ""):
            return True
        return has_marker(container.get("parts"), marker)
    return False


def inject_into_items(kwargs: dict[str, Any], text: str, marker: str) -> str:
    """往 ``items`` 追加一条 System 快照（原地修改 kwargs）。返回状态说明。"""
    version = kwargs.get("item_schema_version")
    if isinstance(version, int) and version > SUPPORTED_ITEM_SCHEMA_VERSION:
        return f"跳过注入：item_schema_version={version} 高于支持的 {SUPPORTED_ITEM_SCHEMA_VERSION}"
    items = kwargs.get("items")
    if not isinstance(items, list):
        return "跳过注入：kwargs 里没有 items 列表"
    if has_marker(items, marker):
        return "跳过注入：本次请求里已有注入标记（幂等）"
    items = list(items)
    items.append(build_system_item(text))
    kwargs["items"] = items
    return "已注入 items"


def inject_into_messages(kwargs: dict[str, Any], text: str, marker: str) -> str:
    """``messages`` 形态的兜底注入（部分版本/部分链路给的是这个）。"""
    messages = kwargs.get("messages")
    if not isinstance(messages, list):
        return "跳过注入：kwargs 里没有 messages 列表"
    if has_marker(messages, marker):
        return "跳过注入：messages 里已有注入标记（幂等）"
    messages = list(messages)
    messages.append({"role": "system", "content": text})
    kwargs["messages"] = messages
    return "已注入 messages"


def inject_into_prompt(kwargs: dict[str, Any], text: str, marker: str) -> str:
    """``prompt`` 是纯字符串时的兜底注入。"""
    prompt = kwargs.get("prompt")
    if not isinstance(prompt, str):
        return "跳过注入：kwargs 里没有 prompt 字符串"
    if has_marker(prompt, marker):
        return "跳过注入：prompt 里已有注入标记（幂等）"
    kwargs["prompt"] = f"{prompt}\n\n{text}" if prompt.strip() else text
    return "已注入 prompt"


def apply_injection(kwargs: dict[str, Any], text: str, marker: str) -> str:
    """按 items → messages → prompt 的顺序尝试注入，返回第一个成功的说明。

    顺序不能反：``items`` 是当前版本真正生效的那条路径，另两条只是别的版本/链路的兜底。
    """
    if not text:
        return "跳过注入：没有可注入的内容"
    for injector in (inject_into_items, inject_into_messages, inject_into_prompt):
        status = injector(kwargs, text, marker)
        if not status.startswith("跳过注入：kwargs 里没有"):
            return status
    return "跳过注入：三种载荷形态都不匹配"


def rewrite_components(
    raw_message: Sequence[Any],
    replacements: Sequence[Sequence[Any]],
) -> list[Any]:
    """按图片出现顺序把 image 组件替换成识别结果片段。

    非图片组件原样保留、顺序不变；``replacements`` 比图片数少时，多出来的图片原样留下
    （**声明式**：宁可不改，也不要错位替换成别人的标签）。
    """
    output: list[Any] = []
    image_index = 0
    for component in raw_message:
        is_image = isinstance(component, dict) and component.get("type") == "image"
        if is_image and image_index < len(replacements):
            output.extend(replacements[image_index])
            image_index += 1
            continue
        if is_image:
            image_index += 1
        output.append(component)
    return output


def flatten_text(components: Sequence[Any]) -> str:
    """把组件序列拍平成可读文本，用于同步更新 ``processed_plain_text``。"""
    chunks: list[str] = []
    for component in components:
        if not isinstance(component, dict):
            continue
        kind = component.get("type")
        if kind == "text":
            data = component.get("data")
            text = data if isinstance(data, str) else (
                str(data.get("text") or "") if isinstance(data, dict) else ""
            )
            if text:
                chunks.append(text)
        elif kind == "image":
            chunks.append("[图片]")
    return " ".join(chunks).strip()


def apply_rewrite(message: dict[str, Any], replacements: Sequence[Sequence[Any]]) -> dict[str, Any] | None:
    """改写消息的 ``raw_message`` 与 ``processed_plain_text``，返回新消息或 None。

    **两个字段都要改**：只改 ``raw_message`` 的话，后续 ``message.process()`` 会依据
    原始组件重新生成纯文本，把注入的标签覆盖掉。
    """
    raw_message = message.get("raw_message")
    if not isinstance(raw_message, list):
        return None
    rewritten = rewrite_components(raw_message, replacements)
    if rewritten == list(raw_message):
        return None
    modified = dict(message)
    modified["raw_message"] = rewritten
    original_text = str(message.get("processed_plain_text") or "")
    flattened = flatten_text(rewritten)
    if flattened or not original_text:
        modified["processed_plain_text"] = flattened
    return modified
