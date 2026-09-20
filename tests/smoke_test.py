#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""角色识别插件冒烟测试（FakeHost，不启动 MaiBot、不联网）。

跑法：
    python tests/smoke_test.py

退出码 0 = 全部通过。

这个文件里最值得保留的是**组件清单断言**：装饰器若被辅助方法"插队"，组件会被静默
注册到错误的函数上——名字还在、类型还在，只有 handler 变了。所以每个组件都要同时校验
``type`` 与 ``metadata.handler_name``，只查"存在性"是抓不到这种错的。
"""

import ast
import asyncio
import base64
import json
import sys
import traceback
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PLUGIN_DIR = TESTS_DIR.parent
for path in (str(PLUGIN_DIR), str(TESTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import fakehost  # noqa: E402
from fakehost import FakeHost, bind_context, build_context, get_default_config, load_plugin_module  # noqa: E402

#: 两张不同的合法 2x2 PNG（纯标准库构造，不依赖 Pillow）。
#: 用两张而不是一张，是为了让"多图错位"这类 bug 无处可藏——同一张图会命中识别缓存，
#: 掩盖掉替换片段与实际图片的对应关系。
PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR42mP4z8AARAwQCgAf7gP9Y167WwAAAABJRU5ErkJggg=="
)
PNG_1PX2 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAD0lEQVR42mNgaPgPQhAKACX2Bf0ZCSOMAAAAAElFTkSuQmCC"
)

#: 组件清单：(组件名, 类型, 处理器方法名)。总数与配置里的组件数必须完全一致。
EXPECTED_COMPONENTS = {
    # ── Hook
    "recognize_incoming_images": ("HOOK_HANDLER", "recognize_incoming_images"),
    "inject_character_context": ("HOOK_HANDLER", "inject_character_context"),
    "inject_character_context_planner": ("HOOK_HANDLER", "inject_character_context_planner"),
    # ── Tool（方法名与组件名一致）
    "recognize_image": ("TOOL", "recognize_image"),
    "query_character": ("TOOL", "query_character"),
    "search_character_library": ("TOOL", "search_character_library"),
    "reverse_lookup_image": ("TOOL", "reverse_lookup_image"),
    # ── Command（组件名是 ASCII，用户敲的是 pattern 里的中文，所以方法名 != 组件名）
    "status": ("COMMAND", "cmd_status"),
    "probe": ("COMMAND", "cmd_probe"),
    "character_add": ("COMMAND", "cmd_character_add"),
    "character_add_done": ("COMMAND", "cmd_character_add_done"),
    "character_add_cancel": ("COMMAND", "cmd_character_add_cancel"),
    "character_correct": ("COMMAND", "cmd_character_correct"),
    "character_list": ("COMMAND", "cmd_character_list"),
    "character_view": ("COMMAND", "cmd_character_view"),
    "character_persona": ("COMMAND", "cmd_character_persona"),
    "character_work": ("COMMAND", "cmd_character_work"),
    "character_relationship": ("COMMAND", "cmd_character_relationship"),
    "character_alias_add": ("COMMAND", "cmd_character_alias_add"),
    "character_alias_delete": ("COMMAND", "cmd_character_alias_delete"),
    "character_delete": ("COMMAND", "cmd_character_delete"),
    "rebuild_index": ("COMMAND", "cmd_rebuild_index"),
}


class Runner:
    """极简测试夹具：装载插件、注入假上下文与配置。"""

    def __init__(self, config_overrides=None, returns=None, embed_mode="batch"):
        self.module = load_plugin_module(PLUGIN_DIR)
        self.plugin = self.module.create_plugin()
        self.host = FakeHost("org.mai-mai.character-recognizer", returns=returns)
        self.host.embed_mode = embed_mode
        self.ctx = build_context("org.mai-mai.character-recognizer", rpc_call=self.host.rpc_call)
        config = get_default_config(type(self.plugin).config_model)
        _apply_overrides(config, config_overrides or {})
        bind_context(self.plugin, self.ctx, config)


def _apply_overrides(config: dict, overrides: dict) -> None:
    """按 "a.b.c" 路径覆盖默认配置。"""
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        node = config
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value


def _message(user_id: str = "10001", *, with_image: bool = True, text: str = "看看这个") -> dict:
    components = [{"type": "text", "data": text}]
    if with_image:
        components.append({
            "type": "image",
            "binary_data_base64": base64.b64encode(PNG_1PX).decode("ascii"),
        })
    return {
        "session_id": "stream-1",
        "processed_plain_text": "",
        "raw_message": components,
        "message_info": {"user_info": {"user_id": user_id}},
    }


# ═══════════════════════════════════════════════ 各项检查


def check_component_inventory(runner: Runner) -> list[str]:
    """组件清单：数量 + 类型 + 处理器名。专抓装饰器绑错人。"""
    components = runner.plugin.get_components()
    found = {item["name"]: item for item in components}
    failures: list[str] = []

    if len(components) != len(EXPECTED_COMPONENTS):
        failures.append(f"组件总数应为 {len(EXPECTED_COMPONENTS)}，实际 {len(components)}")
    missing = sorted(set(EXPECTED_COMPONENTS) - set(found))
    if missing:
        failures.append(f"缺少组件：{missing}")
    unexpected = sorted(set(found) - set(EXPECTED_COMPONENTS))
    if unexpected:
        failures.append(f"多出未预期的组件：{unexpected}")

    for name, (expected_type, expected_handler) in EXPECTED_COMPONENTS.items():
        item = found.get(name)
        if item is None:
            continue
        if item.get("type") != expected_type:
            failures.append(f"{name}: 类型应为 {expected_type}，实际 {item.get('type')}")
        handler = (item.get("metadata") or {}).get("handler_name")
        if handler != expected_handler:
            failures.append(
                f"{name}: handler 应为 {expected_handler}，实际 {handler!r}"
                "（装饰器被辅助方法插队时会这样）"
            )
    return failures


def check_command_patterns() -> list[str]:
    """命令的中文触发词必须真能匹配到——组件名是 ASCII，匹配全靠 pattern。"""
    module = load_plugin_module(PLUGIN_DIR)
    import re

    cases = [
        ("cmd_status", "/识别状态", "／状态"),
        ("cmd_probe", "/识别探测", "／探测"),
        ("cmd_character_add", "/角色添加 阿罗娜", "/角色添加 阿罗娜 蔚蓝档案 同伴"),
        ("cmd_character_add_cancel", "/取消角色添加", None),
        ("cmd_character_correct", "/识图修正 阿罗娜", None),
        ("cmd_character_list", "/角色列表", "／角色库"),
        ("cmd_character_view", "/查看角色 阿罗娜", "/查看人设 阿罗娜"),
        ("cmd_character_persona", "/设置人设 阿罗娜 来自蔚蓝档案的AI", None),
        ("cmd_character_work", "/设置作品 阿罗娜 蔚蓝档案", None),
        ("cmd_character_relationship", "/设置关系 阿罗娜 同伴", None),
        ("cmd_character_alias_add", "/添加别名 阿罗娜 アロナ", None),
        ("cmd_character_alias_delete", "/删除别名 阿罗娜 アロナ", None),
        ("cmd_character_delete", "/删除角色 阿罗娜", None),
        ("cmd_rebuild_index", "/重建索引", "／刷新索引"),
    ]
    failures: list[str] = []
    for method_name, *patterns in cases:
        method = getattr(module.CharacterRecognizerPlugin, method_name, None)
        if method is None:
            failures.append(f"找不到命令方法 {method_name}")
            continue
        info = getattr(method, "__maibot_component_info__", None)
        if info is None or not info.command_pattern:
            failures.append(f"{method_name} 没有 command_pattern")
            continue
        for pattern in patterns:
            if pattern is None:
                continue
            if not re.fullmatch(info.command_pattern, pattern):
                failures.append(f"{method_name} 的正则匹配不上 {pattern!r}")

    # 命令不能误吞普通聊天
    status = getattr(module.CharacterRecognizerPlugin.cmd_status, "__maibot_component_info__")
    for ordinary in ("你好呀", "/help", "识别状态是什么"):
        if re.fullmatch(status.command_pattern, ordinary):
            failures.append(f"cmd_status 误匹配普通文本 {ordinary!r}")
    return failures


def check_hook_rewrites_message(runner: Runner) -> list[str]:
    """Hook 必须同时改 raw_message 与 processed_plain_text，且不丢组件、不主动发消息。

    这条测试刻意断言**组件完整性**：只断言"raw_message 变了"是抓不到错位的——
    替换片段按图片顺序消费，若按全部组件建表，图片前插一个文字组件就会让第一张图
    拿到文字组件的结果，表现为整张图被替换掉、甚至消失。
    """
    failures: list[str] = []
    runner.plugin._repository.upsert(name="阿罗娜", work="蔚蓝档案", appearance_cards=["蓝白长发", "发光圆环"])
    message = _message()

    result = asyncio.run(runner.plugin.recognize_incoming_images(message=message))
    if not isinstance(result, dict) or result.get("action") != "continue":
        return [f"hook 返回值不对：{result!r}"]
    modified = (result.get("modified_kwargs") or {}).get("message") or {}
    rewritten = modified.get("raw_message")

    if rewritten == message.get("raw_message"):
        failures.append("raw_message 未被改写")
        return failures
    if not isinstance(rewritten, list):
        return ["raw_message 改写后不是列表"]

    # 文字组件必须原样保留在原位
    if not rewritten or rewritten[0] != message["raw_message"][0]:
        failures.append(f"首个文字组件被破坏了：{rewritten[:1]!r}")

    # 图片要么被替换成文本标签，要么原样保留；绝不能变成别的组件类型
    images_before = [c for c in message["raw_message"] if c.get("type") == "image"]
    images_after = [c for c in rewritten if isinstance(c, dict) and c.get("type") == "image"]
    labels = [
        c for c in rewritten
        if isinstance(c, dict) and c.get("type") == "text"
        and ("图片[" in str(c.get("data") or "") or "[图片：" in str(c.get("data") or ""))
    ]
    if len(images_after) + len(labels) < len(images_before):
        failures.append(
            f"图片被吞掉了：改写前 {len(images_before)} 张，改写后剩 "
            f"{len(images_after)} 张图 + {len(labels)} 条标签"
        )
    if not labels and len(images_after) != len(images_before):
        failures.append("既没有产出标签，图片数量也对不上")

    if "processed_plain_text" not in modified or modified["processed_plain_text"] == message["processed_plain_text"]:
        failures.append("processed_plain_text 未被同步改写（会被 message.process() 覆盖）")
    if runner.host.calls_of("send.text"):
        failures.append("入站识别不该主动发消息，但调用了 send.text")
    return failures


def check_image_replacement_alignment() -> list[str]:
    """两张图必须各自拿到自己的结论，不能错位。

    图片前/中夹着文字组件，且两张图内容不同（因此不会命中同一份缓存）——
    这样"替换片段按全部组件建表"的错误会在组件类型序列上立刻暴露。
    """
    failures: list[str] = []
    runner = Runner(config_overrides={"anime_trace.enabled": False})
    asyncio.run(runner.plugin.on_load())
    runner.plugin._repository.upsert(name="阿罗娜", appearance_cards=["蓝白长发", "发光圆环"])
    message = {
        "session_id": "stream-1",
        "processed_plain_text": "",
        "raw_message": [
            {"type": "text", "data": "第一句"},
            {"type": "image", "binary_data_base64": base64.b64encode(PNG_1PX).decode("ascii")},
            {"type": "text", "data": "中间一句"},
            {"type": "image", "binary_data_base64": base64.b64encode(PNG_1PX2).decode("ascii")},
        ],
        "message_info": {"user_info": {"user_id": "10001"}},
    }
    try:
        result = asyncio.run(runner.plugin.recognize_incoming_images(message=message))
        rewritten = ((result or {}).get("modified_kwargs") or {}).get("message", {}).get("raw_message")
        if not isinstance(rewritten, list):
            return ["多图消息改写失败"]

        texts = [str(c.get("data")) for c in rewritten if isinstance(c, dict) and c.get("type") == "text"]
        if "第一句" not in texts or "中间一句" not in texts:
            failures.append(f"原有文字被破坏或移位：{texts}")

        # 每张图要么原样保留、要么被换成一条文本标签，总数不能少
        tracked = [
            c for c in rewritten
            if isinstance(c, dict) and (
                c.get("type") == "image"
                or (c.get("type") == "text" and "图片[" in str(c.get("data") or ""))
            )
        ]
        if len(tracked) < 2:
            failures.append(f"两张图只对应上 {len(tracked)} 个结果，发生了错位或丢失")
        # 组件顺序：两个原始文字组件仍在，且第一个组件就是它
        if not rewritten or str(rewritten[0].get("data")) != "第一句":
            failures.append(f"首个组件被破坏：{rewritten[:1]!r}")
    finally:
        asyncio.run(runner.plugin.on_unload())
    return failures


def check_no_image_message_is_untouched(runner: Runner) -> list[str]:
    """纯文字消息必须原样放行，不能有副作用。"""
    result = asyncio.run(runner.plugin.recognize_incoming_images(message=_message(with_image=False)))
    if result is not None:
        return [f"无图片的消息不该被改写，实际返回 {result!r}"]
    return []


def check_multi_source_confirmation_labels_image() -> list[str]:
    """多源一致 → 自动贴标签。这条验证的是"融合框架真的支持 N 个源"。

    两个 adapter 都用桩替换，所以这里不联网。第二个源故意用一个未实现的名字，
    证明后补源不需要改融合层或插件主流程。
    """
    failures: list[str] = []
    runner = Runner(
        config_overrides={
            "saucenao.enabled": True,
            "vision.enabled": False,
        }
    )
    import sources as sources_module

    async def stub_anime_trace(image_bytes, config):
        from models import SourceHit
        return (SourceHit("anime_trace", raw_name="アロナ", work="蔚蓝档案", confident=True),)

    async def stub_saucenao(image_bytes, config):
        from models import SourceHit
        return (SourceHit("saucenao", raw_name="阿罗娜", work="蔚蓝档案", confident=True, score=93.0),)

    original = dict(sources_module.ADAPTERS)
    sources_module.ADAPTERS.update({"anime_trace": stub_anime_trace, "saucenao": stub_saucenao})
    try:
        asyncio.run(runner.plugin.on_load())
        runner.plugin._repository.upsert(
            name="阿罗娜", work="蔚蓝档案", relationship="同伴",
            aliases=["アロナ"], appearance_cards=["蓝白长发", "发光圆环"],
        )
        result = asyncio.run(runner.plugin._recognize(PNG_1PX))
    finally:
        sources_module.ADAPTERS.clear()
        sources_module.ADAPTERS.update(original)

    if "阿罗娜" not in result.label:
        failures.append(f"两个源一致时应贴上角色标签，实际 {result.label!r}")
    if "同伴" not in result.label:
        failures.append(f"标签应带关系后缀，实际 {result.label!r}")
    if result.fusion.tier != "TA":
        failures.append(f"应为 TA（多源确认）层，实际 {result.fusion.tier}")
    if "阿罗娜" not in result.injection:
        failures.append("知识块里应包含该角色的资料")
    if "ROLE_DATA" not in result.injection:
        failures.append("知识块必须用定界符把数据与指令分开")
    return failures


def check_single_source_does_not_label() -> list[str]:
    """单源命中默认不贴标签：错贴一个人名比不贴更伤。"""
    runner = Runner(config_overrides={"vision.enabled": False})
    import sources as sources_module

    async def stub(image_bytes, config):
        from models import SourceHit
        return (SourceHit("anime_trace", raw_name="アロナ", work="蔚蓝档案", confident=True),)

    original = dict(sources_module.ADAPTERS)
    sources_module.ADAPTERS.update({"anime_trace": stub})
    try:
        asyncio.run(runner.plugin.on_load())
        runner.plugin._repository.upsert(name="阿罗娜", aliases=["アロナ"], appearance_cards=["蓝白长发"])
        result = asyncio.run(runner.plugin._recognize(PNG_1PX))
    finally:
        sources_module.ADAPTERS.clear()
        sources_module.ADAPTERS.update(original)

    failures: list[str] = []
    if result.label != "图片[未识别]":
        failures.append(f"单源命中不该自动贴标签，实际 {result.label!r}")
    if "阿罗娜" not in result.injection:
        failures.append("虽然不贴标签，但角色资料仍应注入供模型参考")
    return failures


def check_injection_is_idempotent(runner: Runner) -> list[str]:
    """模型请求 hook 每次 attempt/retry 都会触发，注入必须幂等。"""
    failures: list[str] = []
    runner.plugin._latest_images["stream-1:10001"] = (PNG_1PX, "图片[阿罗娜（同伴）]")
    runner.plugin._repository.upsert(name="阿罗娜", relationship="同伴", appearance_cards=["蓝白长发"])
    marker = runner.module.INJECT_MARKER

    first_kwargs = {"items": [], "item_schema_version": 1, "session_id": "stream-1"}
    first = asyncio.run(runner.plugin.inject_character_context(**first_kwargs))
    if not first:
        return ["第一次注入就失败了，说明知识块没构造出来"]
    items_after_first = first["modified_kwargs"]["items"]
    if len(items_after_first) != 1:
        failures.append(f"第一次注入后应有 1 条 item，实际 {len(items_after_first)}")
    if first["modified_kwargs"].get("item_schema_version") != 1:
        failures.append("item_schema_version 必须原样回显")
    if marker not in items_after_first[0]["parts"][0]["text"]:
        failures.append("注入内容必须带标记，否则幂等判断失效")
    if "logical_turn_id" not in items_after_first[0]["meta"]:
        failures.append("meta 必须含 logical_turn_id（可为 null），缺键会被 Host 拒掉")

    second = asyncio.run(runner.plugin.inject_character_context(**first["modified_kwargs"]))
    if second is not None:
        failures.append("第二次触发应因命中标记而跳过注入")
    return failures


def check_unsupported_item_schema_is_skipped(runner: Runner) -> list[str]:
    """收到更高版本的 item schema 宁可跳过，也不要注入畸形 item。"""
    runner.plugin._latest_images["stream-1:10001"] = (PNG_1PX, "图片[阿罗娜]")
    result = asyncio.run(runner.plugin.inject_character_context(
        items=[], item_schema_version=99, session_id="stream-1"
    ))
    if result is not None:
        return ["item_schema_version 高于支持版本时应跳过注入"]
    return []


def check_admin_gate() -> list[str]:
    """鉴权 7 组组合。配置侧与运行时侧必须走同一个归一函数。"""
    failures: list[str] = []
    cases = [
        # (配置里的 admin_ids, 触发者 user_id, is_local_operator, 是否应放行)
        (["123456789"], "123456789", False, True),
        (["qq:123456789"], "123456789", False, True),
        (["123456789"], "qq:123456789", False, True),
        (["qq:123456789"], "qq:123456789", False, True),
        ([" 123456789 "], "123456789", False, True),
        ([], "123456789", False, False),
        (["123456789"], "999", False, False),
        (["123456789"], "999", True, True),
    ]
    for admin_ids, user_id, is_local, should_pass in cases:
        runner = Runner(config_overrides={"library.admin_ids": admin_ids})
        asyncio.run(runner.plugin.on_load())
        try:
            allowed = runner.plugin._is_admin({"user_id": user_id, "is_local_operator": is_local})
        finally:
            asyncio.run(runner.plugin.on_unload())
        if allowed != should_pass:
            failures.append(
                f"admin_ids={admin_ids} user_id={user_id!r} local={is_local} → 应"
                f"{'放行' if should_pass else '拒绝'}，实际{'放行' if allowed else '拒绝'}"
            )
    return failures


def check_commands_reply_and_return_intercept_level() -> list[str]:
    """命令必须显式发回执；发送成功时 intercept_level=2。"""
    failures: list[str] = []
    runner = Runner(config_overrides={"library.admin_ids": ["10001"]})
    asyncio.run(runner.plugin.on_load())
    try:
        ok, reply, level = asyncio.run(runner.plugin.cmd_status(stream_id="stream-1", user_id="10001"))
        if not ok or not reply:
            failures.append("cmd_status 未返回可用回执")
        if level != 2:
            failures.append(f"发送成功时 intercept_level 应为 2，实际 {level}")
        if not runner.host.sent_texts:
            failures.append("cmd_status 没有显式发送回执（命令返回值不会自动发群）")

        # 非管理员：拒绝但仍要回执
        runner.host.reset()
        _, reply, level = asyncio.run(
            runner.plugin.cmd_character_list(stream_id="stream-1", user_id="999")
        )
        if "管理员" not in reply:
            failures.append(f"非管理员应被告知需要管理员，实际回执 {reply!r}")
        if level != 2:
            failures.append("拒绝时也应把回执发出去，而不是石沉大海")
    finally:
        asyncio.run(runner.plugin.on_unload())
    return failures


def check_character_lifecycle_commands() -> list[str]:
    """角色管理的写入路径：增改删别名都要落盘并刷新索引。"""
    failures: list[str] = []
    runner = Runner(config_overrides={"library.admin_ids": ["10001"]})
    asyncio.run(runner.plugin.on_load())
    try:
        kwargs = {"stream_id": "stream-1", "user_id": "10001"}

        asyncio.run(runner.plugin.cmd_character_persona(
            **kwargs, matched_groups={"name": "阿罗娜", "value": "来自蔚蓝档案的AI助手"}
        ))
        # 角色还不存在 → 应给出可读错误而不是抛
        if not any("不存在" in text for text in runner.host.sent_texts):
            failures.append("对不存在的角色设置人设时应回执可读错误")

        runner.host.reset()
        asyncio.run(runner.plugin.cmd_character_add(
            **kwargs, matched_groups={"name": "阿罗娜", "work": "蔚蓝档案", "relationship": "同伴"}
        ))
        pending = runner.plugin._pending_additions.get("stream-1:10001")
        if pending is None:
            failures.append("角色添加没有登记待处理项")
        elif pending.work != "蔚蓝档案" or pending.relationship != "同伴":
            failures.append(f"待处理项丢了作品/关系参数：{pending!r}")

        runner.plugin._pending_additions.clear()
        runner.plugin._repository.upsert(name="阿罗娜", relationship="同伴", appearance_cards=["蓝白长发", "发光圆环"])
        runner.host.reset()

        asyncio.run(runner.plugin.cmd_character_persona(
            **kwargs, matched_groups={"name": "阿罗娜", "value": "来自蔚蓝档案的AI助手"}
        ))
        asyncio.run(runner.plugin.cmd_character_alias_add(
            **kwargs, matched_groups={"name": "阿罗娜", "value": "アロナ"}
        ))
        view = asyncio.run(runner.plugin.cmd_character_view(**kwargs, matched_groups={"name": "アロナ"}))
        if "来自蔚蓝档案" not in view[1]:
            failures.append("按别名查看角色时读到的资料不完整")

        runner.host.reset()
        asyncio.run(runner.plugin.cmd_character_delete(**kwargs, matched_groups={"name": "阿罗娜"}))
        if len(runner.plugin._repository) != 0:
            failures.append("删除角色后库里仍有残留")
    finally:
        asyncio.run(runner.plugin.on_unload())
    return failures


def check_no_undefined_self_calls() -> list[str]:
    """扫出"被调用但类里没定义"的 ``self._xxx``。

    `check_plugin.py` 只做静态结构检查，抓不到这种漏改；冒烟测试又可能被桩掩盖。
    2026-09-18 真机就栽在这里：重构时漏回一个 ``_cards_from_component``，每张图都抛
    AttributeError——而当时的桩**正好打在那个方法上**，测试全绿。这条检查让同类问题
    在本地就现形。
    """
    failures: list[str] = []
    tree = ast.parse((PLUGIN_DIR / "plugin.py").read_text(encoding="utf-8"))
    for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
        defined = {
            node.name
            for node in cls.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assigned = {
            node.attr
            for node in ast.walk(cls)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and isinstance(node.ctx, ast.Store)
            and node.attr.startswith("_")
        }
        called = {
            node.func.attr
            for node in ast.walk(cls)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr.startswith("_")
        }
        for name in sorted(called - defined - assigned):
            failures.append(f"{cls.name} 调用了未定义的 self.{name}()")
    return failures


def check_character_add_collects_multiple_batches() -> list[str]:
    """连续登记：多批图的卡片必须并到同一个角色上，最后由 /结束角色添加 汇总。

    这条路径替代了"一张图建卡 + 反复 /识图修正"。要验证的核心是 ``upsert`` 的幂等合并
    真把多批卡并到一起了，以及登记状态在结束后确实被清掉——否则下一张普通图会被误吃。
    """
    failures: list[str] = []
    runner = Runner(config_overrides={"library.admin_ids": ["10001"]})
    asyncio.run(runner.plugin.on_load())
    try:
        plugin = runner.plugin
        admin = {"stream_id": "stream-1", "user_id": "10001"}
        asyncio.run(plugin.cmd_character_add(**admin, matched_groups={"name": "鸣澜"}))
        if plugin._pending_additions.get("stream-1:10001") is None:
            return ["角色添加没有登记待处理项"]

        batches = [
            ["浅蓝色长发，刘海整齐覆盖额头", "金色瞳孔，眼神明亮清晰"],
            ["深蓝色短袖制服上衣，领口有白色荷叶边", "白色百褶短裙，搭配白色中筒袜和黑色玛丽珍鞋"],
            ["黑白女仆装，白色蕾丝边女仆帽"],
        ]
        queue = list(batches)

        async def fake_cards(image_bytes: bytes, existing=()) -> list[str]:
            # 桩要打在**最底层**的抽卡上，让 _cards_from_component 的真实实现照常执行。
            # 早先这一行打的是 _cards_from_component 本身，结果那个方法被漏删后测试仍然全绿、
            # 真机第一张图就 AttributeError——桩把要测的东西整个替换掉了，等于没测。
            return queue.pop(0) if queue else []

        plugin._cards_from_bytes = fake_cards  # type: ignore[method-assign]
        message = {
            "session_id": "stream-1",
            "message_info": {"user_info": {"user_id": "10001"}},
            # 组件必须带真实可解的载荷：_cards_from_component 会先解析它再往下传
            "raw_message": [
                {"type": "image", "base64": base64.b64encode(PNG_1PX).decode("ascii")}
            ],
        }

        runner.host.reset()
        for _ in batches:
            asyncio.run(plugin.recognize_incoming_images(message=message))

        pending = plugin._pending_additions.get("stream-1:10001")
        if pending is None:
            failures.append("收集过程中登记状态丢了")
        elif pending.images != len(batches):
            failures.append(f"累计张数不对：{pending.images} != {len(batches)}")

        character = plugin._repository.find_name("鸣澜")
        total = len(character.appearance_cards) if character else 0
        expected = sum(len(batch) for batch in batches)
        if total != expected:
            failures.append(f"多批外观卡没有合并到同一角色：{total} != {expected}")

        reply = asyncio.run(plugin.cmd_character_add_done(**admin))
        if f"外观卡 {expected} 条" not in reply[1]:
            failures.append(f"结束登记的回执没有汇总卡片：{reply[1]}")
        if "发色发型" not in reply[1]:
            failures.append(f"汇总没有按类别分组：{reply[1]}")
        if plugin._pending_additions:
            failures.append("结束登记后待处理状态没有清掉")
    finally:
        asyncio.run(runner.plugin.on_unload())
    return failures


def check_incremental_card_extraction_gets_existing_cards() -> list[str]:
    """连续收集时，抽卡必须把库里已有的卡片一起带上。

    不带的话第二张图必然把"浅蓝色长发"再写一遍——真机 15 条里有一半是这么攒出来的。
    桩打在 ``_cards_from_bytes``（底层），顺带验证 ``_cards_from_component`` 把它透传下来了。
    """
    failures: list[str] = []
    runner = Runner(config_overrides={"library.admin_ids": ["10001"]})
    asyncio.run(runner.plugin.on_load())
    try:
        plugin = runner.plugin
        plugin._repository.upsert(name="鸣澜", appearance_cards=["已有的甲", "已有的乙"])
        asyncio.run(plugin.cmd_character_add(
            stream_id="stream-1", user_id="10001", matched_groups={"name": "鸣澜"},
        ))

        seen: list[tuple[str, ...]] = []

        async def fake_cards(image_bytes: bytes, existing=()) -> list[str]:
            seen.append(tuple(existing))
            return [f"新特征 {len(seen)}"]

        plugin._cards_from_bytes = fake_cards  # type: ignore[method-assign]
        asyncio.run(plugin.recognize_incoming_images(message={
            "session_id": "stream-1",
            "message_info": {"user_info": {"user_id": "10001"}},
            "raw_message": [
                {"type": "image", "base64": base64.b64encode(PNG_1PX).decode("ascii")}
            ],
        }))

        if not seen:
            failures.append("抽卡没有被调用")
        elif seen[0] != ("已有的甲", "已有的乙"):
            failures.append(f"抽卡没有带上已有卡片：{seen[0]}")
    finally:
        asyncio.run(runner.plugin.on_unload())
    return failures


def check_finish_compresses_cards_and_keeps_data_on_failure() -> list[str]:
    """结束登记时整理一遍：**在后台跑**、写回是**替换**、失败时卡片原样不动。

    整理是慢活（模型 30~55s），回执不该陪着等——所以它被丢到后台，完成后另发一条通知。
    """
    failures: list[str] = []
    runner = Runner(config_overrides={"library.admin_ids": ["10001"]})
    asyncio.run(runner.plugin.on_load())
    try:
        plugin = runner.plugin
        cards = [f"浅蓝色长发变体 {index}" for index in range(6)] + [
            "金色瞳孔", "深蓝色制服上衣", "白色百褶裙", "黑色玛丽珍鞋",
        ]
        admin = {"stream_id": "stream-1", "user_id": "10001"}
        good = {
            "success": True, "model": "fake",
            "response": (
                '{"appearance_cards":["浅蓝色长发（齐刘海、侧边黑饰）","金色瞳孔",'
                '"深蓝色制服上衣","白色百褶裙与黑色玛丽珍鞋"]}'
            ),
        }

        def arm() -> None:
            """把这个角色**重置**成"刚结束一轮登记"的状态。

            注意 ``upsert`` 是**合并**语义：光靠它重置不掉上一轮压缩的结果，所以后面还要
            ``set_appearance_cards`` 替换一次。走真实路径（``cmd_character_add``）而不是
            直接塞状态对象，顺带也验证了登记入口本身。
            """
            plugin._repository.upsert(name="鸣澜", appearance_cards=cards)
            plugin._repository.set_appearance_cards("鸣澜", cards)
            plugin._pending_additions.clear()
            asyncio.run(plugin.cmd_character_add(**admin, matched_groups={"name": "鸣澜"}))

        def finish() -> str:
            """跑完结束登记，并等后台整理收尾（真机上事件循环会继续跑，这里要模拟出来）。"""

            async def drive() -> str:
                reply = await plugin.cmd_character_add_done(**admin)
                await asyncio.gather(*list(plugin._tasks), return_exceptions=True)
                return reply[1]

            return asyncio.run(drive())

        arm()
        runner.host.returns["llm.generate"] = good
        text = finish()
        if "已在后台进行" not in text:
            failures.append(f"回执没有说明整理在后台进行：{text}")
        stored = list(plugin._repository.find_name("鸣澜").appearance_cards)
        if len(stored) != 4:
            failures.append(f"后台整理结果没有替换旧卡片：{stored}")

        arm()
        runner.host.returns["llm.generate"] = {
            "success": True, "response": "这不是 JSON", "model": "fake",
        }
        finish()
        after = list(plugin._repository.find_name("鸣澜").appearance_cards)
        if after != cards:
            failures.append(f"整理失败时卡片被改动了——绝不允许：{after}")

        # 整理期间用户又补了卡 → 必须放弃本次结果，不能拿旧快照覆盖
        arm()
        snapshot = list(plugin._known_cards("鸣澜"))
        runner.host.returns["llm.generate"] = good

        async def stale() -> None:
            plugin._repository.append_appearance_cards("鸣澜", ["整理期间新加的卡"])
            await plugin._compress_and_report("stream-1", "鸣澜", snapshot)

        asyncio.run(stale())
        after = list(plugin._repository.find_name("鸣澜").appearance_cards)
        if "整理期间新加的卡" not in after:
            failures.append(f"整理期间补的卡被旧快照覆盖了：{after}")
    finally:
        asyncio.run(runner.plugin.on_unload())
    return failures


def check_vision_timeout_layering() -> list[str]:
    """超时要分层，而且要跟得上真实的模型速度。

    背景：真机视觉模型单次 27~53s，而 `image_timeout_seconds` 曾是 25s——识别一半直接超时。
    建卡/补卡更不该和实时识别共用短超时（那时用户已经预期要等）。
    另外超时提示必须**可操作**且只出现一次：光说"超时"用户不知道该改哪。
    """
    failures: list[str] = []
    runner = Runner(config_overrides={"library.admin_ids": ["10001"]})
    asyncio.run(runner.plugin.on_load())
    try:
        plugin = runner.plugin
        conf = plugin.config.plugin
        if conf.card_timeout_seconds <= conf.image_timeout_seconds:
            failures.append("建卡超时应当比识别宽松")
        if conf.image_timeout_seconds < 45:
            failures.append(f"识别超时盖不住实测的 27~53s：{conf.image_timeout_seconds}")

        # 桩要打在**插件模块自己的命名空间**上：plugin.py 里写的是
        # `from vision import build_appearance_cards`，它持有自己的引用，
        # 改 vision 模块的属性对它毫无影响（改了等于没改）。
        plugin_module = sys.modules.get(type(plugin).__module__)
        if plugin_module is None or not hasattr(plugin_module, "build_appearance_cards"):
            failures.append("找不到插件模块里的 build_appearance_cards，无法验证建卡超时")
        else:
            seen: list[float] = []

            async def fake_build(**kwargs):
                seen.append(kwargs.get("timeout_seconds"))
                return ["甲", "乙"]

            original = plugin_module.build_appearance_cards
            plugin_module.build_appearance_cards = fake_build
            try:
                asyncio.run(plugin._cards_from_bytes(PNG_1PX))
            finally:
                plugin_module.build_appearance_cards = original
            if seen != [conf.card_timeout_seconds]:
                failures.append(f"建卡没有用 card_timeout_seconds：{seen}")

        first, second = plugin._vision_timeout_hint(), plugin._vision_timeout_hint()
        if not first:
            failures.append("超时提示为空")
        if second:
            failures.append("超时提示应当只给一次，否则刷屏")
        if "image_timeout_seconds" not in first:
            failures.append(f"提示要说明改哪个配置：{first}")
    finally:
        asyncio.run(runner.plugin.on_unload())
    return failures


def check_llm_calls_carry_explicit_rpc_timeout() -> list[str]:
    """LLM 调用必须**显式**带 RPC 超时。

    真机踩过：SDK 的 ``ctx.llm.generate`` 转发到 ``cap.call`` 时没传 ``timeout_ms``，于是
    固定吃 30 秒默认值，而视觉模型实测 54.9s——每次都在 30s 被切断
    （``RPCError: [E_TIMEOUT] 请求 cap.call 超时 (30000ms)``）。底层 ``call_capability``
    本身支持覆盖，所以插件必须自己带上。这条守着"别退回裸转发"。
    """
    failures: list[str] = []
    runner = Runner()
    asyncio.run(runner.plugin.on_load())
    try:
        runner.host.reset()
        asyncio.run(runner.plugin._generate(prompt="ping", task_name="utils"))
        paired = list(zip(runner.host.calls, runner.host.rpc_timeouts))
        timeouts = [timeout for (capability, _), timeout in paired if capability == "llm.generate"]
        if not timeouts:
            failures.append("llm.generate 没有被调用")
        elif any(timeout is None for timeout in timeouts):
            failures.append("llm.generate 没带 RPC 超时，会吃 SDK 的 30s 默认值")
        elif any(timeout < 60_000 for timeout in timeouts):
            failures.append(f"RPC 超时太小：{timeouts}")
    finally:
        asyncio.run(runner.plugin.on_unload())
    return failures


def check_compress_uses_general_task_not_vision_model() -> list[str]:
    """整理必须走通用小任务，不能挂在视觉大模型上。

    真机实测：挂在视觉任务上单次 54.9~56.8s，而 Host 的 ``cap.call`` RPC 硬超时只有 **30s**
    ——插件侧把超时调到 90s 也没用，先断的是 RPC 那一层。所以整理默认走 ``utils``；
    并且**不能再传视觉模型名**，那会把小任务压回大模型。
    """
    failures: list[str] = []
    runner = Runner(config_overrides={"library.admin_ids": ["10001"]})
    asyncio.run(runner.plugin.on_load())
    try:
        plugin = runner.plugin
        cards = [f"卡片 {index}" for index in range(10)]
        plugin._repository.upsert(name="鸣澜", appearance_cards=cards)
        plugin._pending_additions.clear()
        admin = {"stream_id": "stream-1", "user_id": "10001"}
        asyncio.run(plugin.cmd_character_add(**admin, matched_groups={"name": "鸣澜"}))

        runner.host.returns["llm.generate"] = {
            "success": True, "model": "fake",
            "response": json.dumps({"appearance_cards": ["甲", "乙", "丙"]}, ensure_ascii=False),
        }
        runner.host.reset()
        asyncio.run(plugin.cmd_character_add_done(**admin))

        calls = [args for name, args in runner.host.calls if name == "llm.generate"]
        if not calls:
            failures.append("整理没有发起调用")
        else:
            if calls[-1].get("task_name") != "utils":
                failures.append(f"整理没有走通用小任务：task_name={calls[-1].get('task_name')!r}")
            if calls[-1].get("model_name"):
                failures.append("整理不该传视觉模型名，那会把小任务压回大模型")
    finally:
        asyncio.run(runner.plugin.on_unload())
    return failures


def check_correct_uses_every_image_in_referenced_message() -> list[str]:
    """引用消息里有多张图时，/识图修正 必须**全部**用上。

    真机用法就是把同一角色的多张图合成一条消息发出来。只取第一张的话，管理员从回执
    上看不出被扔了——卡少几条而已，不会报错。
    """
    failures: list[str] = []
    runner = Runner(config_overrides={"library.admin_ids": ["10001"]})
    asyncio.run(runner.plugin.on_load())
    try:
        plugin = runner.plugin
        plugin._repository.upsert(name="鸣澜", appearance_cards=["浅蓝色长发，刘海整齐"])
        runner.host.returns["message.get_by_id"] = {
            "raw_message": [
                {"type": "image", "base64": base64.b64encode(PNG_1PX).decode("ascii")},
                {"type": "image", "base64": base64.b64encode(PNG_1PX2).decode("ascii")},
                {"type": "text", "data": "同一角色的两张图"},
            ]
        }
        seen: list[bytes] = []

        async def fake_cards(image_bytes: bytes, existing=()) -> list[str]:
            # 打桩：这里测的是"引用的图有没有都被处理"，不是外观卡抽取本身。
            seen.append(image_bytes)
            return [f"第 {len(seen)} 张图的外观"]

        plugin._cards_from_bytes = fake_cards  # type: ignore[method-assign]
        reply = asyncio.run(plugin.cmd_character_correct(
            stream_id="stream-1", user_id="10001", reply_to="msg-1",
            matched_groups={"name": "鸣澜"},
        ))

        if len(seen) != 2:
            failures.append(f"引用的两张图没有都被处理：只处理了 {len(seen)} 张")
        character = plugin._repository.find_name("鸣澜")
        total = len(character.appearance_cards) if character else 0
        if total != 3:
            failures.append(f"两张图的卡没有都入库：{total} != 3")
        if "2 张图" not in reply[1]:
            failures.append(f"回执没有说明用了几张图：{reply[1]}")
    finally:
        asyncio.run(runner.plugin.on_unload())
    return failures


def check_correct_without_reference_uses_recent_images() -> list[str]:
    """不引用消息时 /识图修正 要吃下"刚发的那几张"。

    这是管理员要的用法：连着发几张图，直接敲命令，不必先引用。所以最近图片记忆
    必须留得住一段历史——只留最后一张的话，"不引用"就退化成"只能补一张卡"。
    """
    failures: list[str] = []
    runner = Runner(config_overrides={"library.admin_ids": ["10001"]})
    asyncio.run(runner.plugin.on_load())
    try:
        plugin = runner.plugin
        plugin._repository.upsert(name="鸣澜", appearance_cards=["浅蓝色长发，刘海整齐"])
        key = "stream-1:10001"
        for data in (PNG_1PX, PNG_1PX2, PNG_1PX):
            plugin._remember_image(key, data, "图片[鸣澜]")  # 识别路径会这样留下历史

        if plugin._recent_images(key, limit=5, window=-1.0):
            failures.append("时间窗没生效：过期的图仍会被当成“刚发的”")

        seen: list[bytes] = []

        async def fake_cards(image_bytes: bytes, existing=()) -> list[str]:
            seen.append(image_bytes)
            return [f"第 {len(seen)} 张图的外观"]

        plugin._cards_from_bytes = fake_cards  # type: ignore[method-assign]
        reply = asyncio.run(plugin.cmd_character_correct(
            stream_id="stream-1", user_id="10001", matched_groups={"name": "鸣澜"},
        ))

        if len(seen) != 3:
            failures.append(f"不引用时应吃下最近 3 张图，实际 {len(seen)} 张")
        character = plugin._repository.find_name("鸣澜")
        total = len(character.appearance_cards) if character else 0
        if total != 4:
            failures.append(f"三张图的卡没有都入库：{total} != 4")
        if "3 张图" not in reply[1]:
            failures.append(f"回执没有说明用了几张图：{reply[1]}")
    finally:
        asyncio.run(runner.plugin.on_unload())
    return failures


def check_embedding_degradation_paths() -> list[str]:
    """embedding 不可用时必须降级而不是报错——这是真机最常见的情况。"""
    failures: list[str] = []
    for mode in ("single", "batch", "error"):
        runner = Runner(config_overrides={"vision.enabled": False}, embed_mode=mode)
        import sources as sources_module

        async def stub(image_bytes, config):
            return ()

        original = dict(sources_module.ADAPTERS)
        sources_module.ADAPTERS.update({"anime_trace": stub})
        try:
            asyncio.run(runner.plugin.on_load())
            for index in range(6):
                runner.plugin._repository.upsert(
                    name=f"角色{index}", appearance_cards=[f"特征{index}", "通用外观"]
                )
            try:
                result = asyncio.run(runner.plugin._recognize(PNG_1PX))
            except Exception as exc:  # noqa: BLE001
                failures.append(f"embed_mode={mode} 时识别抛异常：{type(exc).__name__}: {exc}")
                continue
            if not result.ok:
                failures.append(f"embed_mode={mode} 时识别未完成")
        finally:
            sources_module.ADAPTERS.clear()
            sources_module.ADAPTERS.update(original)
    return failures


def check_embed_adapter_against_host_shapes() -> list[str]:
    """验证插件对 ``llm.embed`` 返回值的解析能同时吃下两种形态。

    这是最容易在真机上悄悄退化的一环：解析失败不会抛错，只是静默降级成关键词检索，
    表面上"还能用"，实际上整个向量层已经死了。所以这里逐个形态断言 ``degraded``。
    """
    failures: list[str] = []
    from retrieval import EmbeddingIndex, RetrievalOptions, retrieve
    from models import Character, Query

    characters = [
        Character(character_id=f"char-{i}", name=f"角色{i}", appearance_cards=(f"特征{i}", "通用外观"))
        for i in range(4)
    ]
    query = Query(description="通用外观")

    for mode, expect_degraded in (("batch", False), ("single", False), ("error", True)):
        runner = Runner(embed_mode=mode)
        asyncio.run(runner.plugin.on_load())
        try:
            result = asyncio.run(retrieve(
                characters, query,
                options=RetrievalOptions(),
                embed=runner.plugin._embed_texts,
                index=EmbeddingIndex(),
            ))
        finally:
            asyncio.run(runner.plugin.on_unload())
        if bool(result.degraded) != expect_degraded:
            failures.append(
                f"embed_mode={mode} 时 degraded 应为 {expect_degraded}，实际 {result.degraded}"
            )
        if not result.selected:
            failures.append(f"embed_mode={mode} 时一个候选都没选出来")
    return failures


def check_unload_cleans_up() -> list[str]:
    failures: list[str] = []
    runner = Runner()
    asyncio.run(runner.plugin.on_load())
    runner.plugin._repository.upsert(name="阿罗娜", appearance_cards=["蓝白长发", "发光圆环"])
    runner.plugin._latest_images["stream-1:10001"] = (PNG_1PX, "图片[阿罗娜]")
    runner.plugin._cache.put("k", "v")
    runner.plugin._index.put(runner.plugin._repository.characters[0], [1.0] * 8)
    asyncio.run(runner.plugin.on_unload())

    if len(runner.plugin._cache) or len(runner.plugin._index) or runner.plugin._latest_images:
        failures.append("卸载后缓存/索引/图片记忆未清空")
    if runner.plugin._tasks:
        failures.append("卸载后仍有未回收的后台任务")
    return failures


def check_vector_cache_reuse() -> list[str]:
    """同一张图第二次识别应命中识别缓存，不再重复请求视觉与反查。"""
    failures: list[str] = []
    runner = Runner(config_overrides={"vision.enabled": False})
    import sources as sources_module

    calls = {"count": 0}

    async def stub(image_bytes, config):
        calls["count"] += 1
        return ()

    original = dict(sources_module.ADAPTERS)
    sources_module.ADAPTERS.update({"anime_trace": stub})
    try:
        asyncio.run(runner.plugin.on_load())
        runner.plugin._repository.upsert(name="阿罗娜", appearance_cards=["蓝白长发", "发光圆环"])
        asyncio.run(runner.plugin._recognize(PNG_1PX))
        after_first = calls["count"]
        asyncio.run(runner.plugin._recognize(PNG_1PX))
        if calls["count"] != after_first:
            failures.append("同一张图第二次识别没有命中缓存，重复请求了反查源")
    finally:
        sources_module.ADAPTERS.clear()
        sources_module.ADAPTERS.update(original)
    return failures


CHECKS = [
    ("自调方法都有定义（防漏改）", check_no_undefined_self_calls),
    ("组件清单（数量/类型/处理器名）", lambda: check_component_inventory(_shared_runner())),
    ("命令中文触发词正则", check_command_patterns),
    ("Hook 改写入站消息", lambda: check_hook_rewrites_message(_shared_runner())),
    ("多图替换片段不错位", check_image_replacement_alignment),
    ("纯文字消息原样放行", lambda: check_no_image_message_is_untouched(_shared_runner())),
    ("多源一致自动贴标签", check_multi_source_confirmation_labels_image),
    ("单源命中不贴标签", check_single_source_does_not_label),
    ("注入幂等与 item 结构", lambda: check_injection_is_idempotent(_shared_runner())),
    ("不支持 item schema 时跳过", lambda: check_unsupported_item_schema_is_skipped(_shared_runner())),
    ("管理员鉴权 8 组组合", check_admin_gate),
    ("命令回执与 intercept_level", check_commands_reply_and_return_intercept_level),
    ("角色管理命令读写", check_character_lifecycle_commands),
    ("角色登记连续收集多批图", check_character_add_collects_multiple_batches),
    ("识图修正吃最近图（不需引用）", check_correct_without_reference_uses_recent_images),
    ("抽卡带已有卡片（防重复）", check_incremental_card_extraction_gets_existing_cards),
    ("结束登记整理卡片（失败保原样）", check_finish_compresses_cards_and_keeps_data_on_failure),
    ("整理走通用小任务（不挂视觉模型）", check_compress_uses_general_task_not_vision_model),
    ("LLM 调用显式带 RPC 超时", check_llm_calls_carry_explicit_rpc_timeout),
    ("视觉超时分层与提示", check_vision_timeout_layering),
    ("embedding 三种返回形态", check_embedding_degradation_paths),
    ("embed 返回形态解析（batch/single/error）", check_embed_adapter_against_host_shapes),
    ("识别缓存复用", check_vector_cache_reuse),
    ("卸载清理", check_unload_cleans_up),
]

_runner_cache: dict[str, Runner] = {}


def _shared_runner() -> Runner:
    """多个检查共用一份已加载的插件实例，省去重复装载。

    反查源一律关掉：冒烟测试必须完全离线，不能因为网络抖动变成偶发失败。
    反查路径由下面几个用桩替换 adapter 的检查专门覆盖。

    视觉通道保持开启（``host`` 通道不需要密钥，FakeHost 会返回固定的假描述）——
    有描述才会真的改写消息，否则空描述 + 未识别时消息原样透传，hook 正确返回 None，
    这条测试就退化成"什么都没测"。
    """
    if "runner" not in _runner_cache:
        runner = Runner(config_overrides={"anime_trace.enabled": False})
        asyncio.run(runner.plugin.on_load())
        _runner_cache["runner"] = runner
    return _runner_cache["runner"]


def main() -> int:
    failures: dict[str, list[str]] = {}
    for name, check in CHECKS:
        try:
            problems = check() or []
        except Exception:  # noqa: BLE001
            problems = ["检查本身抛异常：\n" + traceback.format_exc(limit=4)]
        failures[name] = problems
        print(f"{'PASS' if not problems else 'FAIL'}  {name}")
        for problem in problems:
            print(f"      · {problem}")

    leaked = _shared_runner().host.find_binary_leaks()
    if leaked:
        failures["二进制泄漏守卫"] = leaked
        print("FAIL  二进制泄漏守卫")
        for item in leaked:
            print(f"      · {item}")
    else:
        print("PASS  二进制泄漏守卫（无 base64 泄漏）")

    bad = {name: items for name, items in failures.items() if items}
    print("-" * 60)
    if bad:
        print(f"冒烟未通过：{len(bad)}/{len(CHECKS) + 1} 项有问题")
        return 1
    print(f"冒烟全部通过：{len(CHECKS) + 1}/{len(CHECKS) + 1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
