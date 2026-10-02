# -*- coding: utf-8 -*-
"""回执 / 状态 / 诊断 / 限额的**纯渲染**层：零 ctx、零 asyncio、零网络。

本模块从 plugin.py 拆出（重构路线 A）：所有函数都是「数据进、字符串出」的纯函数，
可以脱机单测。plugin.py 保留同名（含旧私有名）的 re-export，既有调用与测试不变。

纪律：本模块**不许 import plugin**（会形成循环导入）；需要的常量由本模块自持，
plugin.py 反向 re-export。
"""

from typing import Any, Sequence

try:
    from .models import UNRECOGNIZED_LABEL, Character, RecognitionResult, TIER_LOCAL
    from .textutil import CORE_APPEARANCE_CATEGORIES, group_appearance_cards
except ImportError:  # pragma: no cover - Runner 只把目录塞进 sys.path 时走这条
    from models import UNRECOGNIZED_LABEL, Character, RecognitionResult, TIER_LOCAL
    from textutil import CORE_APPEARANCE_CATEGORIES, group_appearance_cards

#: 视觉请求的 ``max_tokens`` **运行期下限**。
#:
#: 真机实录（09-28）：`describe` 走默认 220 触顶、`identify` 按配置的 700 也触顶，两次都是
#: "达到最大输出 token 限制"。要注意**推理模型的思考 token 也算在这个上限里**，所以
#: "描述最多 100 字 + 3 个候选"这种看起来几百 token 就够的输出，700 照样会被截断。
#:
#: 截断的代价是**静默失败**：JSON 断在 evidence 中途 → 整条候选校验作废，用户只看到
#: "识别没结果"。而多留上限不会多花钱（只按实际输出计费），所以这里设一个下限兜底，
#: 用户真机已有的 ``max_tokens = 700`` 不必改配置就能生效。
VISION_MAX_TOKENS_FLOOR = 1600

#: 单图超时最多只能占整条消息预算的这个比例。**必须严格小于 1**。
#:
#: 真机踩过的不变量破坏：用户照着插件自己的提示把 ``image_timeout_seconds`` 调到了 120，
#: 而 ``message_timeout_seconds`` 还是默认的 110——于是**单图超时永远不可能先触发**：
#:   13:12:13 收到图 → 13:14:03（+110s）"单条消息识别超预算"
#:                    → 13:14:18（+120s）那张图的任务才报"视觉请求超时"
#: 表现就是"超预算"日志刷屏、图片白等，而真正该看到的"视觉超时"晚 15 秒才出现。
#: 留出余量后，超时会先于预算触发，归因才清楚。
MESSAGE_BUDGET_RESERVE_RATIO = 0.8

#: 检索分的实测波动区间（2026-09-18 真机：同一张图、同一份卡片，分数在 0.34~0.40 之间漂，
#: 因为 description 由视觉模型每次现生成、措辞一变向量就变）。阈值落进这个区间会表现为
#: "同一张图时贴时不贴"——比设错更难查，因为它看起来像随机故障。所以状态里要喊出来。
SCORE_JITTER_BAND = (0.30, 0.42)


def effective_image_timeout(config: Any) -> float:
    """按整条消息预算夹取后的单图超时。**生效值的唯一真相来源。**

    做成模块级纯函数（而不是只留在插件方法里）是为了让 ``/识图状态`` 报的数与真正
    用的数**必然一致**。真机踩过"说的和做的不一样"：配置写着 120、实际按 110 的预算走，
    用户看完状态栏以为配置没生效，又去反复改配置。
    """
    configured = float(getattr(config.plugin, "image_timeout_seconds", 0.0) or 0.0)
    budget = float(getattr(config.plugin, "message_timeout_seconds", 0.0) or 0.0)
    if budget <= 0:
        return configured
    return max(1.0, min(configured, budget * MESSAGE_BUDGET_RESERVE_RATIO))


def effective_max_tokens(config: Any) -> int:
    """视觉请求实际使用的 ``max_tokens``（配置值与下限取大）。纯函数。"""
    configured = int(getattr(config.vision, "max_tokens", 0) or 0)
    return max(configured, VISION_MAX_TOKENS_FLOOR)


def names_from_label(label: str) -> "list[str]":
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


def appearance_summary(cards: "Sequence[str]") -> str:
    """一行类别计数，如「发色发型 2｜眼睛 1｜服装 3」。没有卡片时返回空串。"""
    groups = group_appearance_cards(cards)
    order = [*CORE_APPEARANCE_CATEGORIES, "配饰", "其他"]
    return "｜".join(f"{name} {len(groups[name])}" for name in order if name in groups)


def fallback_line(config: Any) -> str:
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


def render_appearance_cards(cards: "Sequence[str]", title: str = "") -> str:
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
    lines = [f"{head} {total} 条（{appearance_summary(cards)}）"]
    for name in present:
        lines.append(f"▸ {name}")
        lines.extend(f"  · {card}" for card in groups[name])
    missing = [name for name in CORE_APPEARANCE_CATEGORIES if name not in groups]
    if missing:
        lines.append(f"⚠ 缺「{'、'.join(missing)}」：再补一张能看到这些特征的图，核对时会稳很多")
    return "\n".join(lines)


def render_character(character: Character, field: str = "auto") -> str:
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
        return render_appearance_cards(character.appearance_cards)
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
        lines.append(render_appearance_cards(character.appearance_cards))
    if not character.persona and not character.work:
        lines.append("（还没填设定：可用 /设置人设、/设置作品 补充）")
    return "\n".join(lines)


def describe_result(result: RecognitionResult) -> str:
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


def render_source_error(detail: str) -> str:
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


def diagnose_local_confirm(diag: dict) -> str:
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


def render_diagnosis(diag: "dict | None") -> str:
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
        lines.append(f"    · {render_source_error(str(item))}")
    raw = str(diag.get("vision_raw") or "").strip().replace("\n", " ")
    if raw:
        lines.append(f"  视觉模型原始输出：{raw[:300]}" + ("…" if len(raw) > 300 else ""))
    if diag.get("reason"):
        lines.append(f"  依据：{diag['reason']}")
    lines.append("  本地库兜底通路：" + diagnose_local_confirm(diag))
    return "\n".join(lines)


def render_probe(summary: dict) -> str:
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


def status_lines(config: Any, repository: Any, source_configs: dict, cache_size: int,
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
        limits_line(config),
        "反查源：" + "、".join(
            f"{name}{'开' if cfg.enabled else '关'}" for name, cfg in source_configs.items()
        ),
        "单源自动贴标签：" + ("开" if config.fusion.auto_apply_single_source else "关（更安全）"),
        fallback_line(config.fusion),
        f"缓存：{cache_size} 条｜向量索引：{index_size} 条｜最近图片记忆：{image_count} 条",
        "向量检索：" + ("可用" if embed_available else "不可用，已降级为关键词检索"),
    ]
    return "\n".join(lines)


def limits_line(config: Any) -> str:
    """把**生效的**视觉限额写进状态，并标出它与配置不一致的地方。

    这一行的存在理由全是真机踩出来的：用户按提示把 ``image_timeout_seconds`` 调到 120、
    而 ``max_tokens`` 还留着 700，两处都是"配了但不生效"（前者被预算夹取、后者被下限抬走）。
    不把生效值报出来，用户看完状态栏会以为配置没生效，然后反复改配置——越改越远。
    """
    text = (
        f"视觉限额：单图超时 {effective_image_timeout(config):.0f}s"
        f"（总预算 {config.plugin.message_timeout_seconds:.0f}s）"
        f"｜max_tokens {effective_max_tokens(config)}"
    )
    notes: list[str] = []
    if effective_image_timeout(config) < float(config.plugin.image_timeout_seconds or 0.0):
        notes.append("超时已被消息预算夹取")
    if effective_max_tokens(config) > int(config.vision.max_tokens or 0):
        notes.append("max_tokens 低于下限已被抬高")
    return text + ("｜" + "、".join(notes) if notes else "")
