# -*- coding: utf-8 -*-
"""文本归一与分词（纯标准库，不依赖 ctx，可脱机单测）。

跨源反查返回的角色名来自不同语言、不同站点：AnimeTrace 常给日文或中文名，
SauceNAO 给下划线分隔的拉丁 tag。要让它们互相比较，必须先经同一套归一化。
本模块是全插件唯一允许做"名字相等判断"的地方，别在别处另写一套。
"""

from __future__ import annotations

import re
import unicodedata

# 归一化时先剥掉的尾注，如"初音ミク（CV：藤田咲）"、"角色(CV: 某某)"
_CV_SUFFIX_RE = re.compile(r"[（(][^（()）]*(?:cv|声优|配音)[^（()）]*[)）]\s*$", re.IGNORECASE)

# 拉丁/数字整词
_LATIN_WORD_RE = re.compile(r"[a-z0-9]+")

# CJK 连续段（平假名 / 片假名 / 扩展 A / 基本区 / 半角片假名）
_CJK_RUN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f]+")

# 注入中和：把用户可见的定界序列替换成不易与指令混淆的等价字符
_DELIMITER_REPLACEMENTS = (
    ("<<<", "\u2039\u2039\u2039"),
    ("<<", "\u2039\u2039"),
    (">>>", "\u203a\u203a\u203a"),
    (">>", "\u203a\u203a"),
    ("\u3010", "\u3014"),
    ("\u3011", "\u3015"),
)


def _fold(text: object) -> str:
    """NFKC 兼容分解 + 大小写折叠。全角斜杠、半角片假名等都在这一步统一。"""
    return unicodedata.normalize("NFKC", str(text or "")).casefold()


def normalize_name(text: object) -> str:
    """把角色名归一成可比较的键。

    去空白的做法是**只保留字母与数字**（``str.isalnum``），它天然覆盖 CJK ——
    如果改用 ``[^a-z0-9]`` 过滤，所有中文名都会归一成空串并互相误判为同一个角色。
    """
    value = _CV_SUFFIX_RE.sub("", _fold(text))
    return "".join(ch for ch in value if ch.isalnum())


def tokenize(text: object) -> frozenset[str]:
    """切出用于重叠打分的 token 集合。

    拉丁/数字按整词；CJK 按字符 bigram（单字段保留单字）。bigram 而非单字是为了
    让"初音"与"初音未来"有区分度，也不至于让长句里每个字都命中。
    """
    value = _fold(text)
    tokens: set[str] = set(_LATIN_WORD_RE.findall(value))
    for run in _CJK_RUN_RE.findall(value):
        if len(run) == 1:
            tokens.add(run)
            continue
        tokens.update(run[index:index + 2] for index in range(len(run) - 1))
    return frozenset(tokens)


def overlap_score(query_tokens: frozenset[str], document_tokens: frozenset[str]) -> float:
    """余弦型重叠度 ``|Q∩D| / sqrt(|Q|*|D|)``。

    刻意不用 Jaccard：一个 3 字别名去比 200 字的角色卡，Jaccard 恒趋近 0，
    会在长文本上把真实命中压死；除以几何平均才对长度不敏感。
    """
    if not query_tokens or not document_tokens:
        return 0.0
    shared = len(query_tokens & document_tokens)
    if not shared:
        return 0.0
    return shared / ((len(query_tokens) * len(document_tokens)) ** 0.5)


def neutralize(text: object) -> str:
    """中和文本里可能与外层定界符冲突的序列。

    VLM 描述与反查回来的名字都是**不可信文本**——图片内容可以诱导模型输出任意
    字符串。拼进提示词前必须把 ``<<<`` 一类序列打断，否则它们能伪造出"数据块
    结束"的假象，把后续内容抬成指令。
    """
    value = str(text or "")
    for source, replacement in _DELIMITER_REPLACEMENTS:
        value = value.replace(source, replacement)
    return value


def clean_display_name(text: object, limit: int = 40) -> str:
    """把源返回的名字整成可展示形态：折叠空白、去首尾标点、限长。"""
    value = " ".join(str(text or "").split())
    value = value.strip(" \t\r\n-_|,;.·・\u3001")
    return value[:limit]


def clean_booru_tag(tag: object) -> str:
    """Danbooru/Gelbooru 风格 tag 转可读名：``hatsune_miku`` → ``Hatsune Miku``。

    SauceNAO 的 ``characters`` 字段是 booru tag 风格（小写下划线），直接塞进提示词
    会和 AnimeTrace 返回的日文/中文名看起来完全无关，导致跨源"同一角色"判不出来。
    """
    value = " ".join(str(tag or "").replace("_", " ").split())
    if not value:
        return ""
    if value.isascii() and value.islower() and not any(ch.isdigit() for ch in value):
        value = value.title()
    return clean_display_name(value)


def truncate(text: object, limit: int) -> str:
    """按字符数截断，超长补省略号。"""
    value = str(text or "")
    if limit <= 0 or len(value) <= limit:
        return value
    return value[:limit] + "\u2026"


# ------------------------------------------------------------ 外观卡分类

#: 判定候选时**真正参与核对**的三类（见 prompts 的「三类核对」规则）。缺了哪一类，
#: 回执要提醒用户去补——所以这个常量是给展示层用的，不是检索用的。
CORE_APPEARANCE_CATEGORIES = ("发色发型", "眼睛", "服装")

#: 关键词表。**顺序即优先级**：一条卡同时提到"发"和"饰"时先命中哪个就归哪个。
#: 刻意不用裸的「发」字（会误吃"出发""发送"），只列真正指向头发的词。
_CATEGORY_KEYWORDS: "tuple[tuple[str, tuple[str, ...]], ...]" = (
    ("眼睛", ("瞳孔", "眼眸", "眼神", "眼睛", "虹膜", "瞳", "眸", "眼")),
    ("发色发型", ("头发", "发色", "发型", "发丝", "发尾", "发梢", "刘海",
                 "短发", "长发", "卷发", "直发", "辫", "呆毛")),
    # 刻意不放裸的「装」：它会把「装饰」「装置」一并吃成服装，而帽子这类含"装饰"的
    # 卡片其实属于配饰。要写就写全（制服 / 套装 / 装束），别用单字兜。
    # 裸「领」会误吃「领结」（那是配饰），所以只收真正的领部词。
    ("服装", ("衣", "服", "裙", "鞋", "袜", "袖", "裤", "袍", "衫",
             "立领", "翻领", "领口", "衣领", "领子", "高领", "圆领", "肩带",
             "外套", "斗篷", "大氅", "靴", "制服", "套装", "装束", "正装", "裙装", "上装", "下装")),
    ("配饰", ("饰", "环", "冠", "耳机", "蝴蝶结", "腰带", "包", "挂件", "项链",
             "耳坠", "帽", "十字架", "勋章")),
)


def classify_appearance_card(card: object) -> str:
    """把一条外观卡归到 眼睛 / 发色发型 / 服装 / 配饰 / 其他 之一。

    只服务**展示分组**（让用户一眼看出三类齐不齐），不参与检索与判定——
    所以宁可归到"其他"也不猜：判错了只是提示看着不准，不会影响识别结果。
    """
    text = str(card or "")
    for category, keywords in _CATEGORY_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return category
    return "其他"


def group_appearance_cards(cards: object) -> "dict[str, list[str]]":
    """按类别分组，组内保持原顺序，空类别不出现。"""
    groups: "dict[str, list[str]]" = {}
    for card in cards or ():
        value = str(card or "").strip()
        if not value:
            continue
        groups.setdefault(classify_appearance_card(value), []).append(value)
    return groups


def is_plausible_name(text: object) -> bool:
    """名字要能参与匹配，至少得有两个有效字符，且不能是纯数字或纯占位词。"""
    normalized = normalize_name(text)
    if len(normalized) < 2:
        return False
    if normalized.isdigit():
        return False
    return normalized not in {"unknown", "未识别", "无", "none", "null", "na"}
