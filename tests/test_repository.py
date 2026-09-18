# -*- coding: utf-8 -*-
"""角色库读写、v1 迁移、损坏降级的回归测试。"""

import json
import sys
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from repository import SCHEMA_VERSION, CharacterRepository, derive_id  # noqa: E402


def make_repo(tmp_path: Path) -> CharacterRepository:
    return CharacterRepository(tmp_path / "characters.json")


def write_v1(path: Path) -> None:
    """参考实现（QlzqQlzq/character-knowledge-plugin）的角色库格式。"""
    path.write_text(json.dumps({
        "characters": [
            {
                "id": "char-abc123",
                "name": "初音未来",
                "aliases": ["初音ミク", "Hatsune Miku"],
                "relationship": "朋友",
                "appearance_cards": ["蓝绿色双马尾", "头上有一对黑色发饰"],
            },
            {
                "id": "char-def456",
                "name": "阿罗娜",
                "aliases": ["アロナ"],
                "relationship": "",
                "appearance_cards": ["蓝白配色长发", "头顶有发光圆环"],
            },
        ]
    }, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- v1 迁移

def test_v1_library_migrates_and_backs_up(tmp_path: Path) -> None:
    path = tmp_path / "characters.json"
    write_v1(path)
    repo = make_repo(tmp_path)
    repo.load()

    assert len(repo) == 2
    assert repo.find_name("初音未来") is not None
    # 旧字段必须原样保留
    miku = repo.find_name("初音ミク")
    assert miku is not None and miku.relationship == "朋友"
    assert miku.appearance_cards == ("蓝绿色双马尾", "头上有一对黑色发饰")
    # 新字段补空而不是丢条目
    assert miku.work == ""
    assert miku.persona == ""
    # 原文备份 + schema 升版落盘
    backups = list(tmp_path.glob("characters.v1.bak-*.json"))
    assert len(backups) == 1
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == SCHEMA_VERSION
    assert len(on_disk["characters"]) == 2


def test_derive_id_is_stable_for_non_ascii_names() -> None:
    """中文名 slug 化会全塌成空串，所以 id 必须走摘要。"""
    assert derive_id("初音未来") == derive_id("初音未来")
    assert derive_id("初音未来") != derive_id("阿罗娜")
    assert derive_id("初音未来").startswith("char-")
    assert len({
        derive_id(name) for name in ("初音未来", "阿罗娜", "言和", "乐正绫", "星尘")
    }) == 5


# ---------------------------------------------------------------- 损坏降级

def test_corrupt_json_does_not_raise_and_quarantines(tmp_path: Path) -> None:
    """参考实现在这里 raise，把 on_load 直接打挂；我们必须降级。"""
    path = tmp_path / "characters.json"
    path.write_text("{ this is not json", encoding="utf-8")
    repo = make_repo(tmp_path)

    repo.load()  # 不抛

    assert len(repo) == 0
    assert list(tmp_path.glob("characters.corrupt-*.json")), "坏文件应改名留证"


def test_wrong_shape_does_not_raise(tmp_path: Path) -> None:
    path = tmp_path / "characters.json"
    path.write_text(json.dumps({"characters": "not-a-list"}), encoding="utf-8")
    repo = make_repo(tmp_path)
    repo.load()
    assert len(repo) == 0


def test_missing_file_is_not_treated_as_corrupt(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    repo.load()
    assert len(repo) == 0
    assert not list(tmp_path.glob("characters.corrupt-*.json"))


# ---------------------------------------------------------------- 写入

def test_upsert_creates_then_updates_without_tmp_residue(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    repo.load()
    first = repo.upsert(name="阿罗娜", work="蔚蓝档案", appearance_cards=["蓝白长发", "发光圆环"])
    assert first.character_id.startswith("char-")

    updated = repo.upsert(name="阿罗娜", relationship="同伴", appearance_cards=["头顶深色圆环"])
    assert updated.relationship == "同伴"
    # 外观卡是合并不是覆盖
    assert "蓝白长发" in updated.appearance_cards
    assert "头顶深色圆环" in updated.appearance_cards
    assert len(repo) == 1
    assert not list(tmp_path.glob("*.tmp")), "写入后不应残留 .tmp"


def test_set_appearance_cards_replaces_instead_of_merging(tmp_path: Path) -> None:
    """整理（压缩）后的写回必须是**替换**。

    这里若走合并逻辑，压缩就白做了：刚合并出来的 4 条会和被它替代的 10 条并排放着，
    条数不减反增。
    """
    repo = make_repo(tmp_path)
    repo.load()
    repo.upsert(name="鸣澜", relationship="搭档", appearance_cards=["旧的甲", "旧的乙", "旧的丙"])
    updated = repo.set_appearance_cards("鸣澜", ["整理后甲", "整理后乙", "整理后丙"])
    assert list(updated.appearance_cards) == ["整理后甲", "整理后乙", "整理后丙"]
    assert updated.relationship == "搭档", "替换外观卡不该动身份字段"
    assert not list(tmp_path.glob("*.tmp"))


def test_set_appearance_cards_refuses_empty(tmp_path: Path) -> None:
    """空列表会把用户的卡片清光——宁可报错，也不要静默抹掉。"""
    repo = make_repo(tmp_path)
    repo.load()
    repo.upsert(name="鸣澜", appearance_cards=["旧的甲"])
    with pytest.raises(ValueError):
        repo.set_appearance_cards("鸣澜", [])
    assert list(repo.find_name("鸣澜").appearance_cards) == ["旧的甲"]


def test_upsert_rejects_empty_name_and_name_collision(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    repo.load()
    repo.upsert(name="阿罗娜")
    with pytest.raises(ValueError):
        repo.upsert(name="   ")
    with pytest.raises(ValueError):
        repo.upsert(name="阿罗娜", aliases=["普拉娜"])
        repo.upsert(name="普拉娜")  # 已被上一个角色占为别名


def test_alias_uniqueness_across_characters(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    repo.load()
    repo.upsert(name="初音未来", aliases=["初音ミク"])
    repo.upsert(name="阿罗娜")

    with pytest.raises(ValueError):
        repo.add_alias("阿罗娜", "初音ミク")
    # 同一角色下重复添加也不该静默通过
    with pytest.raises(ValueError):
        repo.add_alias("初音未来", "初音ミク")


def test_alias_add_remove_roundtrip(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    repo.load()
    repo.upsert(name="初音未来")
    repo.add_alias("初音未来", "Hatsune Miku")
    assert repo.find_name("hatsune  miku") is not None  # 归一后应命中（空白与大小写无关）

    repo.remove_alias("初音未来", "Hatsune Miku")
    assert repo.find_name("Hatsune Miku") is None
    with pytest.raises(ValueError):
        repo.remove_alias("初音未来", "Hatsune Miku")


def test_delete_character(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    repo.load()
    repo.upsert(name="阿罗娜")
    repo.delete("阿罗娜")
    assert len(repo) == 0
    with pytest.raises(ValueError):
        repo.delete("阿罗娜")


def test_field_setters(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    repo.load()
    repo.upsert(name="阿罗娜")
    assert repo.set_persona("阿罗娜", "来自蔚蓝档案的 AI 助手").persona.startswith("来自蔚蓝档案")
    assert repo.set_work("阿罗娜", "蔚蓝档案").work == "蔚蓝档案"
    assert repo.set_relationship("阿罗娜", "同伴").relationship == "同伴"
    with pytest.raises(ValueError):
        repo.set_work("不存在的角色", "x")


# ---------------------------------------------------------------- 解析与桥接

def test_resolve_uses_alias_then_name_bridge(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    repo.load()
    alona = repo.upsert(name="阿罗娜", aliases=["アロナ"])
    assert repo.resolve("アロナ") is alona

    # 源用了一个既非正名也非别名的写法 —— 只能靠桥接
    assert repo.resolve("Arona") is None
    repo.remember_bridge("Arona", alona)
    assert repo.resolve("Arona") is alona

    # 桥接必须持久化，重新加载后仍然有效
    reloaded = make_repo(tmp_path)
    reloaded.load()
    assert reloaded.resolve("Arona") is not None


def test_find_by_work_matches_work_aliases(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    repo.load()
    repo.upsert(name="阿罗娜", work="蔚蓝档案")
    repo.upsert(name="普拉娜", work="蔚蓝档案")
    repo.upsert(name="初音未来", work="VOCALOID")
    matched = repo.find_by_work("蔚蓝档案")
    assert {item.name for item in matched} == {"阿罗娜", "普拉娜"}
    assert repo.find_by_work("不存在的作品") == ()


def test_revision_changes_on_write(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    repo.load()
    before = repo.revision
    repo.upsert(name="阿罗娜")
    assert repo.revision != before


# ---------------------------------------------------------------- 卫生

def test_ctx_never_leaks_outside_plugin_py() -> None:
    """check_plugin.py 只扫描 plugin.py 推导能力名。

    其它模块里写 self.ctx 会"静态检查全绿、真机拒绝授权"，所以这条是硬约束。
    """
    offenders = []
    for path in PLUGIN_ROOT.glob("*.py"):
        if path.name == "plugin.py":
            continue
        if "self.ctx" in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert not offenders, f"这些模块出现了 self.ctx：{offenders}"
