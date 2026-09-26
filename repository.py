# -*- coding: utf-8 -*-
"""本地角色库读写（纯标准库，不依赖 ctx）。

设计要点：

* **损坏绝不上抛**。参考实现在库文件损坏且备份失败时 ``raise RuntimeError``，
  直接把 ``on_load`` 打挂、整个插件加载失败。这里一律降级为"改名留证 + 空库启动"。
* **原子写**。先写 ``.tmp`` 再 ``os.replace``，避免进程被杀在写入中途留下半个 JSON。
* **v1 迁移**。能直接读参考插件的 ``characters.json``（``schema_version`` 缺失、
  条目只有 id/name/aliases/relationship/appearance_cards），迁移前先备份原文。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable

try:  # Runner 可能按包加载，也可能只把目录塞进 sys.path
    from .models import Character
    from .textutil import normalize_name
except ImportError:  # pragma: no cover - 取决于加载方式
    from models import Character
    from textutil import normalize_name

SCHEMA_VERSION = 2

#: 参考实现 v1 条目的特征字段，用于判定"这是旧库"
V1_MARKER_FIELDS = {"id", "name", "aliases", "relationship", "appearance_cards"}

LogFn = Callable[[str, str], None]


def _noop_log(level: str, message: str) -> None:  # pragma: no cover - 默认实现
    del level, message


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def derive_id(name: str) -> str:
    """由角色名派生稳定 id。

    用 sha256 而不是 ASCII slug：中文名经过 slug 化会全部塌成同一个空串。
    """
    import hashlib

    digest = hashlib.sha256(normalize_name(name).encode("utf-8")).hexdigest()
    return f"char-{digest[:12]}"


class CharacterRepository:
    """角色库。所有写操作都会立刻落盘并刷新内存副本。"""

    def __init__(self, path: Path, log: LogFn | None = None) -> None:
        self._path = Path(path)
        self._log = log or _noop_log
        self._characters: tuple[Character, ...] = ()
        self._bridge: dict[str, str] = {}
        self._revision = 0
        self._index: dict[str, Character] = {}

    # ------------------------------------------------------------ 读取

    @property
    def path(self) -> Path:
        return self._path

    @property
    def characters(self) -> tuple[Character, ...]:
        return self._characters

    @property
    def revision(self) -> int:
        """每次库内容变化递增，供检索层的向量索引判断是否需要重建。"""
        return self._revision

    @property
    def name_bridge(self) -> dict[str, str]:
        return dict(self._bridge)

    def __len__(self) -> int:
        return len(self._characters)

    def load(self) -> None:
        """加载角色库。任何异常都被吸收，最多留一个空库。"""
        self._characters = ()
        self._bridge = {}
        self._index = {}
        if not self._path.exists():
            self._revision += 1
            return
        try:
            raw_text = self._path.read_text(encoding="utf-8")
            payload = json.loads(raw_text)
        except (OSError, json.JSONDecodeError) as exc:
            self._quarantine(f"角色库无法解析（{exc}）")
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("characters"), list):
            self._quarantine("角色库缺少 characters 列表")
            return
        if "schema_version" not in payload:
            payload = self._migrate_v1(payload, raw_text)
            if payload is None:
                return
            try:
                self._write_payload(payload)
            except OSError as exc:
                self._log("warning", f"v1 迁移结果落盘失败（本次仍按内存副本使用）：{exc}")
        try:
            characters = [item for item in (Character.from_dict(entry) for entry in payload["characters"]) if item]
        except Exception as exc:  # 防御：单条畸形条目不该拖垮整个库
            self._quarantine(f"角色库条目解析失败（{exc}）")
            return
        self._characters = tuple(sorted(characters, key=lambda item: item.name))
        bridge = payload.get("name_bridge")
        self._bridge = {
            str(key): str(value)
            for key, value in (bridge.items() if isinstance(bridge, dict) else [])
            if str(key).strip() and str(value).strip()
        }
        self._rebuild_index()
        self._revision += 1

    def ensure_exists(self) -> None:
        if self._path.exists():
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._write_payload({"schema_version": SCHEMA_VERSION, "characters": [], "name_bridge": {}})
        except OSError as exc:
            self._log("error", f"无法创建角色库文件：{exc}")

    def import_from(self, source: str) -> bool:
        """从外部文件导入角色库（只在本地库尚不存在时执行）。"""
        if self._path.exists():
            self._log("warning", "角色库已存在，跳过 import_from 导入")
            return False
        candidate = Path(str(source)).expanduser()
        if not candidate.is_file():
            self._log("warning", f"import_from 指向的文件不存在：{candidate}")
            return False
        try:
            raw_text = candidate.read_text(encoding="utf-8")
            payload = json.loads(raw_text)
        except (OSError, json.JSONDecodeError) as exc:
            self._log("error", f"import_from 文件无法读取：{exc}")
            return False
        if not isinstance(payload, dict) or not isinstance(payload.get("characters"), list):
            self._log("error", "import_from 文件结构不是角色库")
            return False
        if "schema_version" not in payload:
            payload = self._migrate_v1(payload, raw_text, backup=False)
            if payload is None:
                return False
        try:
            self.ensure_exists()
            self._write_payload(payload)
        except OSError as exc:
            self._log("error", f"导入角色库写入失败：{exc}")
            return False
        self.load()
        self._log("info", f"已从 {candidate} 导入 {len(self._characters)} 个角色")
        return True

    def _migrate_v1(self, payload: dict[str, Any], raw_text: str, backup: bool = True) -> dict[str, Any] | None:
        """把参考实现的 v1 角色库迁移成 v2。返回 None 表示迁移失败。"""
        entries = payload.get("characters")
        if not isinstance(entries, list):
            self._quarantine("角色库缺少 characters 列表")
            return None
        looks_like_v1 = all(
            isinstance(entry, dict) and set(entry) & V1_MARKER_FIELDS for entry in entries[:5]
        )
        if not looks_like_v1 and entries:
            self._quarantine("角色库结构无法识别（既非 v1 也非 v2）")
            return None
        if backup:
            backup_path = self._path.with_name(f"{self._path.stem}.v1.bak-{_timestamp()}.json")
            try:
                backup_path.write_text(raw_text, encoding="utf-8")
            except OSError as exc:
                self._log("warning", f"v1 备份失败，继续迁移：{exc}")
            else:
                self._log("info", f"已备份 v1 角色库到 {backup_path.name}")
        migrated: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            migrated.append({
                "id": str(entry.get("id") or "").strip() or derive_id(name),
                "name": name,
                "aliases": [str(item) for item in (entry.get("aliases") or []) if str(item).strip()],
                "work": str(entry.get("work") or "").strip(),
                "work_aliases": [str(item) for item in (entry.get("work_aliases") or []) if str(item).strip()],
                "persona": str(entry.get("persona") or "").strip(),
                "tags": [str(item) for item in (entry.get("tags") or []) if str(item).strip()],
                "relationship": str(entry.get("relationship") or "").strip(),
                "appearance_cards": [str(item) for item in (entry.get("appearance_cards") or []) if str(item).strip()],
                "source": "migrated-v1",
            })
        self._log("info", f"角色库 v1 → v2 迁移完成，共 {len(migrated)} 个角色")
        return {"schema_version": SCHEMA_VERSION, "characters": migrated, "name_bridge": {}}

    def _quarantine(self, reason: str) -> None:
        """把坏文件改名留证并以空库继续，绝不抛。"""
        target = self._path.with_name(f"{self._path.stem}.corrupt-{_timestamp()}.json")
        try:
            self._path.replace(target)
            self._log("error", f"{reason}；已改名为 {target.name} 并以空角色库继续")
        except OSError as exc:
            self._log("error", f"{reason}；且改名失败（{exc}），以空角色库继续")
        self._characters = ()
        self._bridge = {}
        self._index = {}
        self._revision += 1

    def _rebuild_index(self) -> None:
        index: dict[str, Character] = {}
        for character in self._characters:
            for name in character.all_names:
                key = normalize_name(name)
                if key and key not in index:
                    index[key] = character
        self._index = index

    # ------------------------------------------------------------ 查找

    def find_name(self, name: Any) -> Character | None:
        """按正名或别名精确查找（两侧同样归一，忽略大小写与全半角差异）。"""
        key = normalize_name(name)
        if not key:
            return None
        return self._index.get(key)

    def resolve(self, raw_name: Any) -> Character | None:
        """把源返回的名字解析到本地角色：先查正名/别名，再查 name_bridge。"""
        character = self.find_name(raw_name)
        if character is not None:
            return character
        bridged = self._bridge.get(normalize_name(raw_name))
        if bridged:
            return next((item for item in self._characters if item.character_id == bridged), None)
        return None

    def remember_bridge(self, raw_name: Any, character: Character) -> None:
        """记录"源用这个名字指代该角色"，后续同名命中可直接解析。"""
        key = normalize_name(raw_name)
        if not key or self._bridge.get(key) == character.character_id:
            return
        raw = self._read_raw()
        bridge = raw.get("name_bridge")
        if not isinstance(bridge, dict):
            bridge = {}
        bridge[key] = character.character_id
        raw["name_bridge"] = bridge
        try:
            self._write_payload(raw)
        except OSError as exc:
            self._log("warning", f"name_bridge 写入失败：{exc}")
            return
        self._bridge = {str(k): str(v) for k, v in bridge.items()}

    def find_by_work(self, work: Any) -> tuple[Character, ...]:
        """按作品名找角色（作品跨源匹配比角色名可靠得多，可作强信号）。"""
        key = normalize_name(work)
        if not key:
            return ()
        matched = []
        for character in self._characters:
            if any(normalize_name(item) == key for item in character.all_works):
                matched.append(character)
        return tuple(matched)

    def list_characters(self) -> tuple[Character, ...]:
        return self._characters

    def search(self, keyword: Any, limit: int = 20) -> tuple[Character, ...]:
        """简单子串搜索，供命令与工具做陈列用（检索打分不走这里）。"""
        key = normalize_name(keyword)
        if not key:
            return self._characters[:limit]
        matched = [
            character
            for character in self._characters
            if key in character.normalized_profile
            or any(key in normalize_name(item) for item in character.all_names)
        ]
        return tuple(matched[:limit])

    # ------------------------------------------------------------ 写入

    def _read_raw(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"schema_version": SCHEMA_VERSION, "characters": [], "name_bridge": {}}
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._log("error", f"写入前读取角色库失败：{exc}")
            return {"schema_version": SCHEMA_VERSION, "characters": [], "name_bridge": {}}
        if not isinstance(payload, dict) or not isinstance(payload.get("characters"), list):
            return {"schema_version": SCHEMA_VERSION, "characters": [], "name_bridge": {}}
        payload.setdefault("schema_version", SCHEMA_VERSION)
        payload.setdefault("name_bridge", {})
        return payload

    def _write_payload(self, payload: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, self._path)

    def _commit(self, payload: dict[str, Any]) -> None:
        self._write_payload(payload)
        self.load()

    def _assert_name_free(self, name: str, *, exclude_id: str = "") -> None:
        """名称与别名必须全局唯一，否则"精确命中"会指向不确定的角色。"""
        key = normalize_name(name)
        owner = self._index.get(key)
        if owner is not None and owner.character_id != exclude_id:
            raise ValueError(f"名称或别名「{name}」已属于角色「{owner.name}」")

    def upsert(
        self,
        *,
        name: str,
        work: str = "",
        relationship: str = "",
        persona: str = "",
        tags: Iterable[str] = (),
        appearance_cards: Iterable[str] = (),
        aliases: Iterable[str] = (),
    ) -> Character:
        normalized_name = str(name or "").strip()
        if not normalized_name:
            raise ValueError("角色名不能为空")
        existing = self.find_name(normalized_name)
        if existing is not None and normalize_name(existing.name) != normalize_name(normalized_name):
            # 这个名字被别的角色占为别名。若放行，同一个名字会指向两个条目，
            # 之后"精确命中"就再也说不清是哪一个 —— 必须在这里拦住。
            raise ValueError(f"名称「{normalized_name}」已属于角色「{existing.name}」的别名")
        exclude_id = existing.character_id if existing else ""
        self._assert_name_free(normalized_name, exclude_id=exclude_id)
        for alias in aliases:
            self._assert_name_free(str(alias), exclude_id=exclude_id)

        cards = [str(item).strip() for item in appearance_cards if str(item).strip()]
        tag_values = [str(item).strip() for item in tags if str(item).strip()]
        payload = self._read_raw()
        entries = payload["characters"]
        target = next(
            (
                entry
                for entry in entries
                if isinstance(entry, dict)
                and normalize_name(entry.get("name")) == normalize_name(normalized_name)
            ),
            None,
        )
        merged_aliases: list[str] = []
        seen: set[str] = set()
        for alias in [*(existing.aliases if existing else ()), *aliases]:
            key = normalize_name(alias)
            if key and key != normalize_name(normalized_name) and key not in seen:
                seen.add(key)
                merged_aliases.append(str(alias).strip())
        merged_cards: list[str] = []
        card_seen: set[str] = set()
        for card in [*(existing.appearance_cards if existing else ()), *cards]:
            key = card.casefold()
            if key not in card_seen:
                card_seen.add(key)
                merged_cards.append(card)

        entry = {
            "id": (existing.character_id if existing else "") or derive_id(normalized_name),
            "name": normalized_name,
            "aliases": merged_aliases,
            "work": str(work or (existing.work if existing else "")).strip(),
            "work_aliases": list(existing.work_aliases) if existing else [],
            "persona": str(persona or (existing.persona if existing else "")).strip(),
            "tags": tag_values or (list(existing.tags) if existing else []),
            "relationship": str(relationship or (existing.relationship if existing else "")).strip(),
            "appearance_cards": merged_cards[:15],
            "source": (existing.source if existing else "admin"),
        }
        payload["characters"] = [
            item for item in entries if item is not target
        ] + [entry]
        self._commit(payload)
        created = self.find_name(normalized_name)
        if created is None:  # pragma: no cover - 写入后必然可查到
            raise RuntimeError("角色库写入后未能查到该角色")
        return created

    def _update(self, name: str, field: str, value: Any) -> Character:
        character = self.find_name(name)
        if character is None:
            raise ValueError(f"角色库中不存在「{str(name).strip()}」")
        payload = self._read_raw()
        target = None
        for entry in payload["characters"]:
            if isinstance(entry, dict) and str(entry.get("id") or "") == character.character_id:
                target = entry
                break
        if target is None:
            raise ValueError(f"角色库中不存在「{str(name).strip()}」")
        target[field] = value
        self._commit(payload)
        updated = self.find_name(character.name)
        if updated is None:  # pragma: no cover
            raise RuntimeError("角色库更新后未能查到该角色")
        return updated

    def append_appearance_cards(self, name: str, appearance_cards: Iterable[str], limit: int = 15) -> Character:
        """合并新的外观卡，保留既有身份与关系（去重、只留最后 limit 条）。"""
        character = self.find_name(name)
        if character is None:
            raise ValueError(f"角色库中不存在「{str(name).strip()}」")
        cards = [str(item).strip() for item in appearance_cards if str(item).strip()]
        if not cards:
            raise ValueError("没有可补充的外观卡")
        merged = list(dict.fromkeys([*character.appearance_cards, *cards]))[-limit:]
        return self._update(character.name, "appearance_cards", merged)

    def set_appearance_cards(
        self, name: str, appearance_cards: Iterable[str], limit: int = 15
    ) -> Character:
        """整体**替换**外观卡（压缩后写回用）。

        与 ``append_appearance_cards`` 的差别就在"替换"：压缩的产物是要顶掉旧的那一堆，
        再走合并逻辑等于白压。调用方负责保证传入的是完整、已整理过的一组。
        """
        cards = [str(item).strip() for item in appearance_cards if str(item).strip()]
        if not cards:
            raise ValueError("没有可写入的外观卡")
        return self._update(name, "appearance_cards", cards[-limit:])

    def set_relationship(self, name: str, relationship: str) -> Character:
        return self._update(name, "relationship", str(relationship).strip())

    def set_work(self, name: str, work: str) -> Character:
        return self._update(name, "work", str(work).strip())

    def set_persona(self, name: str, persona: str) -> Character:
        return self._update(name, "persona", str(persona).strip())

    def add_alias(self, name: str, alias: str) -> Character:
        character = self.find_name(name)
        if character is None:
            raise ValueError(f"角色库中不存在「{str(name).strip()}」")
        normalized_alias = str(alias).strip()
        if not normalized_alias:
            raise ValueError("别名不能为空")
        if normalize_name(normalized_alias) == normalize_name(character.name):
            raise ValueError("别名与角色正名相同")
        self._assert_name_free(normalized_alias, exclude_id=character.character_id)
        if any(normalize_name(item) == normalize_name(normalized_alias) for item in character.aliases):
            raise ValueError(f"角色「{character.name}」已有别名「{normalized_alias}」")
        return self._update(character.name, "aliases", [*character.aliases, normalized_alias])

    def remove_alias(self, name: str, alias: str) -> Character:
        character = self.find_name(name)
        if character is None:
            raise ValueError(f"角色库中不存在「{str(name).strip()}」")
        key = normalize_name(alias)
        remaining = [item for item in character.aliases if normalize_name(item) != key]
        if len(remaining) == len(character.aliases):
            raise ValueError(f"角色「{character.name}」没有别名「{str(alias).strip()}」")
        return self._update(character.name, "aliases", remaining)

    def delete(self, name: str) -> Character:
        character = self.find_name(name)
        if character is None:
            raise ValueError(f"角色库中不存在「{str(name).strip()}」")
        payload = self._read_raw()
        payload["characters"] = [
            entry
            for entry in payload["characters"]
            if not (isinstance(entry, dict) and str(entry.get("id") or "") == character.character_id)
        ]
        self._commit(payload)
        return character

    def catalog(self, limit: int = 40) -> list[dict[str, Any]]:
        """给视觉提示词用的精简目录。**注意**：这是给调用方兜底的静态截断路径，
        真正进提示词的候选应由 ``retrieval.py`` 按相关性挑选后传入。"""
        return [
            {
                "id": character.character_id,
                "name": character.name,
                "aliases": list(character.aliases),
                "work": character.work,
                "appearance_cards": list(character.appearance_cards[:3]),
            }
            for character in self._characters[:limit]
        ]
