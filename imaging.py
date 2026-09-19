# -*- coding: utf-8 -*-
"""图片解码与压缩（不依赖 ctx）。

AnimeTrace 对上传体积有硬限制（超限直接 HTTP 413，官方没写具体数值），所以发出去
之前必须能压缩。压缩副本**只用于上传**，绝不改写原图，也绝不进缓存键。
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import ipaddress
import socket
import urllib.error
import urllib.request
from urllib.parse import urljoin, urlsplit

#: 逐级缩小的边长阶梯 + 逐级下调的 JPEG 质量。
#: 两层阶梯一起走，才能保证高熵截图（PNG 截图、噪点图）也压得下来，
#: 而不是"压一次还是超限就直接放弃原图"。
EDGE_LADDER = (1600, 1280, 1024, 800, 640, 512, 384, 256)
QUALITY_LADDER = (88, 80, 72, 64, 56, 48, 40)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode_base64_image(encoded: object) -> bytes | None:
    """解码 base64 图片载荷，失败返回 None（绝不抛）。"""
    if isinstance(encoded, bytes):
        return encoded or None
    if not isinstance(encoded, str) or not encoded:
        return None
    payload = encoded.strip()
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    try:
        return base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error):
        # 有些适配器会带换行或 padding 不规范，退一步用非严格模式再试一次
        try:
            return base64.b64decode(payload, validate=False)
        except (ValueError, binascii.Error):
            return None


def sniff_mime_type(data: bytes) -> str:
    """按魔数判断图片类型（比信任扩展名可靠）。"""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"BM"):
        return "image/bmp"
    return "image/png"


def prepare_upload(
    image_bytes: bytes,
    *,
    max_bytes: int,
    edge_ladder: tuple[int, ...] = EDGE_LADDER,
    quality_ladder: tuple[int, ...] = QUALITY_LADDER,
) -> tuple[bytes, str]:
    """产出体积受控的上传副本，返回 ``(数据, mime)``。

    没超限就原样返回（省一次重编码）。超限则用 Pillow 逐级缩放 + 降质量。
    Pillow 缺失时给一句能照做的中文错误，而不是 ImportError 堆栈。
    """
    if not image_bytes:
        raise ValueError("图片内容为空")
    if len(image_bytes) <= max_bytes:
        return image_bytes, sniff_mime_type(image_bytes)
    try:
        import io

        from PIL import Image
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        raise ValueError(
            f"图片超过上传上限 {max_bytes} 字节，且环境缺少 Pillow 无法压缩；"
            "请安装 Pillow（pip install Pillow）或调高该源的上传上限"
        ) from exc

    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            original = source.convert("RGB")
    except Exception as exc:
        raise ValueError(f"图片无法解析，无法压缩上传：{exc}") from exc

    for edge in edge_ladder:
        candidate = original.copy()
        candidate.thumbnail((edge, edge))
        for quality in quality_ladder:
            buffer = io.BytesIO()
            candidate.save(buffer, format="JPEG", quality=quality, optimize=True)
            result = buffer.getvalue()
            if len(result) <= max_bytes:
                return result, "image/jpeg"
    raise ValueError(f"图片压缩后仍超过上传上限 {max_bytes} 字节")


def extract_image_payload(component: object) -> tuple[bytes | None, str]:
    """从消息的图片组件里取出图片数据与来源说明。

    返回 ``(数据或 None, 取得方式的说明)``。说明进日志用于排障：不同适配器保留的
    字段不一样，`binary_data_base64` 缺失时能一眼看出还剩什么可用。
    """
    if not isinstance(component, dict):
        return None, "组件不是字典"
    for field in ("binary_data_base64", "data", "base64", "file"):
        data = decode_base64_image(component.get(field))
        if data:
            return data, f"binary_data_base64/{field}"
    url = str(component.get("url") or component.get("file") or "").strip()
    if url.startswith(("http://", "https://")):
        return None, f"仅有 url，需下载：{url[:80]}"
    present = sorted(key for key in component if key not in {"type"})
    return None, f"无可用图片载荷，字段={present}"


#: 视为重定向的状态码。
_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """禁止 urllib 自动跟随重定向。

    默认行为是自动跟的，而**一条 302 就足以把请求带出校验过的目标**：先给一个解析到
    公网的域名骗过检查，再 302 到 `127.0.0.1` 或云元数据地址。所以这里关掉自动跟随，
    由 ``download_image`` 手动逐跳重新校验。
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


_OPENER = urllib.request.build_opener(_NoRedirectHandler)


def host_resolves_to_public(host: str) -> bool:
    """主机是否**只**解析到公网地址。

    只拦字面量 IP 会被"域名 A 记录指向内网"绕过（`evil.example.com` → `192.168.1.1`），
    所以必须真正解析后逐条判定。解析不出也判为不可信——无法证明它安全。

    注意这是**阻塞**调用（DNS），只在 ``asyncio.to_thread`` 里用。
    """
    if not str(host or "").strip():
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, OSError, ValueError):
        return False
    if not infos:
        return False
    for info in infos:
        try:
            address = ipaddress.ip_address(info[4][0])
        except (IndexError, ValueError):
            return False
        if (address.is_private or address.is_loopback or address.is_link_local
                or address.is_reserved or address.is_multicast or address.is_unspecified):
            return False
    return True


def download_image(
    url: str,
    *,
    timeout_seconds: float = 15.0,
    max_bytes: int = 8_388_608,
    max_redirects: int = 3,
) -> bytes | None:
    """从 URL 下载图片（部分适配器只保留 url 而不带 base64）。

    这个 URL **来自入站消息组件，是攻击者可控的**，所以按 SSRF 要求逐层设防：

    * 只接受 http / https（`file://`、`gopher://` 之类直接拒）；
    * 主机必须解析到公网——见 ``host_resolves_to_public``；
    * **手动逐跳跟随重定向**并每跳重新校验，避免 302 绕过；
    * 体积与超时都有上限。

    失败一律返回 None，调用方降级为"这张图没有可用数据"（并会记日志）。
    """
    target = str(url or "").strip()
    for _ in range(max(0, max_redirects) + 1):
        parts = urlsplit(target)
        if parts.scheme not in ("http", "https"):
            return None
        if not host_resolves_to_public(parts.hostname or ""):
            return None
        try:
            request = urllib.request.Request(
                target, headers={"User-Agent": "MaiBot-character-recognizer/1.0"}
            )
            with _OPENER.open(request, timeout=timeout_seconds) as response:
                data = response.read(max_bytes + 1)
                status = int(getattr(response, "status", 200) or 200)
        except urllib.error.HTTPError as exc:
            location = exc.headers.get("Location") if exc.headers else None
            if exc.code in _REDIRECT_CODES and location:
                target = urljoin(target, str(location))
                continue
            return None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return None
        if status in _REDIRECT_CODES:
            return None  # 带重定向码却没有 Location：放弃
        if not data or len(data) > max_bytes:
            return None
        return data
    return None
