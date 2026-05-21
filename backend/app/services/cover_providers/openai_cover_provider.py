"""OpenAI 兼容图片接口（/v1/images/generations）的封面图片 Provider"""
from __future__ import annotations

import base64
from typing import Any

import httpx

from app.logger import get_logger
from app.services.cover_providers.base_cover_provider import BaseCoverProvider, CoverGenerationResult

logger = get_logger(__name__)


class OpenAICompatibleCoverProvider(BaseCoverProvider):
    """基于 OpenAI 兼容图片接口（/v1/images/generations）的封面生成实现。

    适用于：OpenAI 官方（DALL-E 3 / gpt-image-1）、SiliconFlow 图片、
    One-API / New-API 图片代理、ChatAnywhere、阿里百炼兼容模式 等任意以
    OpenAI 协议暴露的图片生成服务。

    与 GrokCoverProvider 的差异：
    - 不发送 xAI 专属的 ``aspect_ratio`` / ``resolution`` 字段；
    - 不发送 ``response_format`` ——某些上游（例如 gpt-image-1）会拒绝该字段，
      而 DALL-E / SiliconFlow 默认就会返回 ``b64_json`` 或 ``url`` 中的一种。
      本实现同时兼容两种返回形态。
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        # 仅去掉末尾斜杠；不主动改写路径，让用户能精确控制带 /v1 或不带
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")

    async def generate_cover(
        self,
        *,
        prompt: str,
        model: str,
        width: int,
        height: int,
    ) -> CoverGenerationResult:
        url = f"{self.base_url}/images/generations"
        # 直接传 size="<width>x<height>"。
        # - DALL-E 3 仅支持 1024x1024 / 1024x1792 / 1792x1024 三个尺寸；
        # - gpt-image-1 / SiliconFlow 等支持任意尺寸；
        # 不一致时上游返回的 400 错误会通过 httpx.HTTPStatusError -> upstream
        # detail 透传给用户，避免在 provider 层做有损猜测。
        payload: dict[str, Any] = {
            "model": model,
            "prompt": (prompt or "").strip(),
            "n": 1,
            "size": f"{max(width, 1)}x{max(height, 1)}",
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.debug(
            "Custom endpoint 封面生成请求开始: url=%s model=%s size=%sx%s prompt_len=%s prompt_preview=%s",
            url,
            model,
            width,
            height,
            len(prompt or ""),
            (prompt or "")[:300].replace("\n", " "),
        )

        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(url, headers=headers, json=payload)

            logger.debug(
                "Custom endpoint 封面生成响应: status=%s content_type=%s body_preview=%s",
                response.status_code,
                response.headers.get("content-type"),
                response.text[:1000],
            )

            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Custom endpoint 封面生成 HTTP 错误: status=%s response=%s",
                exc.response.status_code if exc.response else None,
                exc.response.text[:2000] if exc.response is not None else None,
            )
            raise
        except Exception:
            logger.error("Custom endpoint 封面生成请求异常", exc_info=True)
            raise

        images = data.get("data") or []
        if not images:
            logger.error("Custom endpoint 未返回图片结果: data=%s", data)
            raise ValueError("Custom endpoint 未返回图片结果")

        image_item = images[0]
        revised_prompt = image_item.get("revised_prompt")

        # 优先取 b64_json（DALL-E b64 模式 / gpt-image-1 默认 / SiliconFlow 等）
        b64_json = image_item.get("b64_json")
        if b64_json:
            content = self._decode_base64_image(b64_json)
            return {
                "content": content,
                "mime_type": "image/png",
                "file_extension": "png",
                "revised_prompt": revised_prompt,
                "provider": "openai",
                "model": model,
            }

        # 退路：DALL-E 3 默认返回的是临时 URL，需要二次拉取
        image_url = image_item.get("url")
        if image_url:
            logger.debug("Custom endpoint 返回图片 URL，开始下载: %s", image_url)
            try:
                async with httpx.AsyncClient(timeout=180.0) as client:
                    image_response = await client.get(image_url)
                image_response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "Custom endpoint 图片下载 HTTP 错误: status=%s response=%s",
                    exc.response.status_code if exc.response else None,
                    exc.response.text[:2000] if exc.response is not None else None,
                )
                raise
            except Exception:
                logger.error("Custom endpoint 图片下载异常", exc_info=True)
                raise

            content_type = image_response.headers.get("content-type", "image/png")
            file_extension = self._guess_extension(content_type=content_type, image_url=image_url)
            return {
                "content": image_response.content,
                "mime_type": content_type,
                "file_extension": file_extension,
                "revised_prompt": revised_prompt,
                "provider": "openai",
                "model": model,
            }

        logger.error("Custom endpoint 返回内容中既没有 b64_json，也没有 url: %s", data)
        raise ValueError("Custom endpoint 未返回可用的图片数据")

    @staticmethod
    def _decode_base64_image(value: str) -> bytes:
        if value.startswith("data:") and "," in value:
            value = value.split(",", 1)[1]
        return base64.b64decode(value)

    @staticmethod
    def _guess_extension(*, content_type: str, image_url: str) -> str:
        lowered_content_type = (content_type or "").lower()
        lowered_url = (image_url or "").lower()
        if (
            "jpeg" in lowered_content_type
            or "jpg" in lowered_content_type
            or lowered_url.endswith(".jpg")
            or lowered_url.endswith(".jpeg")
        ):
            return "jpg"
        if "webp" in lowered_content_type or lowered_url.endswith(".webp"):
            return "webp"
        return "png"
