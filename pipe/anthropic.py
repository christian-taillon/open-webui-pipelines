"""
title: Final Unified Anthropic Pipe with Dynamic Discovery, Caching, and Streaming Tools
authors: Balaxxe, nbellochi, Bermont, Mark Kazakov, Christian Taillon (Consolidated & Enhanced by AI)
author_url: https://github.com/christian-taillon
funding_url: https://github.com/open-webui
version: 8.2
license: MIT
requirements: pydantic>=2.0.0, aiohttp>=3.8.0
environment_variables:
    - ANTHROPIC_API_KEY (required)
This script is the definitive, all-in-one integration for Anthropic models in OpenWebUI. It
combines the best features from multiple community scripts into a single, robust, and
future-proof solution.
Changelog v8.2:
- Improved cache control to use default 5-minute ephemeral caching (per Anthropic spec)
- Improved content block normalization with dedicated helper methods
- Added DISPLAY_THINKING valve to optionally hide Claude's thinking process
- Enhanced thinking display to properly handle redacted_thinking blocks
- Added Claude 4.5 one-sampler logic with configurable temperature/top_p preference
- Updated system message handling with new _prepare_system_blocks method
- Fixed top_k logic to exclude when thinking is enabled (incompatible parameter)
Changelog v8.1:
- FIX: Corrected max_tokens for 128K models from 131,072 to the correct 128,000 limit.
- Added Full Streaming Tool Use: Correctly handles and streams `tool_use` events.
- Corrected Non-Streaming Tool Parsing: Fixed a bug in parsing tool calls from non-streamed responses.
Key Features:
- Dynamic Model Discovery: Fetches models directly from the Anthropic API (`/v1/models`)
  with a graceful fallback to a static list, ensuring it's always up-to-date.
- Asynchronous Core: Uses `aiohttp` for high-performance, non-blocking API calls.
- Complete Feature Set:
    - Extended Thinking: With configurable budgets.
    - 128K Output Tokens: For Claude 3.7 and 4 models.
    - Function Calling / Tool Use: Fully supported in both streaming and non-streaming modes.
- Multi-Modal Support: Image and PDF document processing.
- Intelligent Caching: Reduces costs and latency, with optional display of savings.
- Robust Error Handling: Automatic retries with exponential backoff and jitter.
- Highly Configurable: Uses OpenWebUI Valves to easily toggle features.
"""
import os
import json
import re
import logging
import asyncio
import random
from typing import List, Union, Dict, Optional, AsyncIterator
import aiohttp
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message
class Pipe:
    # Core API and Header Configuration
    API_VERSION = "2023-06-01"
    MESSAGES_URL = "https://api.anthropic.com/v1/messages"
    MODELS_URL = "https://api.anthropic.com/v1/models"
    REQUEST_TIMEOUT = 300
    BETA_HEADERS = {
        "CACHING": "prompt-caching-2024-07-31",
        "PDF": "pdfs-2024-09-25",
        "OUTPUT_128K": "output-128k-2025-02-19",
        "CONTEXT_1M": "context-1m-2025-08-07",
    }
    # Static capability metadata to enrich the dynamic model list
    MODEL_MAX_TOKENS = {
        "claude-3-opus": 4096,
        "claude-3-sonnet": 4096,
        "claude-3-haiku": 4096,
        "claude-3-5-haiku": 8192,
        "claude-3-5-sonnet": 8192,
        "claude-3-7-sonnet": 64000,
        "claude-opus-4": 32000,
        "claude-opus-4-1": 32000,
        "claude-sonnet-4": 64000,
        "claude-sonnet-4-5": 64000,
    }
    MODEL_PRICING = {  # Per million tokens (Input, Cache Write, Cache Read, Output)
        "claude-3-opus": (15.0, 18.75, 1.5, 75.0),
        "claude-3-sonnet": (3.0, 3.75, 0.3, 15.0),
        "claude-3-haiku": (0.25, 0.3125, 0.025, 1.25),
        "claude-3-5-sonnet": (3.0, 3.75, 0.3, 15.0),
        "claude-3-7-sonnet": (3.0, 3.75, 0.3, 15.0),
        "claude-sonnet-4": (4.0, 5.0, 0.4, 20.0),
        "claude-opus-4": (20.0, 25.0, 2.0, 100.0),
    }
    # File and Content Constants
    SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    MAX_IMAGE_SIZE = 5 * 1024 * 1024
    MAX_PDF_SIZE = 32 * 1024 * 1024
    class Valves(BaseModel):
        """Configurable settings for the pipe, adjustable in OpenWebUI."""
        ANTHROPIC_API_KEY: str = Field(default="")
        ENABLE_THINKING: bool = True
        DISPLAY_THINKING: bool = Field(default=True, description="Display Claude's thinking process in the chat")
        MAX_OUTPUT_TOKENS: bool = True
        ENABLE_TOOL_CHOICE: bool = True
        ENABLE_CACHING: bool = True
        SHOW_CACHE_INFO: bool = True
        ENABLE_1M_CONTEXT: bool = False
        CLAUDE_45_USE_TEMPERATURE: bool = Field(default=True, description="For Claude 4.5: Use temperature (True) or top_p (False)")
    def __init__(self):
        logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
        self.type = "manifold"
        self.id = "anthropic"
        self.valves = self.Valves(
            ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY", ""),
            ENABLE_CACHING=os.getenv("ANTHROPIC_ENABLE_CACHING", "true").lower()
            == "true",
            SHOW_CACHE_INFO=os.getenv("ANTHROPIC_SHOW_CACHE_INFO", "true").lower()
            == "true",
        )
        self.request_id = None
        self._models_list_cache = None
    def _get_model_base(self, model_id: str) -> str:
        """Extracts the base name of a model for capability lookups."""
        match = re.search(r"(claude-[\d\.\-a-z]+)-(\d{8}|latest)", model_id)
        if match:
            base_id = match.group(1)
            if "sonnet-4" in base_id:
                return "claude-sonnet-4"
            if "opus-4" in base_id:
                return "claude-opus-4"
            return base_id
        return "claude-3-5-sonnet"
    async def _fetch_and_format_models_from_api(self) -> List[Dict]:
        """Fetches all models from the Anthropic API and formats them for the UI."""
        all_models_raw = []
        params = {"limit": 100}
        headers = {
            "x-api-key": self.valves.ANTHROPIC_API_KEY,
            "anthropic-version": self.API_VERSION,
        }
        async with aiohttp.ClientSession() as session:
            while True:
                async with session.get(
                    self.MODELS_URL, headers=headers, params=params
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    all_models_raw.extend(data.get("data", []))
                    if data.get("has_more") and data.get("last_id"):
                        params["after_id"] = data.get("last_id")
                    else:
                        break
        formatted_models = []
        for model in all_models_raw:
            model_id, display_name = model["id"], model["display_name"]
            base_model = self._get_model_base(model_id)
            supports_thinking = any(
                v in base_model for v in ["3-7", "sonnet-4", "opus-4"]
            )
            formatted_models.append(
                {"id": f"anthropic/{model_id}", "name": display_name}
            )
            if supports_thinking:
                formatted_models.append(
                    {
                        "id": f"anthropic/{model_id}-thinking",
                        "name": f"{display_name} (Extended Thinking)",
                    }
                )
        return sorted(formatted_models, key=lambda x: x["name"])
    def _get_fallback_models(self) -> List[Dict]:
        """Returns a static list of models if the API call fails."""
        logging.warning("Using fallback model list.")
        model_ids = [
            ("claude-3-opus-20240229", "Claude 3 Opus"),
            ("claude-3-5-sonnet-latest", "Claude 3.5 Sonnet (Latest)"),
            ("claude-3-haiku-20240307", "Claude 3 Haiku"),
            ("claude-3-7-sonnet-latest", "Claude 3.7 Sonnet (Latest)"),
            ("claude-sonnet-4-latest", "Claude 4 Sonnet (Latest)"),
        ]
        formatted = []
        for mid, name in model_ids:
            formatted.append({"id": f"anthropic/{mid}", "name": name})
            if any(v in mid for v in ["3-7", "sonnet-4"]):
                formatted.append(
                    {
                        "id": f"anthropic/{mid}-thinking",
                        "name": f"{name} (Extended Thinking)",
                    }
                )
        return formatted
    async def pipes(self) -> List[dict]:
        """Provides the list of available models, fetched dynamically."""
        if self._models_list_cache is None:
            if not self.valves.ANTHROPIC_API_KEY:
                return [{"id": "anthropic/setup", "name": "API Key Not Set"}]
            try:
                self._models_list_cache = await self._fetch_and_format_models_from_api()
            except Exception as e:
                logging.error(
                    f"Failed to fetch models dynamically: {e}. Using fallback list."
                )
                self._models_list_cache = self._get_fallback_models()
        return self._models_list_cache
    def _get_cache_info(self, usage_data: Dict, model_id: str) -> str:
        """Formats cache usage information and cost savings for display."""
        if not self.valves.SHOW_CACHE_INFO or not usage_data:
            return ""
        input_tokens, output_tokens, cached_tokens = (
            usage_data.get("input_tokens", 0),
            usage_data.get("output_tokens", 0),
            usage_data.get("cache_read_input_tokens", 0),
        )
        base_model = self._get_model_base(model_id)
        prices = self.MODEL_PRICING.get(
            base_model, self.MODEL_PRICING["claude-3-5-sonnet"]
        )
        if cached_tokens > 0:
            cache_percentage = (
                (cached_tokens / input_tokens * 100) if input_tokens > 0 else 0
            )
            savings = (
                (input_tokens - cached_tokens) * (prices[0] - prices[2])
            ) / 1_000_000
            return f"```\n✅ CACHE HIT: {cache_percentage:.1f}% cached. Savings: ~${savings:.6f}\n   Tokens: {input_tokens:,} In / {output_tokens:,} Out\n```\n\n"
        else:
            return f"```\n❌ CACHE MISS: No cache used.\n   Tokens: {input_tokens:,} In / {output_tokens:,} Out\n```\n\n"
    def _normalize_content_blocks(self, raw_content):
        """Normalize content into proper block format."""
        blocks = []
        if isinstance(raw_content, list):
            items = raw_content
        else:
            items = [raw_content]

        for item in items:
            if isinstance(item, dict) and item.get("type"):
                blocks.append(dict(item))
            elif isinstance(item, dict) and "content" in item:
                blocks.extend(self._normalize_content_blocks(item["content"]))
            elif item is not None:
                blocks.append({"type": "text", "text": str(item)})
        return blocks

    def _attach_cache_control(self, block: dict):
        """Attach cache control to a content block."""
        if not isinstance(block, dict):
            return block
        if block.get("type") in {"thinking", "redacted_thinking"}:
            return block
        if not block.get("type"):
            block["type"] = "text"
            if "text" not in block:
                block["text"] = ""

        block["cache_control"] = {"type": "ephemeral"}
        return block

    def _prepare_system_blocks(self, system_message):
        """Prepare system message with cache control."""
        if not system_message:
            return None
        content = (
            system_message.get("content")
            if isinstance(system_message, dict) and "content" in system_message
            else system_message
        )
        normalized_blocks = self._normalize_content_blocks(content)
        cached_blocks = [self._attach_cache_control(block) for block in normalized_blocks]
        return cached_blocks if cached_blocks else None

    def _apply_cache_control_to_last_message(self, messages):
        """Apply cache control to the last user message."""
        if not messages:
            return
        last_message = messages[-1]
        if last_message.get("role") != "user":
            return
        for block in reversed(last_message.get("content", [])):
            if isinstance(block, dict) and block.get("type") not in {"thinking", "redacted_thinking"}:
                self._attach_cache_control(block)
                break

    def _process_content_item(self, item: Dict) -> Dict:
        if item["type"] == "image_url":
            url = item["image_url"]["url"]
            if url.startswith("data:image"):
                mime_type, base64_data = url.split(",", 1)
                media_type = mime_type.split(":", 1)[1].split(";", 1)[0]
                if media_type not in self.SUPPORTED_IMAGE_TYPES:
                    raise ValueError(f"Unsupported image type: {media_type}")
                if len(base64_data) * 3 / 4 > self.MAX_IMAGE_SIZE:
                    raise ValueError("Image size exceeds 5MB.")
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data,
                    },
                }
            return {"type": "image", "source": {"type": "url", "url": url}}
        if item["type"] == "pdf_url":
            url = item["pdf_url"]["url"]
            if url.startswith("data:application/pdf"):
                _, base64_data = url.split(",", 1)
                if len(base64_data) * 3 / 4 > self.MAX_PDF_SIZE:
                    raise ValueError("PDF size exceeds 32MB.")
                return {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": base64_data,
                    },
                }
            return {"type": "document", "source": {"type": "url", "url": url}}
        return item
    async def pipe(
        self, body: Dict, __event_emitter__=None
    ) -> Union[str, AsyncIterator[str]]:
        if not self.valves.ANTHROPIC_API_KEY:
            return "Error: ANTHROPIC_API_KEY is not set."
        try:
            system_message, messages = pop_system_message(body["messages"])
            model_id_full = body["model"].split("/")[-1]
            is_thinking_variant = model_id_full.endswith("-thinking")
            model_id = model_id_full.replace("-thinking", "")
            base_model = self._get_model_base(model_id)
            processed_messages, beta_headers_needed = [], set()
            for msg in messages:
                content_list = (
                    msg["content"]
                    if isinstance(msg["content"], list)
                    else [{"type": "text", "text": msg["content"]}]
                )
                processed_content = []
                for item in content_list:
                    if item.get("type") == "pdf_url":
                        beta_headers_needed.add(self.BETA_HEADERS["PDF"])
                    processed_item = self._process_content_item(item)
                    if self.valves.ENABLE_CACHING and item.get("type") in [
                        "tool_calls",
                        "tool_results",
                    ]:
                        processed_item["cache_control"] = {"type": "ephemeral"}
                        beta_headers_needed.add(self.BETA_HEADERS["CACHING"])
                    processed_content.append(processed_item)
                processed_messages.append(
                    {"role": msg["role"], "content": processed_content}
                )
            max_tokens_default = self.MODEL_MAX_TOKENS.get(base_model, 4096)
            requested_max = body.get("max_tokens")

            # Decide output cap and beta headers
            out_cap = max_tokens_default
            if "3-7" in base_model and self.valves.MAX_OUTPUT_TOKENS:
                out_cap = 128000
                beta_headers_needed.add(self.BETA_HEADERS["OUTPUT_128K"])
            if requested_max is not None:
                out_cap = min(out_cap, requested_max)

            # Determine if thinking will be enabled
            will_enable_thinking = is_thinking_variant and self.valves.ENABLE_THINKING

            payload = {
                "model": model_id,
                "messages": processed_messages,
                "max_tokens": out_cap,
                "temperature": body.get("temperature"),
                "top_p": body.get("top_p"),
                "stream": body.get("stream", False),
            }

            # Only include top_k when thinking is NOT enabled
            if not will_enable_thinking:
                payload["top_k"] = body.get("top_k", 40)

            payload = {k: v for k, v in payload.items() if v is not None}

            # Claude 4.5 one-sampler logic
            ONE_SAMPLER_PREFIXES = ("claude-sonnet-4-5", "claude-opus-4")
            if model_id.startswith(ONE_SAMPLER_PREFIXES):
                if will_enable_thinking:
                    payload.pop("top_p", None)
                    payload["temperature"] = 1.0
                elif self.valves.CLAUDE_45_USE_TEMPERATURE:
                    payload.pop("top_p", None)
                    payload["temperature"] = body.get("temperature", 0.8)
                else:
                    payload.pop("temperature", None)
                    payload["top_p"] = body.get("top_p", 0.9)
            # Prepare system message with cache control
            if self.valves.ENABLE_CACHING:
                system_blocks = self._prepare_system_blocks(system_message)
                self._apply_cache_control_to_last_message(processed_messages)
                if system_blocks:
                    payload["system"] = system_blocks
                    beta_headers_needed.add(self.BETA_HEADERS["CACHING"])
            elif system_message:
                payload["system"] = str(system_message)
            if is_thinking_variant and self.valves.ENABLE_THINKING:
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": 32000 if "4" in base_model else 16000,
                }
            if "tools" in body and self.valves.ENABLE_TOOL_CHOICE:
                payload["tools"], payload["tool_choice"] = body["tools"], body.get(
                    "tool_choice"
                )
            if self.valves.ENABLE_1M_CONTEXT and base_model in {"claude-sonnet-4", "claude-sonnet-4-5"}:
                beta_headers_needed.add(self.BETA_HEADERS["CONTEXT_1M"])
            headers = {
                "x-api-key": self.valves.ANTHROPIC_API_KEY,
                "anthropic-version": self.API_VERSION,
                "content-type": "application/json",
            }
            if beta_headers_needed:
                headers["anthropic-beta"] = ",".join(sorted(list(beta_headers_needed)))
            if payload["stream"]:
                return self._stream_response(headers, payload)
            response_data = await self._send_request(headers, payload)
            if isinstance(response_data, str):
                return response_data
            cache_info = self._get_cache_info(response_data.get("usage"), model_id)
            content = response_data.get("content", [])
            if any(c.get("type") == "tool_use" for c in content):
                tool_calls = [
                    {
                        "id": c["tool_use"]["id"],
                        "type": "function",
                        "function": {
                            "name": c["tool_use"]["name"],
                            "arguments": json.dumps(c["tool_use"]["input"]),
                        },
                    }
                    for c in content
                    if c.get("type") == "tool_use"
                ]
                return json.dumps({"type": "tool_calls", "tool_calls": tool_calls})
            response_text = "".join(
                c.get("text", "") for c in content if c.get("type") == "text"
            )
            return cache_info + response_text
        except Exception as e:
            logging.error(f"Error in pipe method: {e}", exc_info=True)
            return f"An unexpected error occurred: {e}"
    async def _stream_response(
        self, headers: Dict, payload: Dict
    ) -> AsyncIterator[str]:
        is_thinking, is_tool_use = False, False
        tool_call_chunks = {}
        try:
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT)
                async with session.post(
                    self.MESSAGES_URL, headers=headers, json=payload, timeout=timeout
                ) as response:
                    self.request_id = response.headers.get("x-request-id")
                    if response.status != 200:
                        yield f"API Error: HTTP {response.status} - {await response.text()}"
                        return
                    async for line in response.content:
                        if not line.startswith(b"data: "):
                            continue
                        data = json.loads(line[6:])
                        event_type = data.get("type")
                        if event_type == "content_block_start":
                            block = data.get("content_block", {})
                            block_type = block.get("type")
                            if block_type == "thinking":
                                is_thinking = True
                                if self.valves.ENABLE_THINKING and self.valves.DISPLAY_THINKING:
                                    yield "<thinking>"
                            elif block_type == "redacted_thinking":
                                is_thinking = True
                            elif block_type == "tool_use":
                                is_tool_use = True
                                tool_use = block.get("tool_use", {})
                                tool_call_chunks[data["index"]] = {
                                    "id": tool_use.get("id"),
                                    "name": tool_use.get("name"),
                                    "input_chunks": [],
                                }
                            else:
                                is_thinking = is_tool_use = False
                        elif event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            if (
                                is_thinking
                                and delta.get("type") == "thinking_delta"
                                and self.valves.ENABLE_THINKING
                                and self.valves.DISPLAY_THINKING
                            ):
                                yield delta.get("thinking", "")
                            elif (
                                is_tool_use and delta.get("type") == "input_json_delta"
                            ):
                                tool_call_chunks[data["index"]]["input_chunks"].append(
                                    delta.get("partial_json", "")
                                )
                            elif (
                                not is_thinking
                                and not is_tool_use
                                and delta.get("type") == "text_delta"
                            ):
                                yield delta.get("text", "")
                        elif event_type == "content_block_stop":
                            if is_thinking and self.valves.ENABLE_THINKING and self.valves.DISPLAY_THINKING:
                                yield "</thinking>"
                            if is_tool_use:
                                tool = tool_call_chunks.get(data["index"])
                                if tool:
                                    full_input = "".join(tool["input_chunks"])
                                    tool_call = {
                                        "id": tool["id"],
                                        "type": "function",
                                        "function": {
                                            "name": tool["name"],
                                            "arguments": full_input,
                                        },
                                    }
                                    yield json.dumps(
                                        {
                                            "type": "tool_calls",
                                            "tool_calls": [tool_call],
                                        }
                                    )
                            is_thinking = is_tool_use = False
                        elif event_type == "message_stop":
                            logging.info(
                                f"Stream finished. {self._get_cache_info(data.get('usage'), payload['model'])}"
                            )
                            break
        except Exception as e:
            logging.error(f"Streaming error: {e}", exc_info=True)
            yield f"Stream Error: {e}"
    async def _send_request(self, headers: Dict, payload: Dict) -> Union[Dict, str]:
        for attempt in range(5):
            try:
                async with aiohttp.ClientSession() as session:
                    timeout = aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT)
                    async with session.post(
                        self.MESSAGES_URL,
                        headers=headers,
                        json=payload,
                        timeout=timeout,
                    ) as response:
                        self.request_id = response.headers.get("x-request-id")
                        if response.status == 200:
                            return await response.json()
                        if response.status in [429, 500, 502, 503, 504] and attempt < 4:
                            delay = int(
                                response.headers.get("Retry-After", 2 ** (attempt + 1))
                            )
                            await asyncio.sleep(delay + random.uniform(0, 1))
                            continue
                        return f"API Error: HTTP {response.status} - {await response.text()}"
            except asyncio.TimeoutError:
                if attempt < 4:
                    await asyncio.sleep((2 ** (attempt + 1)) + random.uniform(0, 1))
                    continue
                return f"API Error: Request timed out after {self.REQUEST_TIMEOUT}s and multiple retries."
            except aiohttp.ClientError as e:
                return f"Network Error: {e}"
        return "API Error: Max retries exceeded."
