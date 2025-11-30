"""
title: Final Unified Anthropic Pipe with Dynamic Discovery, Caching, and Streaming Tools
authors: Balaxxe, nbellochi, Bermont, Mark Kazakov, Christian Taillon (Consolidated & Enhanced by AI)
author_url: https://github.com/christian-taillon
funding_url: https://github.com/open-webui
version: 8.5
license: MIT
requirements: pydantic>=2.0.0, aiohttp>=3.8.0
environment_variables:
    - ANTHROPIC_API_KEY (required): Your Anthropic API key
    - ANTHROPIC_ENABLE_CACHING (optional, default: "true"): Enable prompt caching
    - ANTHROPIC_SHOW_CACHE_INFO (optional, default: "true"): Display cache statistics
    - ANTHROPIC_REQUEST_TIMEOUT (optional, default: "300"): API request timeout in seconds
    - ANTHROPIC_MODEL_CACHE_TTL (optional, default: "3600"): Model list cache TTL in seconds
    - LOG_LEVEL (optional, default: "INFO"): Logging level
This script is the definitive, all-in-one integration for Anthropic models in OpenWebUI. It
combines the best features from multiple community scripts into a single, robust, and
future-proof solution.
Changelog v8.5:
- Removed cost tracking functionality (ENABLE_COST_TRACKING valve, MODEL_PRICING, cost calculation methods)
- Removed cost display from responses and event emissions
- Simplified _get_cache_info to only show token counts and cache hit percentage
- Removed conversation cost storage and tracking

Changelog v8.3:
- Added comprehensive event emitter integration for user visibility
- Implemented _emit_status() and _emit_message() helper methods
- Added status updates for non-streaming requests (connecting, success, failure)
- Added progress indicators for streaming responses (thinking, tool use, completion)
- Made cache info visible at end of streaming responses (previously only logged)
- Added event emissions for all error conditions with proper status updates
- Enhanced user experience with real-time UI feedback throughout request lifecycle

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
from typing import List, Union, Dict, Optional, AsyncGenerator, Any
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
        "claude-haiku-4-5": 64000,
        "claude-opus-4-5": 64000,
    }

    # File and Content Constants
    SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    MAX_IMAGE_SIZE = 5 * 1024 * 1024
    MAX_PDF_SIZE = 32 * 1024 * 1024
    class Valves(BaseModel):
        """Configurable settings for the Anthropic pipe."""

        ANTHROPIC_API_KEY: str = Field(
            default="",
            description="Your Anthropic API key (get from console.anthropic.com)"
        )

        ENABLE_THINKING: bool = Field(
            default=True,
            description="Enable extended thinking for Claude 3.7+ and 4.x models"
        )

        DISPLAY_THINKING: bool = Field(
            default=True,
            description="Display Claude's thinking process in chat (when thinking enabled)"
        )

        MAX_OUTPUT_TOKENS: bool = Field(
            default=True,
            description="Use 128k maximum output tokens for Sonnet 3.7)"
        )

        ENABLE_TOOL_CHOICE: bool = Field(
            default=True,
            description="Enable tool/function calling capabilities"
        )

        ENABLE_CACHING: bool = Field(
            default=True,
            description="Enable prompt caching to reduce costs and latency"
        )

        SHOW_CACHE_INFO: bool = Field(
            default=True,
            description="Display cache hit statistics and cost savings"
        )

        ENABLE_1M_CONTEXT: bool = Field(
            default=False,
            description="Enable 1M token context for Claude 4.x (beta feature)"
        )

        CLAUDE_45_USE_TEMPERATURE: bool = Field(
            default=True,
            description="Claude 4.5: Use temperature (True) or top_p (False) sampling"
        )

        REQUEST_TIMEOUT: int = Field(
            default=300,
            description="API request timeout in seconds (default: 5 minutes)"
        )

        MODEL_CACHE_TTL: int = Field(
            default=3600,
            description="Model list cache duration in seconds (default: 1 hour, 0 to disable)"
        )


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
            REQUEST_TIMEOUT=int(os.getenv("ANTHROPIC_REQUEST_TIMEOUT", "300")),
            MODEL_CACHE_TTL=int(os.getenv("ANTHROPIC_MODEL_CACHE_TTL", "3600")),
        )
        self.request_id = None
        self._models_list_cache = None
        self._models_cache_time = None
    def _get_model_base(self, model_id: str) -> str:
        """Extracts the base name of a model for capability lookups."""
        match = re.search(r"(claude-[\d.\-a-z]+)-(\d{8}|latest)", model_id)
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

    async def _emit_status(
        self,
        __event_emitter__: Optional[Any],
        description: str,
        done: bool = False
    ) -> None:
        """
        Emit status event to UI.

        Args:
            __event_emitter__: Event emitter from OpenWebUI
            description: Status description to display
            done: Whether the status is complete
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": description,
                        "done": done
                    }
                }
            )

    def _format_error(
        self,
        message: str,
        error_code: str = "UNKNOWN",
        http_status: Optional[int] = None,
        request_id: Optional[str] = None
    ) -> str:
        """
        Format error responses consistently.

        Args:
            message: Human-readable error message
            error_code: Error category (API_ERROR, VALIDATION_ERROR, TIMEOUT, etc.)
            http_status: HTTP status code if from API
            request_id: Anthropic request ID for debugging

        Returns:
            Formatted error string for display
        """
        error_parts = [f"❌ {error_code}"]

        if http_status:
            error_parts.append(f"HTTP {http_status}")

        if request_id:
            error_parts.append(f"[Request: {request_id}]")

        error_parts.append(f"\n{message}")

        return " | ".join(error_parts[:3]) + error_parts[-1]

    def _get_cache_info(self, usage_data: Dict, model_id: str) -> str:
        """Formats cache usage information for display."""
        if not self.valves.SHOW_CACHE_INFO or not usage_data:
            return ""
        input_tokens, output_tokens, cached_tokens = (
            usage_data.get("input_tokens", 0),
            usage_data.get("output_tokens", 0),
            usage_data.get("cache_read_input_tokens", 0),
        )
        if cached_tokens > 0:
            cache_percentage = (
                (cached_tokens / input_tokens * 100) if input_tokens > 0 else 0
            )
            return f"```\n✅ CACHE HIT: {cache_percentage:.1f}% cached.\n   Tokens: {input_tokens:,} In / {output_tokens:,} Out\n```\n\n"
        else:
            return f"```\n❌ CACHE MISS: No cache used.\n   Tokens: {input_tokens:,} In / {output_tokens:,} Out\n```\n\n"


    def _normalize_content_blocks(
        self,
        raw_content: Union[List, Dict, str],
        _depth: int = 0,
        _visited: Optional[set] = None
    ) -> List[Dict]:
        """
        Normalize content into proper block format with safety checks.

        Args:
            raw_content: Content to normalize (list, dict, or string)
            _depth: Current recursion depth (internal use)
            _visited: Set of visited object IDs to detect circular refs (internal use)

        Returns:
            List of normalized content blocks

        Raises:
            ValueError: If circular reference or max depth exceeded
        """
        # Initialize visited set on first call
        if _visited is None:
            _visited = set()

        # Check maximum nesting depth (prevent stack overflow)
        MAX_DEPTH = 10
        if _depth > MAX_DEPTH:
            logging.error(f"Content nesting exceeds maximum depth of {MAX_DEPTH}")
            raise ValueError(
                f"Content structure too deeply nested (max {MAX_DEPTH} levels). "
                "This may indicate a circular reference or invalid structure."
            )

        # Circular reference detection for mutable objects
        if isinstance(raw_content, (list, dict)):
            obj_id = id(raw_content)
            if obj_id in _visited:
                logging.error(f"Circular reference detected in content at depth {_depth}")
                raise ValueError(
                    "Circular reference detected in content structure. "
                    "Content cannot reference itself."
                )
            _visited.add(obj_id)

        blocks = []

        # Convert to list if needed
        if isinstance(raw_content, list):
            items = raw_content
        else:
            items = [raw_content]

        # Process each item
        for idx, item in enumerate(items):
            try:
                if isinstance(item, dict) and item.get("type"):
                    # Direct block - copy and continue
                    blocks.append(dict(item))

                elif isinstance(item, dict) and "content" in item:
                    # Nested content - recurse with depth tracking
                    nested_blocks = self._normalize_content_blocks(
                        item["content"],
                        _depth=_depth + 1,
                        _visited=_visited.copy()  # Copy to allow sibling references
                    )
                    blocks.extend(nested_blocks)

                elif item is not None:
                    # Fallback - convert to text
                    # Handle unexpected types gracefully
                    try:
                        text_value = str(item)
                    except Exception as e:
                        logging.warning(
                            f"Failed to convert content item to string at depth {_depth}, "
                            f"index {idx}: {type(item).__name__}. Error: {e}"
                        )
                        text_value = f"<unconvertible {type(item).__name__}>"

                    blocks.append({"type": "text", "text": text_value})

                # else: skip None items

            except ValueError:
                # Re-raise validation errors (circular ref, max depth)
                raise

            except Exception as e:
                # Log unexpected errors but continue processing
                logging.error(
                    f"Error processing content item at depth {_depth}, index {idx}: {e}",
                    exc_info=True
                )
                # Add error indicator instead of crashing
                blocks.append({
                    "type": "text",
                    "text": f"[Error processing content item {idx}: {str(e)[:100]}]"
                })

        return blocks

    def _attach_cache_control(self, block: Dict) -> Dict:
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

    def _prepare_system_blocks(self, system_message: Union[Dict, str, None]) -> Optional[List[Dict]]:
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

    def _apply_cache_control_to_last_message(self, messages: List[Dict]) -> None:
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
        """Process a content item with comprehensive validation."""

        # Validate item structure
        if not isinstance(item, dict):
            raise ValueError(f"Content item must be a dict, got {type(item).__name__}")

        if "type" not in item:
            raise ValueError(f"Content item missing required 'type' field: {item}")

        item_type = item["type"]

        # Process image_url type
        if item_type == "image_url":
            if "image_url" not in item:
                raise ValueError("image_url type requires 'image_url' field")

            if not isinstance(item["image_url"], dict) or "url" not in item["image_url"]:
                raise ValueError("image_url must contain {'url': '...'}")

            url = item["image_url"]["url"]

            if url.startswith("data:image"):
                # Existing base64 processing with validation
                try:
                    mime_type, base64_data = url.split(",", 1)
                    media_type = mime_type.split(":", 1)[1].split(";", 1)[0]
                except (ValueError, IndexError) as e:
                    raise ValueError(f"Invalid data URL format: {e}")

                if media_type not in self.SUPPORTED_IMAGE_TYPES:
                    raise ValueError(
                        f"Unsupported image type: {media_type}. "
                        f"Supported types: {', '.join(self.SUPPORTED_IMAGE_TYPES)}"
                    )

                # Validate size
                estimated_size = len(base64_data) * 3 / 4
                if estimated_size > self.MAX_IMAGE_SIZE:
                    raise ValueError(
                        f"Image size ({estimated_size / 1024 / 1024:.2f}MB) exceeds "
                        f"maximum {self.MAX_IMAGE_SIZE / 1024 / 1024}MB"
                    )

                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data,
                    },
                }

            # URL-based image
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"Image URL must start with http:// or https://, got: {url[:50]}...")

            return {"type": "image", "source": {"type": "url", "url": url}}

        # Process pdf_url type
        if item_type == "pdf_url":
            if "pdf_url" not in item:
                raise ValueError("pdf_url type requires 'pdf_url' field")

            if not isinstance(item["pdf_url"], dict) or "url" not in item["pdf_url"]:
                raise ValueError("pdf_url must contain {'url': '...'}")

            url = item["pdf_url"]["url"]

            if url.startswith("data:application/pdf"):
                try:
                    _, base64_data = url.split(",", 1)
                except ValueError as e:
                    raise ValueError(f"Invalid PDF data URL format: {e}")

                estimated_size = len(base64_data) * 3 / 4
                if estimated_size > self.MAX_PDF_SIZE:
                    raise ValueError(
                        f"PDF size ({estimated_size / 1024 / 1024:.2f}MB) exceeds "
                        f"maximum {self.MAX_PDF_SIZE / 1024 / 1024}MB"
                    )

                return {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": base64_data,
                    },
                }

            # URL-based PDF
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"PDF URL must start with http:// or https://, got: {url[:50]}...")

            return {"type": "document", "source": {"type": "url", "url": url}}

        # Pass through other types unchanged
        return item
    async def pipe(
        self,
        body: Dict,
        __user__: Optional[Dict] = None,
        __event_emitter__: Optional[Any] = None,
    ) -> Union[str, AsyncGenerator[str, None]]:
        if not self.valves.ANTHROPIC_API_KEY:
            return "Error: ANTHROPIC_API_KEY is not set."

        # ===== INPUT VALIDATION =====

        # Validate required fields
        if "messages" not in body:
            return "Error: Missing required field 'messages' in request body"

        if not isinstance(body["messages"], list):
            return f"Error: 'messages' must be a list, got {type(body['messages']).__name__}"

        if len(body["messages"]) == 0:
            return "Error: 'messages' list cannot be empty"

        if "model" not in body:
            return "Error: Missing required field 'model' in request body"

        # Validate model format
        if not isinstance(body["model"], str) or "/" not in body["model"]:
            return f"Error: Invalid model format '{body.get('model')}'. Expected format: 'anthropic/model-name'"

        # Validate optional numeric parameters
        if "max_tokens" in body:
            max_tokens = body["max_tokens"]
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                return f"Error: max_tokens must be a positive integer, got {max_tokens}"

        if "temperature" in body:
            temp = body["temperature"]
            if not isinstance(temp, (int, float)) or not (0 <= temp <= 2):
                return f"Error: temperature must be between 0 and 2, got {temp}"

        if "top_p" in body:
            top_p = body["top_p"]
            if not isinstance(top_p, (int, float)) or not (0 <= top_p <= 1):
                return f"Error: top_p must be between 0 and 1, got {top_p}"

        try:
            system_message, messages = pop_system_message(body["messages"])
            model_id_full = body["model"].split("/")[-1]
            is_thinking_variant = model_id_full.endswith("-thinking")
            model_id = model_id_full.replace("-thinking", "")
            base_model = self._get_model_base(model_id)
            processed_messages, beta_headers_needed = [], set()
            try:
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
            except ValueError as e:
                # Validation errors - return to user
                return f"Content validation error: {e}"
            except Exception as e:
                # Unexpected errors - log and return
                logging.error(f"Error processing message content: {e}", exc_info=True)
                return f"Failed to process message content: {e}"
            max_tokens_default = self.MODEL_MAX_TOKENS.get(base_model, 64000)
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

            logging.debug(f"max_tokens: {out_cap}")
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
                default_thinking_budget = 16000 if re.search(r'[a-z]-3\b', base_model) else 32000
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": min(default_thinking_budget, out_cap - 1),
                }
            if "tools" in body and self.valves.ENABLE_TOOL_CHOICE:
                payload["tools"] = body["tools"]
                if body.get("tool_choice"):
                    payload["tool_choice"] = body.get("tool_choice")
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
                return self._stream_response(headers, payload, __event_emitter__, body)
            
            # Emit start status for non-streaming
            await self._emit_status(__event_emitter__, "Sending request to Anthropic API...")

            response_data = await self._send_request(headers, payload)
            if isinstance(response_data, str):
                # Error response
                await self._emit_status(__event_emitter__, "Request failed", done=True)
                return response_data

            # Emit success status
            await self._emit_status(__event_emitter__, "Response received", done=True)

            cache_info = self._get_cache_info(response_data.get("usage"), model_id)

            # Emit cache info if enabled
            if self.valves.SHOW_CACHE_INFO and response_data.get("usage"):
                usage = response_data.get("usage")
                cached_tokens = usage.get("cache_read_input_tokens", 0)
                if cached_tokens > 0:
                    await self._emit_status(
                        __event_emitter__,
                        f"Cache hit: {cached_tokens:,} tokens served from cache",
                        True
                    )

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
            
            return response_text
        except Exception as e:
            logging.error(f"Error in pipe method: {e}", exc_info=True)
            return f"An unexpected error occurred: {e}"
    async def _stream_response(
        self, headers: Dict, payload: Dict, __event_emitter__: Optional[Any] = None,
        body: Optional[Dict] = None
    ) -> AsyncGenerator[str, None]:
        is_thinking, is_tool_use = False, False
        tool_call_chunks = {}
        usage_data = None

        try:
            # Emit connection status
            await self._emit_status(
                __event_emitter__,
                "Connecting to Anthropic API...",
                done=False
            )

            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=self.valves.REQUEST_TIMEOUT)
                async with session.post(
                    self.MESSAGES_URL, headers=headers, json=payload, timeout=timeout
                ) as response:
                    self.request_id = response.headers.get("x-request-id")
                    logging.info(f"Streaming request initiated [Request ID: {self.request_id}]")
                    if response.status != 200:
                        error_msg = self._format_error(
                            message=await response.text(),
                            error_code="API_ERROR",
                            http_status=response.status,
                            request_id=self.request_id
                        )
                        # Emit error event
                        await self._emit_status(
                            __event_emitter__,
                            "Request failed",
                            done=True
                        )
                        yield error_msg
                        return

                    # Emit streaming started
                    await self._emit_status(
                        __event_emitter__,
                        "Streaming response...",
                        done=False
                    )

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
                                # Emit thinking started event
                                await self._emit_status(
                                    __event_emitter__,
                                    "Claude is thinking...",
                                    done=False
                                )
                                if self.valves.ENABLE_THINKING and self.valves.DISPLAY_THINKING:
                                    yield "<thinking>"
                            elif block_type == "redacted_thinking":
                                is_thinking = True
                                # Emit redacted thinking event
                                await self._emit_status(
                                    __event_emitter__,
                                    "Claude is thinking (redacted)...",
                                    done=False
                                )
                            elif block_type == "tool_use":
                                is_tool_use = True
                                tool_use = block.get("tool_use", {})
                                tool_name = tool_use.get("name", "unknown")
                                # Emit tool use detected
                                await self._emit_status(
                                    __event_emitter__,
                                    f"Using tool: {tool_name}",
                                    done=False
                                )
                                tool_call_chunks[data["index"]] = {
                                    "id": tool_use.get("id"),
                                    "name": tool_name,
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
                            if is_thinking:
                                # Emit thinking complete
                                await self._emit_status(
                                    __event_emitter__,
                                    "Thinking complete",
                                    done=False
                                )
                                if self.valves.ENABLE_THINKING and self.valves.DISPLAY_THINKING:
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
                            # Capture usage data
                            usage_data = data.get('usage')

                            # Emit completion status
                            await self._emit_status(
                                __event_emitter__,
                                "Stream complete",
                                done=True
                            )

                            usage_info = self._get_cache_info(usage_data, payload['model'])
                            logging.info(
                                f"Stream finished [Request ID: {self.request_id}]. {usage_info}"
                            )
                            break

                    # Yield cache info at end of stream
                    if self.valves.SHOW_CACHE_INFO and usage_data:
                        cache_info = self._get_cache_info(usage_data, payload['model'])
                        if cache_info:
                            yield cache_info
        except asyncio.TimeoutError as e:
            logging.error(f"Streaming timeout: {e}", exc_info=True)
            error_msg = self._format_error(
                message=f"Request timed out after {self.valves.REQUEST_TIMEOUT}s",
                error_code="TIMEOUT",
                request_id=self.request_id
            )
            # Emit error event
            await self._emit_status(
                __event_emitter__,
                "Request timed out",
                done=True
            )
            yield error_msg
        except aiohttp.ClientError as e:
            logging.error(f"Streaming network error: {e}", exc_info=True)
            error_msg = self._format_error(
                message=str(e),
                error_code="NETWORK_ERROR",
                request_id=self.request_id
            )
            # Emit error event
            await self._emit_status(
                __event_emitter__,
                "Network error occurred",
                done=True
            )
            yield error_msg
        except Exception as e:
            logging.error(f"Streaming error: {e}", exc_info=True)
            error_msg = self._format_error(
                message=str(e),
                error_code="STREAM_ERROR",
                request_id=self.request_id
            )
            # Emit error event
            await self._emit_status(
                __event_emitter__,
                "Streaming error occurred",
                done=True
            )
            yield error_msg
    async def _send_request(self, headers: Dict, payload: Dict) -> Union[Dict, str]:
        for attempt in range(5):
            try:
                async with aiohttp.ClientSession() as session:
                    timeout = aiohttp.ClientTimeout(total=self.valves.REQUEST_TIMEOUT)
                    async with session.post(
                        self.MESSAGES_URL,
                        headers=headers,
                        json=payload,
                        timeout=timeout,
                    ) as response:
                        self.request_id = response.headers.get("x-request-id")
                        logging.info(f"Non-streaming request sent [Request ID: {self.request_id}]")
                        if response.status == 200:
                            logging.info(f"Request successful [Request ID: {self.request_id}]")
                            return await response.json()
                        if response.status in [429, 500, 502, 503, 504] and attempt < 4:
                            delay = int(
                                response.headers.get("Retry-After", 2 ** (attempt + 1))
                            )
                            await asyncio.sleep(delay + random.uniform(0, 1))
                            continue
                        return self._format_error(
                            message=await response.text(),
                            error_code="API_ERROR",
                            http_status=response.status,
                            request_id=self.request_id
                        )
            except asyncio.TimeoutError:
                if attempt < 4:
                    await asyncio.sleep((2 ** (attempt + 1)) + random.uniform(0, 1))
                    continue
                return self._format_error(
                    message=f"Request timed out after {self.valves.REQUEST_TIMEOUT}s and multiple retries",
                    error_code="TIMEOUT",
                    request_id=self.request_id
                )
            except aiohttp.ClientError as e:
                return self._format_error(
                    message=str(e),
                    error_code="NETWORK_ERROR",
                    request_id=self.request_id
                )
        return self._format_error(
            message="Max retries exceeded",
            error_code="MAX_RETRIES",
            request_id=self.request_id
        )

    async def outlet(
        self, body: dict, __user__: Optional[dict] = None, __event_emitter__=None
    ) -> dict:
        """
        Process the response body after the pipe completes.
        Note: Cost tracking is now handled by injecting into response text.
        """
        return body
