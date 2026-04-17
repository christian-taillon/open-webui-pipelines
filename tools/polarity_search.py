"""
title: Polarity Entity Search and Auto-Enrichment
author: christian-taillon
author_url: https://christiant.io/
funding_url: https://github.com/open-webui/open-webui
github_url: https://github.com/open-webui/open-webui
version: 5.0.0
license: MIT
description: Searches entities through Polarity (polarity.io) integrations with automatic context enrichment for model responses.
requirements: requests
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Dict, List, Optional

import requests
from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        polarity_api_url: str = Field(
            default="",
            description="Polarity server base URL (e.g., https://polarity.example.com).",
        )
        polarity_bearer_token: str = Field(
            default="",
            description="Polarity API Bearer Token.",
            sensitive=True,
        )
        integrations_to_search: str = Field(
            default="ALL",
            description="Comma-separated list of integration IDs to search, or 'ALL' to search all available integrations.",
        )
        auto_enrich_enabled: bool = Field(
            default=True,
            description="Enable automatic entity enrichment for all messages.",
        )
        auto_enrich_integrations: str = Field(
            default="ALL",
            description="Comma-separated list of integration IDs for auto-enrichment, or 'ALL' for all. If empty, uses integrations_to_search.",
        )
        cache_duration_minutes: int = Field(
            default=15,
            description="Duration in minutes to cache entity lookup results.",
        )
        rate_limit_delay_seconds: float = Field(
            default=0.5,
            description="Delay in seconds between integration API calls.",
        )
        max_entities_per_request: int = Field(
            default=10,
            description="Maximum number of entities to lookup per request.",
        )
        include_annotations: bool = Field(
            default=True,
            description="Include Polarity annotations/tags in results.",
        )
        include_details: bool = Field(
            default=True,
            description="Include integration detail payloads in lookup results.",
        )
        max_summary_items_per_result: int = Field(
            default=5,
            description="Maximum number of summary lines to keep per integration result.",
        )
        timeout_seconds: int = Field(
            default=30,
            description="API request timeout in seconds.",
        )
        debug_mode: bool = Field(
            default=False,
            description="Enable debug logging for troubleshooting.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_request_time: Dict[str, float] = {}
        self._integration_cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._integration_cache_time: Optional[datetime] = None

    def _debug_log(self, message: str) -> None:
        """Log debug messages if debug mode is enabled."""
        if self.valves.debug_mode:
            print(f"[Polarity Debug] {message}")

    async def _emit_status(
        self,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]],
        description: str,
        *,
        status: str = "in_progress",
        done: bool = False,
    ) -> None:
        """Emit a status event if the caller provided an emitter."""
        if not __event_emitter__:
            return

        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "status": status,
                    "description": description,
                    "done": done,
                },
            }
        )

    def _get_cache_key(self, entity_value: str, integration_id: str) -> str:
        """Generate cache key for entity-integration pair."""
        return hashlib.md5(f"{entity_value}:{integration_id}".encode()).hexdigest()

    def _is_cache_valid(self, cache_time: Optional[datetime]) -> bool:
        """Check if cached data is still valid."""
        if not cache_time:
            return False
        return datetime.now() - cache_time < timedelta(
            minutes=self.valves.cache_duration_minutes
        )

    async def _apply_rate_limit(self, integration_id: str) -> None:
        """Apply rate limiting per integration without blocking the event loop."""
        if integration_id in self._last_request_time:
            elapsed = time.time() - self._last_request_time[integration_id]
            if elapsed < self.valves.rate_limit_delay_seconds:
                await asyncio.sleep(self.valves.rate_limit_delay_seconds - elapsed)
        self._last_request_time[integration_id] = time.time()

    def _api_not_configured(self) -> bool:
        return not self.valves.polarity_api_url or not self.valves.polarity_bearer_token

    def _build_error(
        self,
        code: str,
        message: str,
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"code": code, "message": message}
        if details:
            payload["details"] = details
        return payload

    def _sanitize_summary(self, summary: Any) -> List[str]:
        if not isinstance(summary, list):
            return []
        sanitized = []
        for item in summary:
            if item is None:
                continue
            sanitized.append(str(item).strip())
        return [item for item in sanitized if item][
            : self.valves.max_summary_items_per_result
        ]

    def _make_api_request_sync(
        self,
        method: str,
        endpoint: str,
        *,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make authenticated API request to Polarity and normalize errors."""
        if self._api_not_configured():
            self._debug_log("API URL or Bearer token not configured")
            return {
                "ok": False,
                "error": self._build_error(
                    "not_configured",
                    "Polarity API not configured. Set polarity_api_url and polarity_bearer_token in tool valves.",
                ),
            }

        url = f"{self.valves.polarity_api_url.rstrip('/')}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.valves.polarity_bearer_token}",
            "Content-Type": "application/vnd.api+json",
            "Accept": "application/vnd.api+json, application/json",
        }

        self._debug_log(f"Making {method} request to {url}")

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                params=params,
                timeout=self.valves.timeout_seconds,
            )
            response.raise_for_status()
            result = response.json()
            self._debug_log(f"Request successful: {response.status_code}")
            return {"ok": True, "data": result}
        except requests.exceptions.HTTPError as exc:
            response = exc.response
            status_code = response.status_code if response is not None else None
            response_text = ""
            if response is not None:
                response_text = response.text[:500]
            self._debug_log(
                f"HTTP error calling {endpoint}: {status_code} {response_text}"
            )
            return {
                "ok": False,
                "error": self._build_error(
                    "http_error",
                    f"Polarity API returned HTTP {status_code} for {endpoint}.",
                    details={
                        "endpoint": endpoint,
                        "status_code": status_code,
                        "response_body": response_text,
                    },
                ),
            }
        except requests.exceptions.Timeout:
            self._debug_log(f"Timeout calling {endpoint}")
            return {
                "ok": False,
                "error": self._build_error(
                    "timeout",
                    f"Timed out calling {endpoint}.",
                    details={"endpoint": endpoint, "timeout_seconds": self.valves.timeout_seconds},
                ),
            }
        except requests.exceptions.RequestException as exc:
            self._debug_log(f"API request failed: {str(exc)}")
            return {
                "ok": False,
                "error": self._build_error(
                    "request_error",
                    f"Failed to call {endpoint}.",
                    details={"endpoint": endpoint, "error": str(exc)},
                ),
            }
        except ValueError as exc:
            self._debug_log(f"Invalid JSON from {endpoint}: {str(exc)}")
            return {
                "ok": False,
                "error": self._build_error(
                    "invalid_json",
                    f"Polarity API returned invalid JSON for {endpoint}.",
                    details={"endpoint": endpoint, "error": str(exc)},
                ),
            }

    async def _make_api_request(
        self,
        method: str,
        endpoint: str,
        *,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self._make_api_request_sync,
            method,
            endpoint,
            data=data,
            params=params,
        )

    async def _get_integrations(self) -> Dict[str, Any]:
        """Fetch and cache available integrations."""
        if self._integration_cache and self._is_cache_valid(
            self._integration_cache_time
        ):
            self._debug_log(
                f"Using cached integrations: {len(self._integration_cache)} available"
            )
            return {"ok": True, "data": self._integration_cache}

        self._debug_log("Fetching integrations from API")
        result = await self._make_api_request(
            "GET",
            "/api/integrations",
            params={"filter[integration.status]": "running", "page[size]": 200},
        )

        if not result["ok"]:
            self._integration_cache = {}
            return result

        payload = result["data"]
        if "data" not in payload:
            self._integration_cache = {}
            return {
                "ok": False,
                "error": self._build_error(
                    "invalid_integrations_response",
                    "Polarity integrations response did not include a data field.",
                ),
            }

        integrations: Dict[str, Dict[str, Any]] = {}
        for item in payload["data"]:
            integration_id = item.get("id")
            if not integration_id:
                continue
            attributes = item.get("attributes", {})
            integrations[integration_id] = {
                "id": integration_id,
                "name": attributes.get("name", integration_id),
                "acronym": attributes.get("acronym", ""),
                "status": attributes.get("status", "unknown"),
                "description": attributes.get("description", ""),
            }

        self._integration_cache = integrations
        self._integration_cache_time = datetime.now()
        self._debug_log(f"Found {len(integrations)} running integrations")
        return {"ok": True, "data": integrations}

    async def _parse_entities(self, text: str) -> Dict[str, Any]:
        """Parse text to extract entities."""
        self._debug_log(f"Parsing entities from text: {text[:100]}...")

        result = await self._make_api_request(
            "POST",
            "/api/parsed-entities",
            data={"data": {"attributes": {"text": text}}},
        )

        if not result["ok"]:
            self._debug_log("Failed to parse entities")
            return result

        payload = result["data"]
        attributes = payload.get("data", {}).get("attributes", {})
        entities = attributes.get("entities", [])
        annotations = attributes.get("annotations", [])

        self._debug_log(f"Found {len(entities)} entities")

        standard_types = {
            "IPv4",
            "IPv6",
            "domain",
            "email",
            "url",
            "MD5",
            "SHA1",
            "SHA256",
            "IPv4CIDR",
        }

        filtered_entities = [
            entity for entity in entities if entity.get("type") in standard_types
        ]
        self._debug_log(f"Filtered to {len(filtered_entities)} standard entities")

        if self.valves.include_annotations and annotations:
            entity_annotations: Dict[str, List[str]] = {}
            for annotation in annotations:
                entity_name = annotation.get("entity", {}).get("entity-name", "")
                if not entity_name:
                    continue
                tags = []
                for context in annotation.get("contexts", []):
                    tag_name = context.get("tag", {}).get("tag-name", "")
                    if tag_name:
                        tags.append(tag_name)
                if tags:
                    entity_annotations[entity_name] = sorted(set(tags))

            for entity in filtered_entities:
                entity_value = entity.get("value", "")
                if entity_value in entity_annotations:
                    entity["annotations"] = entity_annotations[entity_value]

        return {"ok": True, "data": filtered_entities}

    def _normalize_lookup_result(
        self,
        lookup_result: Dict[str, Any],
        fallback_entity: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        data = lookup_result.get("data")
        if not data:
            return None

        normalized = {
            "summary": self._sanitize_summary(data.get("summary", [])),
            "entity": lookup_result.get("entity", fallback_entity),
        }

        details = data.get("details")
        if self.valves.include_details and details:
            normalized["details"] = details

        return normalized

    async def _lookup_entities(
        self, entities: List[Dict[str, Any]], integration_id: str
    ) -> Dict[str, Any]:
        """Lookup entities in a specific integration."""
        if not entities:
            return {"ok": True, "data": {}, "errors": []}

        self._debug_log(f"Looking up {len(entities)} entities in {integration_id}")

        results: Dict[str, Any] = {}
        entities_to_lookup: List[Dict[str, Any]] = []

        for entity in entities:
            cache_key = self._get_cache_key(entity["value"], integration_id)
            if cache_key in self._cache and self._is_cache_valid(
                self._cache[cache_key]["time"]
            ):
                results[entity["value"]] = self._cache[cache_key]["data"]
                self._debug_log(f"Using cached result for {entity['value']}")
            else:
                entities_to_lookup.append(entity)

        errors: List[Dict[str, Any]] = []
        if not entities_to_lookup:
            return {"ok": True, "data": results, "errors": errors}

        await self._apply_rate_limit(integration_id)

        for batch_start in range(
            0, len(entities_to_lookup), self.valves.max_entities_per_request
        ):
            batch = entities_to_lookup[
                batch_start : batch_start + self.valves.max_entities_per_request
            ]
            lookup_result = await self._make_api_request(
                "POST",
                f"/api/integrations/{integration_id}/lookup",
                data={
                    "data": {
                        "type": "integration-lookups",
                        "attributes": {"entities": batch},
                    }
                },
            )

            if not lookup_result["ok"]:
                error = lookup_result["error"]
                error["details"] = {
                    **error.get("details", {}),
                    "integration_id": integration_id,
                }
                errors.append(error)
                self._debug_log(f"Lookup request failed for {integration_id}")
                continue

            response_results = (
                lookup_result.get("data", {})
                .get("data", {})
                .get("attributes", {})
                .get("results", [])
            )
            self._debug_log(
                f"Got {len(response_results)} results from {integration_id}"
            )

            for index, result in enumerate(response_results):
                if index >= len(batch):
                    break

                entity_value = batch[index]["value"]
                processed_result = self._normalize_lookup_result(result, batch[index])
                if not processed_result:
                    self._debug_log(
                        f"No data found for {entity_value} in {integration_id}"
                    )
                    continue

                results[entity_value] = processed_result
                cache_key = self._get_cache_key(entity_value, integration_id)
                self._cache[cache_key] = {
                    "data": processed_result,
                    "time": datetime.now(),
                }

        return {"ok": True, "data": results, "errors": errors}

    async def _get_integrations_to_search(
        self, integration_ids: Optional[str] = None
    ) -> Dict[str, Any]:
        """Determine which integrations to search based on configuration."""
        integration_result = await self._get_integrations()
        if not integration_result["ok"]:
            return integration_result

        available_integrations = integration_result["data"]
        config_value = (
            integration_ids.strip()
            if integration_ids is not None and integration_ids.strip()
            else self.valves.integrations_to_search.strip()
        )

        if config_value.upper() == "ALL":
            requested_integrations = list(available_integrations.keys())
        elif config_value:
            requested_integrations = [
                integration_id.strip()
                for integration_id in config_value.split(",")
                if integration_id.strip()
            ]
        else:
            requested_integrations = []

        valid_integrations = [
            integration_id
            for integration_id in requested_integrations
            if integration_id in available_integrations
            and available_integrations[integration_id]["status"] == "running"
        ]

        invalid_integrations = [
            integration_id
            for integration_id in requested_integrations
            if integration_id not in valid_integrations
        ]

        self._debug_log(f"Valid running integrations: {valid_integrations}")

        return {
            "ok": True,
            "data": {
                "valid": valid_integrations,
                "invalid": invalid_integrations,
                "available": available_integrations,
            },
        }

    def _format_enrichment_context(self, enrichment_data: Dict[str, Any]) -> str:
        """Format enrichment data for model context."""
        if not enrichment_data or not enrichment_data.get("entities"):
            return ""

        context_lines = ["[Entity Enrichment Data]"]
        for entity_info in enrichment_data["entities"]:
            context_lines.append(
                f"\n• {entity_info['value']} ({entity_info['type']}):"
            )

            if entity_info.get("annotations"):
                context_lines.append(
                    f"  Tags: {', '.join(entity_info['annotations'])}"
                )

            for integration_id, data in entity_info.get(
                "integration_results", {}
            ).items():
                summary = data.get("summary", [])
                if summary:
                    context_lines.append(
                        f"  {integration_id}: {' | '.join(summary)}"
                    )

        return "\n".join(context_lines)

    async def _build_search_results(
        self,
        entities: List[Dict[str, Any]],
        valid_integrations: List[str],
        *,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        results = {
            "entities": [],
            "integrations_searched": valid_integrations,
            "timestamp": datetime.now().isoformat(),
            "errors": [],
        }

        total_entities = len(entities)
        for index, entity in enumerate(entities, start=1):
            await self._emit_status(
                __event_emitter__,
                f"Looking up entity {index}/{total_entities}: {entity['value']}",
            )

            entity_result = {
                "value": entity["value"],
                "type": entity["type"],
                "annotations": entity.get("annotations", []),
                "integration_results": {},
            }

            for integration_id in valid_integrations:
                lookup_results = await self._lookup_entities([entity], integration_id)
                if entity["value"] in lookup_results["data"]:
                    entity_result["integration_results"][integration_id] = (
                        lookup_results["data"][entity["value"]]
                    )
                if lookup_results.get("errors"):
                    results["errors"].extend(lookup_results["errors"])

            results["entities"].append(entity_result)

        return results

    async def polarity_entity_search(
        self,
        text: str,
        integration_ids: Optional[str] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
        """
        Search for entities in text using Polarity integrations.

        :param text: Text to parse for entities (IPs, domains, hashes, etc.)
        :param integration_ids: Optional comma-separated integration IDs to search, or 'ALL' for all. Uses configured defaults if not provided.
        :return: JSON string with entity enrichment data
        """
        if self._api_not_configured():
            return json.dumps(
                {
                    "error": self._build_error(
                        "not_configured",
                        "Polarity API not configured. Please configure API URL and Bearer Token in settings.",
                    )
                },
                indent=2,
            )

        await self._emit_status(
            __event_emitter__,
            "Parsing entities with Polarity...",
        )
        parsed_entities_result = await self._parse_entities(text)
        if not parsed_entities_result["ok"]:
            await self._emit_status(
                __event_emitter__,
                "Polarity entity parsing failed.",
                status="error",
                done=True,
            )
            return json.dumps({"error": parsed_entities_result["error"]}, indent=2)

        entities = parsed_entities_result["data"]
        if not entities:
            await self._emit_status(
                __event_emitter__,
                "No supported entities found.",
                status="complete",
                done=True,
            )
            return json.dumps(
                {
                    "message": "No standard entities (IP, domain, hash, etc.) found in text",
                    "supported_entity_types": [
                        "IPv4",
                        "IPv6",
                        "domain",
                        "email",
                        "url",
                        "MD5",
                        "SHA1",
                        "SHA256",
                        "IPv4CIDR",
                    ],
                },
                indent=2,
            )

        await self._emit_status(
            __event_emitter__,
            "Resolving available Polarity integrations...",
        )
        integrations_result = await self._get_integrations_to_search(integration_ids)
        if not integrations_result["ok"]:
            await self._emit_status(
                __event_emitter__,
                "Failed to resolve Polarity integrations.",
                status="error",
                done=True,
            )
            return json.dumps({"error": integrations_result["error"]}, indent=2)

        integration_data = integrations_result["data"]
        valid_integrations = integration_data["valid"]
        if not valid_integrations:
            await self._emit_status(
                __event_emitter__,
                "No valid Polarity integrations were available.",
                status="error",
                done=True,
            )
            return json.dumps(
                {
                    "error": self._build_error(
                        "no_valid_integrations",
                        "No valid integrations were available to search.",
                        details={
                            "requested_integrations": integration_ids,
                            "invalid_integrations": integration_data["invalid"],
                            "available_integrations": list(
                                integration_data["available"].keys()
                            ),
                        },
                    )
                },
                indent=2,
            )

        results = await self._build_search_results(
            entities[: self.valves.max_entities_per_request],
            valid_integrations,
            __event_emitter__=__event_emitter__,
        )
        if integration_data["invalid"]:
            results["errors"].append(
                self._build_error(
                    "invalid_integrations_ignored",
                    "Some requested integrations were ignored because they were unavailable or not running.",
                    details={"integration_ids": integration_data["invalid"]},
                )
            )

        await self._emit_status(
            __event_emitter__,
            f"Polarity search completed for {len(results['entities'])} entities.",
            status="complete",
            done=True,
        )
        return json.dumps(results, indent=2)

    async def on_inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> dict:
        """
        Auto-enrich user messages with entity data before sending to model.
        """
        if not self.valves.auto_enrich_enabled or self._api_not_configured():
            return body

        try:
            messages = body.get("messages", [])
            if not messages:
                return body

            last_user_message = None
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_user_message = msg.get("content", "")
                    break

            if not last_user_message:
                return body

            parsed_entities_result = await self._parse_entities(last_user_message)
            if not parsed_entities_result["ok"]:
                self._debug_log(
                    f"Auto-enrichment entity parsing failed: {parsed_entities_result['error']}"
                )
                return body

            entities = parsed_entities_result["data"]
            if not entities:
                return body

            auto_config = (
                self.valves.auto_enrich_integrations
                or self.valves.integrations_to_search
            )
            integrations_result = await self._get_integrations_to_search(auto_config)
            if not integrations_result["ok"]:
                self._debug_log(
                    f"Auto-enrichment integration lookup failed: {integrations_result['error']}"
                )
                return body

            valid_integrations = integrations_result["data"]["valid"]
            if not valid_integrations:
                return body

            enrichment_data = await self._build_search_results(
                entities[: self.valves.max_entities_per_request],
                valid_integrations,
                __event_emitter__=None,
            )

            if not enrichment_data["entities"]:
                return body

            enrichment_context = self._format_enrichment_context(enrichment_data)
            if not enrichment_context:
                return body

            system_msg_index = None
            for index, msg in enumerate(messages):
                if msg.get("role") == "system":
                    system_msg_index = index
                    break

            if system_msg_index is not None:
                messages[system_msg_index]["content"] += f"\n\n{enrichment_context}"
            else:
                messages.insert(0, {"role": "system", "content": enrichment_context})

            body["messages"] = messages
        except Exception as exc:
            self._debug_log(f"Auto-enrichment error: {str(exc)}")

        return body

    async def get_integration_list(
        self,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
        """
        Get list of available Polarity integrations.

        :return: JSON string with list of available integrations
        """
        await self._emit_status(
            __event_emitter__,
            "Fetching Polarity integrations...",
        )

        integrations_result = await self._get_integrations()
        if not integrations_result["ok"]:
            await self._emit_status(
                __event_emitter__,
                "Failed to fetch Polarity integrations.",
                status="error",
                done=True,
            )
            return json.dumps({"error": integrations_result["error"]}, indent=2)

        integrations = integrations_result["data"]
        result = {
            "integrations": [
                {
                    "id": integration_id,
                    "name": integration_data["name"],
                    "acronym": integration_data["acronym"],
                    "status": integration_data["status"],
                    "description": (
                        integration_data["description"][:100]
                        if integration_data.get("description")
                        else ""
                    ),
                }
                for integration_id, integration_data in integrations.items()
            ],
            "total": len(integrations),
            "configuration_hint": "Use integration IDs in 'integrations_to_search' valve or set to 'ALL'.",
        }

        await self._emit_status(
            __event_emitter__,
            f"Fetched {len(integrations)} Polarity integrations.",
            status="complete",
            done=True,
        )
        return json.dumps(result, indent=2)

    async def clear_cache(
        self,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
        """
        Clear the entity lookup cache.

        :return: Success message
        """
        cache_size = len(self._cache)
        self._cache.clear()
        self._integration_cache = None
        self._integration_cache_time = None

        await self._emit_status(
            __event_emitter__,
            f"Cleared Polarity cache ({cache_size} entries removed).",
            status="complete",
            done=True,
        )
        return f"Cache cleared. Removed {cache_size} cached entries."
