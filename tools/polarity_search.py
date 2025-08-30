"""
title: Polarity Entity Search and Auto-Enrichment
author: christian-taillon
author_url: https://christiant.io/
funding_url: https://github.com/open-webui/open-webui
github_url: https://github.com/open-webui/open-webui
version: 4.1.0
license: MIT
description: Searches entities through Polarity (polarity.io) integrations with automatic context enrichment for model responses.
requirements: requests
"""

import json
import time
import re
import hashlib
from typing import Optional, Dict, List, Any, Callable, Awaitable
from pydantic import BaseModel, Field
import requests
from datetime import datetime, timedelta


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
        self._cache = {}
        self._last_request_time = {}
        self._integration_cache = None
        self._integration_cache_time = None

    def _debug_log(self, message: str):
        """Log debug messages if debug mode is enabled."""
        if self.valves.debug_mode:
            print(f"[Polarity Debug] {message}")

    def _get_cache_key(self, entity_value: str, integration_id: str) -> str:
        """Generate cache key for entity-integration pair."""
        return hashlib.md5(f"{entity_value}:{integration_id}".encode()).hexdigest()

    def _is_cache_valid(self, cache_time: datetime) -> bool:
        """Check if cached data is still valid."""
        if not cache_time:
            return False
        return datetime.now() - cache_time < timedelta(
            minutes=self.valves.cache_duration_minutes
        )

    def _apply_rate_limit(self, integration_id: str):
        """Apply rate limiting per integration."""
        if integration_id in self._last_request_time:
            elapsed = time.time() - self._last_request_time[integration_id]
            if elapsed < self.valves.rate_limit_delay_seconds:
                time.sleep(self.valves.rate_limit_delay_seconds - elapsed)
        self._last_request_time[integration_id] = time.time()

    def _make_api_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Make authenticated API request to Polarity."""
        if not self.valves.polarity_api_url or not self.valves.polarity_bearer_token:
            self._debug_log("API URL or Bearer token not configured")
            return None

        url = f"{self.valves.polarity_api_url.rstrip('/')}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.valves.polarity_bearer_token}",
            "Content-Type": "application/vnd.api+json",
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
            return result
        except requests.exceptions.RequestException as e:
            self._debug_log(f"API request failed: {str(e)}")
            if hasattr(e, "response") and e.response is not None:
                self._debug_log(f"Response status: {e.response.status_code}")
                self._debug_log(f"Response body: {e.response.text[:500]}")
            return None

    def _get_integrations(self) -> Dict[str, Dict]:
        """Fetch and cache available integrations."""
        if self._integration_cache and self._is_cache_valid(
            self._integration_cache_time
        ):
            self._debug_log(
                f"Using cached integrations: {len(self._integration_cache)} available"
            )
            return self._integration_cache

        self._debug_log("Fetching integrations from API")
        result = self._make_api_request(
            "GET",
            "/api/integrations",
            params={"filter[integration.status]": "running", "page[size]": 200},
        )

        if result and "data" in result:
            self._integration_cache = {}
            for item in result["data"]:
                integration_id = item["id"]
                attributes = item.get("attributes", {})
                self._integration_cache[integration_id] = {
                    "id": integration_id,
                    "name": attributes.get("name", integration_id),
                    "acronym": attributes.get("acronym", ""),
                    "status": attributes.get("status", "unknown"),
                    "description": attributes.get("description", ""),
                }
            self._integration_cache_time = datetime.now()
            self._debug_log(
                f"Found {len(self._integration_cache)} running integrations"
            )
        else:
            self._integration_cache = {}
            self._debug_log("No integrations found or request failed")

        return self._integration_cache

    def _parse_entities(self, text: str) -> List[Dict]:
        """Parse text to extract entities."""
        self._debug_log(f"Parsing entities from text: {text[:100]}...")

        result = self._make_api_request(
            "POST",
            "/api/parsed-entities",
            data={"data": {"attributes": {"text": text}}},
        )

        if not result:
            self._debug_log("Failed to parse entities")
            return []

        entities = result.get("data", {}).get("attributes", {}).get("entities", [])
        annotations = (
            result.get("data", {}).get("attributes", {}).get("annotations", [])
        )

        self._debug_log(f"Found {len(entities)} entities")

        # Filter to standard entity types we want to lookup
        standard_types = [
            "IPv4",
            "IPv6",
            "domain",
            "email",
            "url",
            "MD5",
            "SHA1",
            "SHA256",
            "IPv4CIDR",
        ]
        filtered_entities = []

        for entity in entities:
            if entity.get("type") in standard_types:
                filtered_entities.append(entity)

        self._debug_log(f"Filtered to {len(filtered_entities)} standard entities")

        # Merge annotations into entities
        if self.valves.include_annotations and annotations:
            entity_annotations = {}
            for annotation in annotations:
                entity_name = annotation.get("entity", {}).get("entity-name", "")
                if entity_name:
                    tags = []
                    for context in annotation.get("contexts", []):
                        tag_name = context.get("tag", {}).get("tag-name", "")
                        if tag_name:
                            tags.append(tag_name)
                    if tags:
                        entity_annotations[entity_name] = tags

            # Add annotations to entities
            for entity in filtered_entities:
                entity_value = entity.get("value", "")
                if entity_value in entity_annotations:
                    entity["annotations"] = entity_annotations[entity_value]
                    self._debug_log(
                        f"Added annotations to {entity_value}: {entity_annotations[entity_value]}"
                    )

        return filtered_entities

    def _lookup_entities(
        self, entities: List[Dict], integration_id: str
    ) -> Dict[str, Any]:
        """Lookup entities in a specific integration."""
        if not entities:
            return {}

        self._debug_log(f"Looking up {len(entities)} entities in {integration_id}")

        # Check cache first
        results = {}
        entities_to_lookup = []

        for entity in entities:
            cache_key = self._get_cache_key(entity["value"], integration_id)
            if cache_key in self._cache and self._is_cache_valid(
                self._cache[cache_key]["time"]
            ):
                results[entity["value"]] = self._cache[cache_key]["data"]
                self._debug_log(f"Using cached result for {entity['value']}")
            else:
                entities_to_lookup.append(entity)

        # Lookup uncached entities
        if entities_to_lookup:
            self._debug_log(f"Looking up {len(entities_to_lookup)} uncached entities")

            # Apply rate limiting
            self._apply_rate_limit(integration_id)

            # Batch entities if needed
            for i in range(
                0, len(entities_to_lookup), self.valves.max_entities_per_request
            ):
                batch = entities_to_lookup[i : i + self.valves.max_entities_per_request]

                lookup_result = self._make_api_request(
                    "POST",
                    f"/api/integrations/{integration_id}/lookup",
                    data={
                        "data": {
                            "type": "integration-lookups",
                            "attributes": {"entities": batch},
                        }
                    },
                )

                if lookup_result and "data" in lookup_result:
                    lookup_results = (
                        lookup_result.get("data", {})
                        .get("attributes", {})
                        .get("results", [])
                    )
                    self._debug_log(
                        f"Got {len(lookup_results)} results from {integration_id}"
                    )

                    for i, result in enumerate(lookup_results):
                        if i < len(batch):
                            entity_value = batch[i]["value"]

                            # Process result
                            if result.get("data"):
                                processed_result = {
                                    "summary": result["data"].get("summary", []),
                                    "details": result["data"].get("details", {}),
                                    "entity": result.get("entity", batch[i]),
                                }
                                results[entity_value] = processed_result
                                self._debug_log(
                                    f"Found data for {entity_value} in {integration_id}"
                                )

                                # Cache result
                                cache_key = self._get_cache_key(
                                    entity_value, integration_id
                                )
                                self._cache[cache_key] = {
                                    "data": processed_result,
                                    "time": datetime.now(),
                                }
                            else:
                                self._debug_log(
                                    f"No data found for {entity_value} in {integration_id}"
                                )
                else:
                    self._debug_log(f"Lookup request failed for {integration_id}")

        return results

    def _get_integrations_to_search(
        self, integration_ids: Optional[str] = None
    ) -> List[str]:
        """Determine which integrations to search based on configuration."""
        # Get available integrations first
        available_integrations = self._get_integrations()

        # Determine which integrations to use
        if integration_ids:
            config_value = integration_ids.strip()
        else:
            config_value = self.valves.integrations_to_search.strip()

        if config_value.upper() == "ALL":
            # Use all available running integrations
            integrations_to_search = list(available_integrations.keys())
            self._debug_log(f"Using ALL integrations: {integrations_to_search}")
        elif config_value:
            # Use specified integrations
            integrations_to_search = [i.strip() for i in config_value.split(",")]
            self._debug_log(f"Using specified integrations: {integrations_to_search}")
        else:
            # No integrations configured
            integrations_to_search = []
            self._debug_log("No integrations configured")

        # Filter to only running integrations
        valid_integrations = [
            i
            for i in integrations_to_search
            if i in available_integrations
            and available_integrations[i]["status"] == "running"
        ]

        self._debug_log(f"Valid running integrations: {valid_integrations}")

        return valid_integrations

    def _format_enrichment_context(self, enrichment_data: Dict) -> str:
        """Format enrichment data for model context."""
        if not enrichment_data or not enrichment_data.get("entities"):
            return ""

        context_lines = ["[Entity Enrichment Data]"]

        for entity_info in enrichment_data["entities"]:
            value = entity_info["value"]
            entity_type = entity_info["type"]

            context_lines.append(f"\n• {value} ({entity_type}):")

            # Add annotations
            if entity_info.get("annotations"):
                context_lines.append(f"  Tags: {', '.join(entity_info['annotations'])}")

            # Add integration results
            for integration_id, data in entity_info.get(
                "integration_results", {}
            ).items():
                if data and data.get("summary"):
                    context_lines.append(
                        f"  {integration_id}: {' | '.join(data['summary'])}"
                    )

        return "\n".join(context_lines)

    def polarity_entity_search(
        self, text: str, integration_ids: Optional[str] = None
    ) -> str:
        """
        Search for entities in text using Polarity integrations.

        :param text: Text to parse for entities (IPs, domains, hashes, etc.)
        :param integration_ids: Optional comma-separated integration IDs to search, or 'ALL' for all. Uses configured defaults if not provided.
        :return: JSON string with entity enrichment data
        """
        if not self.valves.polarity_api_url or not self.valves.polarity_bearer_token:
            return json.dumps(
                {
                    "error": "Polarity API not configured. Please configure API URL and Bearer Token in settings."
                }
            )

        # Parse entities from text
        entities = self._parse_entities(text)
        if not entities:
            return json.dumps(
                {
                    "message": "No standard entities (IP, domain, hash, etc.) found in text"
                }
            )

        # Get integrations to search
        valid_integrations = self._get_integrations_to_search(integration_ids)

        if not valid_integrations:
            available = self._get_integrations()
            return json.dumps(
                {
                    "error": "No valid integrations to search",
                    "available_integrations": list(available.keys()),
                    "hint": "Set integrations_to_search to 'ALL' or specify integration IDs",
                }
            )

        # Lookup entities in each integration
        results = {
            "entities": [],
            "integrations_searched": valid_integrations,
            "timestamp": datetime.now().isoformat(),
        }

        for entity in entities:
            entity_result = {
                "value": entity["value"],
                "type": entity["type"],
                "annotations": entity.get("annotations", []),
                "integration_results": {},
            }

            for integration_id in valid_integrations:
                lookup_results = self._lookup_entities([entity], integration_id)
                if entity["value"] in lookup_results:
                    entity_result["integration_results"][integration_id] = (
                        lookup_results[entity["value"]]
                    )

            results["entities"].append(entity_result)

        return json.dumps(results, indent=2)

    async def on_inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> dict:
        """
        Auto-enrich user messages with entity data before sending to model.
        """
        if not self.valves.auto_enrich_enabled:
            return body

        try:
            # Extract text from the last user message
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

            # Parse entities from user message
            entities = self._parse_entities(last_user_message)
            if not entities:
                return body

            # Get integrations for auto-enrichment
            auto_config = (
                self.valves.auto_enrich_integrations
                or self.valves.integrations_to_search
            )
            valid_integrations = self._get_integrations_to_search(auto_config)

            if not valid_integrations:
                return body

            # Lookup entities
            enrichment_data = {"entities": [], "auto_enriched": True}

            for entity in entities[
                : self.valves.max_entities_per_request
            ]:  # Limit entities
                entity_result = {
                    "value": entity["value"],
                    "type": entity["type"],
                    "annotations": entity.get("annotations", []),
                    "integration_results": {},
                }

                for integration_id in valid_integrations:
                    lookup_results = self._lookup_entities([entity], integration_id)
                    if entity["value"] in lookup_results:
                        entity_result["integration_results"][integration_id] = (
                            lookup_results[entity["value"]]
                        )

                enrichment_data["entities"].append(entity_result)

            # Add enrichment context to system message
            if enrichment_data["entities"]:
                enrichment_context = self._format_enrichment_context(enrichment_data)

                # Find or create system message
                system_msg_index = None
                for i, msg in enumerate(messages):
                    if msg.get("role") == "system":
                        system_msg_index = i
                        break

                if system_msg_index is not None:
                    messages[system_msg_index]["content"] += f"\n\n{enrichment_context}"
                else:
                    messages.insert(
                        0, {"role": "system", "content": enrichment_context}
                    )

                body["messages"] = messages

        except Exception as e:
            self._debug_log(f"Auto-enrichment error: {str(e)}")

        return body

    def get_integration_list(self) -> str:
        """
        Get list of available Polarity integrations.

        :return: JSON string with list of available integrations
        """
        integrations = self._get_integrations()

        if not integrations:
            return json.dumps(
                {
                    "error": "Could not fetch integrations. Check API configuration.",
                    "integrations": [],
                    "total": 0,
                }
            )

        result = {
            "integrations": [
                {
                    "id": int_id,
                    "name": int_data["name"],
                    "acronym": int_data["acronym"],
                    "status": int_data["status"],
                    "description": (
                        int_data["description"][:100]
                        if int_data.get("description")
                        else ""
                    ),
                }
                for int_id, int_data in integrations.items()
            ],
            "total": len(integrations),
            "configuration_hint": "Use integration IDs in 'integrations_to_search' valve or set to 'ALL'",
        }

        return json.dumps(result, indent=2)

    def clear_cache(self) -> str:
        """
        Clear the entity lookup cache.

        :return: Success message
        """
        cache_size = len(self._cache)
        self._cache.clear()
        self._integration_cache = None
        self._integration_cache_time = None

        return f"Cache cleared. Removed {cache_size} cached entries."

