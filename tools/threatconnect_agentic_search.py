"""
title: ThreatConnect Agentic Tool
author: christian-taillon
author_url: https://christiant.io/
funding_url: https://github.com/open-webui/open-webui
git_url: https://github.com/christian-taillon/open-webui-pipelines
version: 5.2.0
required_open_webui_version: 0.4.0
license: MIT
description: Async ThreatConnect search tool for indicators and groups, including group-to-indicator traversal and lightweight enrichment.
requirements: httpx, tenacity
"""

import asyncio
import base64
import csv
import hashlib
import hmac
import io
import json
import re
import time
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class APIError(Exception):
    pass


class RateLimitError(APIError):
    pass


class NotFoundError(APIError):
    pass


class AuthMethod(str, Enum):
    token = "token"
    access_key = "access_key"


class Tools:
    class Valves(BaseModel):
        tc_api_url: str = Field(
            default="https://app.threatconnect.com",
            description="ThreatConnect base URL, for example https://app.threatconnect.com.",
        )
        api_version: str = Field(
            default="v3",
            description="ThreatConnect API version. This tool is optimized for v3.",
        )
        owner: str = Field(
            default="",
            description="Optional ThreatConnect owner name to scope searches.",
        )

        auth_method: AuthMethod = Field(
            default=AuthMethod.token,
            description="Authentication method: token or access_key.",
        )
        tc_api_token: str = Field(
            default="",
            description="ThreatConnect API token for token auth.",
            sensitive=True,
        )
        tc_api_access_id: str = Field(
            default="",
            description="ThreatConnect API access ID for access_key auth.",
            sensitive=True,
        )
        tc_api_secret_key: str = Field(
            default="",
            description="ThreatConnect API secret key for access_key auth.",
            sensitive=True,
        )

        timeout_seconds: int = Field(
            default=30,
            description="HTTP request timeout in seconds.",
        )
        max_results_per_ioc: int = Field(
            default=10,
            description="Maximum indicator matches to return per IOC lookup.",
        )
        max_search_results: int = Field(
            default=25,
            description="Maximum objects returned by indicator or group searches.",
        )
        max_group_assoc_limit: int = Field(
            default=200,
            description="Page size for group association retrieval. ThreatConnect supports up to 500.",
        )
        max_group_assoc_pages: int = Field(
            default=10,
            description="Maximum pages to retrieve for group associations.",
        )

        include_all_fields: bool = Field(
            default=True,
            description="Include all available fields discovered from the endpoint when possible.",
        )
        custom_fields: str = Field(
            default="",
            description="Comma-separated fields to request when include_all_fields is false.",
        )

        enrich_groups_with_attributes: bool = Field(
            default=True,
            description="Expand associated group details for indicator results.",
        )
        fields_to_remove_from_indicator: str = Field(
            default="investigationLinks,ownerId,trackedUsers",
            description="Comma-separated indicator fields to strip from enriched responses.",
        )

        traverse_associated_groups: bool = Field(
            default=True,
            description="Traverse associated groups when collecting IOCs from groups.",
        )
        max_traversal_depth: int = Field(
            default=2,
            description="Maximum associated-group traversal depth.",
        )
        compact_output_enabled: bool = Field(
            default=True,
            description="Return a compact, model-friendly JSON view instead of the full raw ThreatConnect payload.",
        )
        max_compact_items: int = Field(
            default=5,
            description="Maximum items to keep for arrays such as attributes, tags, labels, or associations in compact output.",
        )
        max_text_field_length: int = Field(
            default=280,
            description="Maximum character length for long text fields in compact output.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.ioc_patterns = {
            "ip": r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
            "domain": r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b",
            "hash": r"\b[a-fA-F0-9]{32}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{64}\b",
            "url": r"https?://[^\s<>\"{}|\\^`\[\]]+",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "cve": r"\bCVE-\d{4}-\d{4,}\b",
        }
        self._available_fields_cache: Dict[str, List[str]] = {}

    async def _emit_status(
        self,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]],
        description: str,
        *,
        done: bool = False,
        status: str = "in_progress",
    ) -> None:
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

    def _validate_configuration(self) -> Optional[str]:
        if self.valves.auth_method == AuthMethod.token and not self.valves.tc_api_token:
            return "ThreatConnect token auth selected, but tc_api_token is not configured."
        if self.valves.auth_method == AuthMethod.access_key and (
            not self.valves.tc_api_access_id or not self.valves.tc_api_secret_key
        ):
            return "ThreatConnect access_key auth selected, but tc_api_access_id or tc_api_secret_key is not configured."
        return None

    def _build_signature_path(
        self, path: str, params: Optional[List[Tuple[str, str]]] = None
    ) -> str:
        if not params:
            return path
        return f"{path}?{urlencode(params, doseq=True)}"

    def get_headers(
        self,
        path: str,
        method: str = "GET",
        params: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, str]:
        if self.valves.auth_method == AuthMethod.token:
            return {"Authorization": f"TC-Token {self.valves.tc_api_token}"}

        timestamp = str(int(time.time()))
        signature_path = self._build_signature_path(path, params)
        signature = f"{signature_path}:{method}:{timestamp}"
        hmac_signature = base64.b64encode(
            hmac.new(
                self.valves.tc_api_secret_key.encode("utf-8"),
                signature.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")
        return {
            "Authorization": f"TC {self.valves.tc_api_access_id}:{hmac_signature}",
            "Timestamp": timestamp,
        }

    def _prepare_params(
        self, params: Optional[Dict[str, Any]]
    ) -> Optional[List[Tuple[str, str]]]:
        if not params:
            params = {}

        prepared: List[Tuple[str, str]] = []
        for key, value in params.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    if item is not None:
                        prepared.append((key, str(item)))
            else:
                prepared.append((key, str(value)))

        if self.valves.owner and not any(key == "owner" for key, _ in prepared):
            prepared.append(("owner", self.valves.owner))

        return prepared or None

    def _coerce_object(self, response_data: Any) -> Dict[str, Any]:
        if isinstance(response_data, list):
            return response_data[0] if response_data else {}
        if isinstance(response_data, dict):
            return response_data
        return {}

    def _split_csv(self, value: str) -> List[str]:
        return [item.strip() for item in value.split(",") if item.strip()]

    def _truncate_text(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        max_len = self.valves.max_text_field_length
        if len(text) <= max_len:
            return text
        return f"{text[:max_len].rstrip()}..."

    def _limit_items(self, values: List[Any]) -> List[Any]:
        return values[: max(self.valves.max_compact_items, 1)]

    def _summarize_named_items(
        self, container: Any, *, preferred_keys: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        preferred_keys = preferred_keys or ["name", "summary", "type", "id"]
        items = []
        data = container.get("data", []) if isinstance(container, dict) else container
        if not isinstance(data, list):
            return items

        for item in self._limit_items(data):
            if not isinstance(item, dict):
                continue
            summarized: Dict[str, Any] = {}
            for key in preferred_keys:
                if item.get(key) is not None:
                    value = item.get(key)
                    summarized[key] = (
                        self._truncate_text(value)
                        if isinstance(value, str)
                        else value
                    )
            if summarized:
                items.append(summarized)
        return items

    def _summarize_attributes(self, attributes: Any) -> List[Dict[str, Any]]:
        items = []
        data = attributes.get("data", []) if isinstance(attributes, dict) else attributes
        if not isinstance(data, list):
            return items

        for item in self._limit_items(data):
            if not isinstance(item, dict):
                continue
            summary: Dict[str, Any] = {}
            for key in ["id", "type", "default", "pinned", "dateAdded", "lastModified"]:
                if item.get(key) is not None:
                    summary[key] = item.get(key)
            if item.get("value") is not None:
                summary["value"] = self._truncate_text(item.get("value"))
            if summary:
                items.append(summary)
        return items

    def _compact_group_summary(self, group: Dict[str, Any]) -> Dict[str, Any]:
        compact: Dict[str, Any] = {}
        for key in [
            "id",
            "name",
            "summary",
            "type",
            "ownerName",
            "dateAdded",
            "lastModified",
            "status",
            "publishDate",
            "eventDate",
        ]:
            if group.get(key) is not None:
                compact[key] = group.get(key)

        if group.get("tags"):
            compact["tags"] = self._summarize_named_items(group["tags"], preferred_keys=["name", "id"])
        if group.get("securityLabels"):
            compact["securityLabels"] = self._summarize_named_items(
                group["securityLabels"], preferred_keys=["name", "id"]
            )
        if group.get("attributes"):
            compact["attributes"] = self._summarize_attributes(group["attributes"])

        return compact

    def _compact_indicator_summary(self, indicator: Dict[str, Any]) -> Dict[str, Any]:
        compact: Dict[str, Any] = {}
        for key in [
            "id",
            "summary",
            "type",
            "ownerName",
            "rating",
            "confidence",
            "active",
            "privateFlag",
            "falsePositives",
            "observations",
            "dateAdded",
            "lastModified",
            "firstSeen",
            "lastSeen",
            "ip",
            "hostName",
            "address",
            "text",
            "md5",
            "sha1",
            "sha256",
        ]:
            if indicator.get(key) is not None:
                compact[key] = indicator.get(key)

        if indicator.get("tags"):
            compact["tags"] = self._summarize_named_items(
                indicator["tags"], preferred_keys=["name", "id", "techniqueId"]
            )
        if indicator.get("securityLabels"):
            compact["securityLabels"] = self._summarize_named_items(
                indicator["securityLabels"], preferred_keys=["name", "id"]
            )
        if indicator.get("attributes"):
            compact["attributes"] = self._summarize_attributes(indicator["attributes"])
        if indicator.get("associatedGroups"):
            compact["associatedGroups"] = self._summarize_named_items(
                indicator["associatedGroups"],
                preferred_keys=["id", "name", "type", "ownerName"],
            )
        return compact

    def _compact_response_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.valves.compact_output_enabled:
            return payload

        compact: Dict[str, Any] = {}
        for key in ["status", "message", "error", "context", "query_time", "total_indicators"]:
            if payload.get(key) is not None:
                compact[key] = payload.get(key)
        if payload.get("details") is not None:
            compact["details"] = payload.get("details")
        if payload.get("_metadata") is not None:
            compact["_metadata"] = payload.get("_metadata")

        data = payload.get("data")
        if isinstance(data, list):
            compact["count"] = len(data)
            compact["data"] = [
                self._compact_indicator_summary(item)
                if item.get("summary") is not None or item.get("type") in {"Address", "Host", "URL", "EmailAddress", "File"}
                else self._compact_group_summary(item)
                for item in self._limit_items(data)
                if isinstance(item, dict)
            ]
        elif isinstance(data, dict):
            compact["data"] = (
                self._compact_indicator_summary(data)
                if data.get("summary") is not None or data.get("type") in {"Address", "Host", "URL", "EmailAddress", "File"}
                else self._compact_group_summary(data)
            )

        results = payload.get("results")
        if isinstance(results, list):
            compact["results"] = []
            for item in self._limit_items(results):
                if not isinstance(item, dict):
                    continue
                compact_result = self._compact_response_payload(item)
                compact["results"].append(compact_result)

        rows = payload.get("rows")
        if rows is not None:
            compact["rows"] = rows

        return compact

    def _default_fields_for(self, endpoint_type: str) -> List[str]:
        if endpoint_type == "groups":
            return [
                "associatedIndicators",
                "associatedGroups",
                "attributes",
                "tags",
                "securityLabels",
            ]
        return [
            "associatedGroups",
            "associatedIndicators",
            "attributes",
            "tags",
            "securityLabels",
            "observations",
            "falsePositives",
            "customAssociations",
            "fileActions",
            "fileOccurrences",
        ]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RateLimitError),
    )
    async def _api_request_async(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        config_error = self._validate_configuration()
        if config_error:
            raise APIError(config_error)

        url = f"{self.valves.tc_api_url.rstrip('/')}{endpoint}"
        prepared_params = self._prepare_params(params)
        headers = self.get_headers(endpoint, method.upper(), prepared_params)

        async with httpx.AsyncClient(timeout=self.valves.timeout_seconds) as client:
            try:
                response = await client.request(
                    method=method.upper(),
                    url=url,
                    headers=headers,
                    params=prepared_params,
                )
                if response.status_code == 429:
                    raise RateLimitError(f"Rate limit exceeded for {url}")
                if response.status_code == 404:
                    raise NotFoundError(f"Resource not found: {url}")
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                raise APIError(
                    f"API request failed for {exc.request.url} with status {exc.response.status_code}: {exc.response.text}"
                ) from exc
            except httpx.RequestError as exc:
                raise APIError(f"Network error for {url}: {exc}") from exc

    def escape_tql_string(self, value: Any) -> str:
        sanitized = (
            str(value)
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("'", "\\'")
            .replace("`", "\\`")
        )
        return f'"{sanitized}"'

    def _build_tql_from_struct(self, conditions: List[Dict[str, Any]]) -> str:
        if not conditions:
            return ""
        return " AND ".join(
            f'{condition["field"]} {condition["operator"]} {self.escape_tql_string(condition["value"])}'
            for condition in conditions
        )

    async def build_fields_params_async(
        self, endpoint_type: str = "indicators"
    ) -> Dict[str, List[str]]:
        if self.valves.include_all_fields:
            if endpoint_type not in self._available_fields_cache:
                try:
                    data = await self._api_request_async(
                        "OPTIONS",
                        f"/api/{self.valves.api_version}/{endpoint_type}/fields",
                    )
                    self._available_fields_cache[endpoint_type] = [
                        field["name"]
                        for field in data.get("data", [])
                        if isinstance(field, dict) and field.get("name")
                    ]
                except APIError:
                    self._available_fields_cache[endpoint_type] = (
                        self._default_fields_for(endpoint_type)
                    )
            fields = self._available_fields_cache[endpoint_type]
        elif self.valves.custom_fields:
            fields = self._split_csv(self.valves.custom_fields)
        else:
            fields = self._default_fields_for(endpoint_type)

        return {"fields": fields} if fields else {}

    def detect_ioc_type(self, indicator: str) -> str:
        indicator = indicator.strip()
        for ioc_type, pattern in self.ioc_patterns.items():
            if re.fullmatch(pattern, indicator):
                return {
                    "ip": "Address",
                    "url": "URL",
                    "email": "EmailAddress",
                    "hash": "File",
                    "domain": "Host",
                    "cve": "Vulnerability",
                }.get(ioc_type, "Unknown")
        return "Unknown"

    async def query_threatconnect_api_async(self, indicator: str) -> Dict[str, Any]:
        try:
            params = {
                "resultLimit": self.valves.max_results_per_ioc,
                "tql": f'summary = {self.escape_tql_string(indicator)}',
            }
            params.update(await self.build_fields_params_async("indicators"))
            return await self._api_request_async(
                "GET",
                f"/api/{self.valves.api_version}/indicators",
                params=params,
            )
        except NotFoundError:
            return {"status": "not_found", "message": "Indicator not found"}
        except APIError as exc:
            return {"error": str(exc)}

    async def _get_group_associated_indicators_async(
        self, group_id: int
    ) -> List[Dict[str, Any]]:
        indicators: List[Dict[str, Any]] = []
        limit = min(self.valves.max_group_assoc_limit, 500)

        for page in range(self.valves.max_group_assoc_pages):
            start = page * limit
            response = await self._api_request_async(
                "GET",
                f"/api/{self.valves.api_version}/groups/{group_id}",
                params={
                    "fields": ["associatedIndicators"],
                    "associatedIndicators.resultStart": start,
                    "associatedIndicators.resultLimit": limit,
                },
            )
            data = self._coerce_object(response.get("data"))
            chunk = data.get("associatedIndicators", {}).get("data", [])
            if not chunk:
                break
            indicators.extend(chunk)
            if len(chunk) < limit:
                break

        return indicators

    async def _get_group_associated_groups_async(
        self, group_id: int
    ) -> List[Dict[str, Any]]:
        groups: List[Dict[str, Any]] = []
        limit = min(self.valves.max_group_assoc_limit, 500)

        for page in range(self.valves.max_group_assoc_pages):
            start = page * limit
            response = await self._api_request_async(
                "GET",
                f"/api/{self.valves.api_version}/groups/{group_id}",
                params={
                    "fields": ["associatedGroups"],
                    "associatedGroups.resultStart": start,
                    "associatedGroups.resultLimit": limit,
                },
            )
            data = self._coerce_object(response.get("data"))
            chunk = data.get("associatedGroups", {}).get("data", [])
            if not chunk:
                break
            groups.extend(chunk)
            if len(chunk) < limit:
                break

        return groups

    async def _get_group_details_async(self, group_id: int) -> Dict[str, Any]:
        try:
            response = await self._api_request_async(
                "GET",
                f"/api/{self.valves.api_version}/groups/{group_id}",
                params={
                    "fields": [
                        "attributes",
                        "tags",
                        "securityLabels",
                        "associatedIndicators",
                        "associatedGroups",
                    ]
                },
            )
            data = self._coerce_object(response.get("data"))
            associated_indicators = await self._get_group_associated_indicators_async(
                group_id
            )
            associated_groups = await self._get_group_associated_groups_async(group_id)
            return {
                "id": data.get("id", group_id),
                "name": data.get("name"),
                "type": data.get("type"),
                "ownerName": data.get("ownerName"),
                "attributes": data.get("attributes", {}),
                "tags": data.get("tags", {}),
                "securityLabels": data.get("securityLabels", {}),
                "associatedIndicators": {"data": associated_indicators},
                "associatedGroups": {"data": associated_groups},
            }
        except APIError as exc:
            return {
                "id": group_id,
                "error": str(exc),
                "associatedIndicators": {"data": []},
                "associatedGroups": {"data": []},
            }

    async def _enrich_results_async(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not self.valves.enrich_groups_with_attributes:
            return results

        group_ids_to_fetch: Set[int] = set()
        for result in results:
            if result.get("status") != "Success":
                continue
            data_items = result.get("data", [])
            if isinstance(data_items, dict):
                data_items = [data_items]
            for item in data_items:
                for group in item.get("associatedGroups", {}).get("data", []):
                    group_id = group.get("id")
                    if group_id:
                        group_ids_to_fetch.add(group_id)

        if not group_ids_to_fetch:
            return results

        group_ids = sorted(group_ids_to_fetch)
        group_details = await asyncio.gather(
            *(self._get_group_details_async(group_id) for group_id in group_ids)
        )
        group_cache = {group_id: details for group_id, details in zip(group_ids, group_details)}

        fields_to_remove = self._split_csv(self.valves.fields_to_remove_from_indicator)
        for result in results:
            if result.get("status") != "Success":
                continue
            data_items = result.get("data", [])
            if isinstance(data_items, dict):
                data_items = [data_items]
            for item in data_items:
                for field in fields_to_remove:
                    item.pop(field, None)
                for group in item.get("associatedGroups", {}).get("data", []):
                    group_id = group.get("id")
                    details = group_cache.get(group_id)
                    if not details:
                        continue
                    group["details"] = {
                        "name": details.get("name"),
                        "type": details.get("type"),
                        "ownerName": details.get("ownerName"),
                        "attributes": details.get("attributes", {}),
                        "tags": details.get("tags", {}),
                        "securityLabels": details.get("securityLabels", {}),
                    }

        return results

    async def _perform_search_async(
        self,
        endpoint_type: str,
        conditions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        try:
            tql_query = self._build_tql_from_struct(conditions)
            if not tql_query:
                raise ValueError("Search requires at least one condition.")

            params = {"resultLimit": self.valves.max_search_results, "tql": tql_query}
            params.update(await self.build_fields_params_async(endpoint_type))

            response = await self._api_request_async(
                "GET",
                f"/api/{self.valves.api_version}/{endpoint_type}",
                params=params,
            )
            if (
                endpoint_type == "indicators"
                and response.get("status") == "Success"
                and self.valves.enrich_groups_with_attributes
            ):
                return (await self._enrich_results_async([response]))[0]
            return response
        except (ValueError, TypeError, APIError) as exc:
            return {"error": str(exc)}

    def _normalize_group_type_filters(self, group_types: Optional[str]) -> List[str]:
        if not group_types:
            return []

        alias_map = {
            "threat actor": "Threat",
            "threatactor": "Threat",
            "intrusionset": "Intrusion Set",
            "attackpattern": "Attack Pattern",
            "courseofaction": "Course of Action",
        }

        normalized: List[str] = []
        for raw_value in self._split_csv(group_types):
            key = raw_value.lower().strip()
            normalized.append(alias_map.get(key, raw_value))
        return normalized

    def _format_multi_search_errors(
        self, errors: List[Dict[str, str]], context: str
    ) -> Dict[str, Any]:
        if len(errors) == 1:
            return {"error": errors[0]["error"], "context": context}
        return {
            "error": f"{context} failed for one or more group types.",
            "details": errors,
        }

    async def get_iocs_for_group_ids(
        self,
        group_ids: str,
        format: str = "json",
        traverse: Optional[bool] = None,
        max_depth: Optional[int] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
        """
        Fetch indicators associated with one or more ThreatConnect groups.

        :param group_ids: Comma, whitespace, or newline separated group IDs.
        :param format: json or csv.
        :param traverse: Whether to walk associated groups.
        :param max_depth: Maximum associated-group traversal depth.
        """
        traverse = (
            self.valves.traverse_associated_groups if traverse is None else traverse
        )
        max_depth = self.valves.max_traversal_depth if max_depth is None else max_depth

        raw_ids = [token for token in re.split(r"[,\s\n]+", group_ids) if token.strip()]
        if not raw_ids:
            return json.dumps({"error": "No group IDs supplied."}, indent=2)

        try:
            ids = [int(token) for token in raw_ids]
        except ValueError:
            return json.dumps(
                {"error": "group_ids must contain only integer ThreatConnect group IDs."},
                indent=2,
            )

        await self._emit_status(
            __event_emitter__, f"Collecting indicators for {len(ids)} group(s)..."
        )

        seen_groups: Set[int] = set()
        seen_indicator_keys: Set[str] = set()
        queue: List[Tuple[int, int]] = [(group_id, 0) for group_id in ids]
        rows: List[Dict[str, Any]] = []

        while queue:
            group_id, depth = queue.pop(0)
            if group_id in seen_groups:
                continue
            seen_groups.add(group_id)

            details = await self._get_group_details_async(group_id)
            group_name = details.get("name") or ""
            group_type = details.get("type") or ""

            for indicator in details.get("associatedIndicators", {}).get("data", []):
                indicator_key = str(
                    indicator.get("id")
                    or f"{indicator.get('type')}|{indicator.get('summary')}"
                )
                if indicator_key in seen_indicator_keys:
                    continue
                seen_indicator_keys.add(indicator_key)
                rows.append(
                    {
                        "groupId": group_id,
                        "groupName": group_name,
                        "groupType": group_type,
                        "indicatorId": indicator.get("id"),
                        "indicatorType": indicator.get("type"),
                        "indicator": indicator.get("summary"),
                        "rating": indicator.get("rating"),
                        "confidence": indicator.get("confidence"),
                        "dateAdded": indicator.get("dateAdded"),
                        "lastModified": indicator.get("lastModified"),
                        "ownerName": indicator.get("ownerName"),
                    }
                )

            if traverse and depth < max_depth:
                for associated_group in details.get("associatedGroups", {}).get("data", []):
                    next_group_id = associated_group.get("id")
                    if next_group_id and next_group_id not in seen_groups:
                        queue.append((next_group_id, depth + 1))

        await self._emit_status(
            __event_emitter__,
            f"Collected {len(rows)} indicator association(s) from {len(seen_groups)} group(s).",
            done=True,
        )

        if format.lower() == "csv":
            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    "groupId",
                    "groupName",
                    "groupType",
                    "indicatorId",
                    "indicatorType",
                    "indicator",
                    "rating",
                    "confidence",
                    "dateAdded",
                    "lastModified",
                    "ownerName",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
            return output.getvalue()

        return json.dumps({"rows": rows}, indent=2)

    async def get_iocs_for_keyword(
        self,
        keywords: str,
        format: str = "json",
        group_types: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        traverse: Optional[bool] = None,
        max_depth: Optional[int] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
        """
        Search ThreatConnect groups by keyword and return associated indicators.

        :param keywords: Group keyword search term, matched against the group summary via TQL CONTAINS.
        :param format: json or csv.
        :param group_types: Optional comma-separated group types such as Report, Campaign, Incident, Malware, Adversary, Threat.
        :param start_date: Optional lower dateAdded bound.
        :param end_date: Optional upper dateAdded bound.
        """
        if not keywords.strip():
            return json.dumps({"error": "keywords is required."}, indent=2)

        await self._emit_status(
            __event_emitter__, f"Searching groups for keyword: {keywords}"
        )

        base_conditions = [
            {"field": "summary", "operator": "CONTAINS", "value": keywords}
        ]
        if start_date:
            base_conditions.append(
                {"field": "dateAdded", "operator": ">=", "value": start_date}
            )
        if end_date:
            base_conditions.append(
                {"field": "dateAdded", "operator": "<=", "value": end_date}
            )

        search_types = self._normalize_group_type_filters(group_types)
        if not search_types:
            search_types = [
                "Report",
                "Campaign",
                "Incident",
                "Malware",
                "Adversary",
                "Threat",
                "Intrusion Set",
                "Vulnerability",
            ]

        groups: List[Dict[str, Any]] = []
        seen_group_ids: Set[int] = set()
        search_errors: List[Dict[str, str]] = []
        for group_type in search_types:
            response = await self._perform_search_async(
                "groups",
                base_conditions
                + [{"field": "typeName", "operator": "=", "value": group_type}],
            )
            if response.get("error"):
                search_errors.append(
                    {"group_type": group_type, "error": str(response["error"])}
                )
                continue
            for group in response.get("data", []):
                group_id = group.get("id")
                if group_id and group_id not in seen_group_ids:
                    seen_group_ids.add(group_id)
                    groups.append(group)

        if search_errors and not groups:
            await self._emit_status(
                __event_emitter__,
                "ThreatConnect group search failed.",
                done=True,
                status="error",
            )
            return json.dumps(
                self._format_multi_search_errors(
                    search_errors, "ThreatConnect group search"
                ),
                indent=2,
            )

        if not groups:
            await self._emit_status(
                __event_emitter__,
                "No groups matched the requested keyword search.",
                done=True,
            )
            return json.dumps({"rows": []}, indent=2)

        return await self.get_iocs_for_group_ids(
            group_ids=",".join(str(group["id"]) for group in groups if group.get("id")),
            format=format,
            traverse=traverse,
            max_depth=max_depth,
            __event_emitter__=__event_emitter__,
        )

    async def get_group_iocs(
        self,
        group_ids: str,
        as_csv: bool = False,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
        """
        Backward-compatible wrapper around get_iocs_for_group_ids.
        """
        return await self.get_iocs_for_group_ids(
            group_ids=group_ids,
            format="csv" if as_csv else "json",
            traverse=self.valves.traverse_associated_groups,
            max_depth=self.valves.max_traversal_depth,
            __event_emitter__=__event_emitter__,
        )

    async def get_group_iocs_by_keyword(
        self,
        keywords: str,
        group_type: str = "",
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
        """
        Backward-compatible wrapper around get_iocs_for_keyword.
        """
        return await self.get_iocs_for_keyword(
            keywords=keywords,
            format="csv",
            group_types=group_type or None,
            traverse=self.valves.traverse_associated_groups,
            max_depth=self.valves.max_traversal_depth,
            __event_emitter__=__event_emitter__,
        )

    async def lookup_ioc(
        self,
        indicator: str,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
        """
        Look up a single IOC in ThreatConnect and return the matching indicator payload.
        """
        indicator = indicator.strip()
        if not indicator:
            return json.dumps({"error": "indicator is required."}, indent=2)

        await self._emit_status(__event_emitter__, f"Looking up {indicator}...")

        ioc_type = self.detect_ioc_type(indicator)
        if ioc_type == "Unknown":
            return json.dumps(
                {
                    "error": f"Could not determine indicator type: {indicator}",
                    "hint": "Use lookup_entity() for ThreatConnect objects that may be Groups, such as APT names or CVE identifiers.",
                },
                indent=2,
            )
        if ioc_type == "Vulnerability":
            await self._emit_status(
                __event_emitter__,
                f"{indicator} detected as a Vulnerability group. Searching groups instead of indicators...",
            )
            return await self.search_groups(
                keywords=indicator,
                group_types="Vulnerability",
                __user__=__user__,
                __event_emitter__=__event_emitter__,
            )

        api_response = await self.query_threatconnect_api_async(indicator)
        enriched_response = (await self._enrich_results_async([api_response]))[0]
        enriched_response["_metadata"] = {
            "indicator": indicator,
            "ioc_type": ioc_type,
            "query_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }

        await self._emit_status(
            __event_emitter__, f"Lookup complete for {indicator}.", done=True
        )
        return json.dumps(self._compact_response_payload(enriched_response), indent=2)

    async def lookup_entity(
        self,
        query: str,
        group_types: Optional[str] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
        """
        Look up a ThreatConnect object by routing between indicators and groups.

        Indicator-like values such as IPs, domains, URLs, hashes, and email addresses
        are sent to indicator lookup. Group-like values such as CVEs, APT names,
        malware names, report titles, and campaign names are sent to group search.
        """
        query = query.strip()
        if not query:
            return json.dumps({"error": "query is required."}, indent=2)

        detected_type = self.detect_ioc_type(query)
        if detected_type != "Unknown":
            return await self.lookup_ioc(
                indicator=query,
                __user__=__user__,
                __event_emitter__=__event_emitter__,
            )

        search_group_types = group_types or ",".join(
            [
                "Adversary",
                "Threat",
                "Intrusion Set",
                "Malware",
                "Tool",
                "Campaign",
                "Report",
                "Incident",
                "Vulnerability",
            ]
        )
        await self._emit_status(
            __event_emitter__,
            f"{query} does not look like an indicator. Searching ThreatConnect groups...",
        )
        return await self.search_groups(
            keywords=query,
            group_types=search_group_types,
            __user__=__user__,
            __event_emitter__=__event_emitter__,
        )

    async def bulk_lookup_iocs(
        self,
        indicators: str,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
        """
        Look up multiple IOCs in ThreatConnect.
        """
        indicator_list = [
            indicator.strip()
            for indicator in re.split(r"[,|\s|\n]+", indicators)
            if indicator.strip()
        ]
        if not indicator_list:
            return json.dumps({"error": "No indicators supplied."}, indent=2)

        await self._emit_status(
            __event_emitter__,
            f"Querying {len(indicator_list)} indicator(s) in ThreatConnect...",
        )

        raw_results = await asyncio.gather(
            *(self.query_threatconnect_api_async(indicator) for indicator in indicator_list)
        )
        enriched_results = await self._enrich_results_async(raw_results)

        final_results = []
        for indicator, result in zip(indicator_list, enriched_results):
            result["_metadata"] = {
                "indicator": indicator,
                "ioc_type": self.detect_ioc_type(indicator),
            }
            final_results.append(result)

        await self._emit_status(
            __event_emitter__, "Bulk IOC lookup complete.", done=True
        )
        response_payload = {
            "query_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "total_indicators": len(indicator_list),
            "results": final_results,
        }
        return json.dumps(self._compact_response_payload(response_payload), indent=2)

    async def search_indicators(
        self,
        keywords: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
        """
        Search ThreatConnect indicators using summary CONTAINS.
        """
        if not keywords.strip():
            return json.dumps({"error": "keywords is required."}, indent=2)

        await self._emit_status(
            __event_emitter__, f"Searching indicators for '{keywords}'..."
        )

        conditions = [{"field": "summary", "operator": "CONTAINS", "value": keywords}]
        if start_date:
            conditions.append(
                {"field": "dateAdded", "operator": ">=", "value": start_date}
            )
        if end_date:
            conditions.append(
                {"field": "dateAdded", "operator": "<=", "value": end_date}
            )

        result = await self._perform_search_async("indicators", conditions)
        await self._emit_status(
            __event_emitter__, "Indicator search complete.", done=True
        )
        return json.dumps(self._compact_response_payload(result), indent=2)

    async def search_groups(
        self,
        keywords: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        group_types: Optional[str] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
        """
        Search ThreatConnect groups by name, optionally constrained to group types.
        """
        if not keywords.strip():
            return json.dumps({"error": "keywords is required."}, indent=2)

        await self._emit_status(
            __event_emitter__, f"Searching groups for '{keywords}'..."
        )

        conditions = [{"field": "summary", "operator": "CONTAINS", "value": keywords}]
        if start_date:
            conditions.append(
                {"field": "dateAdded", "operator": ">=", "value": start_date}
            )
        if end_date:
            conditions.append(
                {"field": "dateAdded", "operator": "<=", "value": end_date}
            )

        type_filters = self._normalize_group_type_filters(group_types)
        if len(type_filters) == 1:
            conditions.append(
                {"field": "typeName", "operator": "=", "value": type_filters[0]}
            )
            result = await self._perform_search_async("groups", conditions)
        elif len(type_filters) > 1:
            merged_data: List[Dict[str, Any]] = []
            seen_ids: Set[int] = set()
            search_errors: List[Dict[str, str]] = []
            for group_type in type_filters:
                response = await self._perform_search_async(
                    "groups",
                    conditions
                    + [{"field": "typeName", "operator": "=", "value": group_type}],
                )
                if response.get("error"):
                    search_errors.append(
                        {"group_type": group_type, "error": str(response["error"])}
                    )
                    continue
                for group in response.get("data", []):
                    group_id = group.get("id")
                    if group_id and group_id not in seen_ids:
                        seen_ids.add(group_id)
                        merged_data.append(group)
            if search_errors and not merged_data:
                result = self._format_multi_search_errors(
                    search_errors, "ThreatConnect group search"
                )
            else:
                result = {"status": "Success", "data": merged_data}
        else:
            result = await self._perform_search_async("groups", conditions)

        await self._emit_status(__event_emitter__, "Group search complete.", done=True)
        return json.dumps(self._compact_response_payload(result), indent=2)
