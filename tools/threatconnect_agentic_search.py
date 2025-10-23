"""
title: ThreatConnect Agentic Tool
author: christian-taillon
author_url: https://christiant.io/
funding_url: https://github.com/open-webui/open-webui
github_url: https://github.com/christian-taillon/open-webui-pipelines
version: 5.0.9
license: MIT
description: An agentic tool for OpenWebUI that interacts with the ThreatConnect API. It allows for looking up IOCs and performing simple, powerful searches for indicators and groups using a safe ThreatConnect Query Language (TQL) builder. It features concurrent API requests, rate limit handling, and efficient data caching.
requirements: httpx, tenacity
"""

import re
import hashlib
import hmac
import base64
import time
import json
import asyncio
from typing import Optional, List, Dict, Any, Callable, Awaitable, Set
from pydantic import BaseModel, Field
import httpx
from urllib.parse import urlencode
from enum import Enum
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


# Custom Exceptions for clear error handling
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
        # ThreatConnect connection
        tc_api_url: str = Field(
            default="https://app.threatconnect.com",
            description="ThreatConnect base URL (e.g., https://app.threatconnect.com).",
        )
        api_version: str = Field(
            default="v3",
            description="ThreatConnect API version to use (v2 or v3).",
        )
        owner: str = Field(
            default="",
            description="ThreatConnect owner/source to search (blank for all).",
        )

        # Authentication
        auth_method: AuthMethod = Field(
            default=AuthMethod.token,
            description="Authentication method: 'token' (TC 7.7+) or 'access_key' (legacy).",
        )
        tc_api_token: str = Field(
            default="",
            description="ThreatConnect API Token (use with auth_method=token).",
        )
        tc_api_access_id: str = Field(
            default="",
            description="ThreatConnect API Access ID (use with auth_method=access_key).",
        )
        tc_api_secret_key: str = Field(
            default="",
            description="ThreatConnect API Secret Key (use with auth_method=access_key).",
        )

        # Query behavior
        max_results_per_ioc: int = Field(
            default=10,
            description="Maximum number of results to return per IOC.",
        )
        max_search_results: int = Field(
            default=10,
            description="Default maximum number of search results to return for indicators and groups.",
        )

        # Fields to include
        include_all_fields: bool = Field(
            default=True,
            description="Include all available fields in API responses.",
        )
        custom_fields: str = Field(
            default="",
            description="Comma-separated list of specific fields to include (if include_all_fields is False).",
        )

        # Enrichment and Formatting
        enrich_groups_with_attributes: bool = Field(
            default=True,
            description="Enable/disable fetching full attributes for associated groups.",
        )
        fields_to_remove_from_indicator: str = Field(
            default="investigationLinks,ownerId,trackedUsers",
            description="Comma-separated list of top-level keys to remove from the final indicator JSON.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.ioc_patterns = {
            "ip": r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
            "domain": r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b",
            "hash": r"\b[a-fA-F0-9]{32}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{64}\b",
            "url": r"https?://[^\s<>\"{}|\\^`\[\]]+",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        }
        self._available_fields_cache = {}

    def get_headers(self, path: str, method: str = "GET") -> Dict[str, str]:
        """Generate authentication headers based on configured method."""
        if self.valves.auth_method == AuthMethod.token:
            return {"Authorization": f"TC-Token {self.valves.tc_api_token}"}
        else:
            timestamp = str(int(time.time()))
            signature = f"{path}:{method}:{timestamp}"
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RateLimitError),
    )
    async def _api_request_async(
        self, method: str, endpoint: str, params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Centralized async API request handler with rate limit retries."""
        async with httpx.AsyncClient() as client:
            url = f"{self.valves.tc_api_url}{endpoint}"
            headers = self.get_headers(endpoint, method.upper())
            try:
                response = await client.request(
                    method, url, params=params, headers=headers, timeout=30
                )
                if response.status_code == 429:
                    raise RateLimitError(f"Rate limit exceeded for {url}")
                if response.status_code == 404:
                    raise NotFoundError(f"Resource not found at {url}")
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                raise APIError(
                    f"API request failed for {e.request.url} with status {e.response.status_code}: {e.response.text}"
                )
            except httpx.RequestError as e:
                raise APIError(f"Network error for {e.request.url}: {str(e)}")

    def escape_tql_string(self, value: Any) -> str:
        """Properly escapes a string for use in a TQL query value."""
        s_value = str(value)
        s_value = s_value.replace("\\", "\\\\")
        s_value = s_value.replace('"', '\\"')
        s_value = s_value.replace("'", "\\'")
        s_value = s_value.replace("`", "\\`")
        return f'"{s_value}"'

    def _build_tql_from_struct(self, conditions: List[Dict[str, Any]]) -> str:
        """Builds a safe TQL query from a list of condition dictionaries."""
        if not conditions:
            return ""
        query_parts = []
        for cond in conditions:
            field, op, value = cond["field"], cond["operator"], cond["value"]
            formatted_value = self.escape_tql_string(value)
            query_parts.append(f"{field} {op} {formatted_value}")
        return " AND ".join(query_parts)

    async def build_fields_params_async(
        self, endpoint_type: str = "indicators"
    ) -> Dict[str, List[str]]:
        """Build the fields parameter for the API request."""
        if self.valves.include_all_fields:
            if endpoint_type not in self._available_fields_cache:
                try:
                    endpoint = f"/api/{self.valves.api_version}/{endpoint_type}/fields"
                    data = await self._api_request_async("OPTIONS", endpoint)
                    self._available_fields_cache[endpoint_type] = [
                        f["name"] for f in data.get("data", [])
                    ]
                except APIError:
                    self._available_fields_cache[endpoint_type] = [
                        "associatedGroups",
                        "attributes",
                        "tags",
                        "securityLabels",
                    ]
            fields = self._available_fields_cache[endpoint_type]
        elif self.valves.custom_fields:
            fields = [f.strip() for f in self.valves.custom_fields.split(",")]
        else:
            fields = ["associatedGroups", "attributes", "tags", "securityLabels"]
        return {"fields": fields} if fields else {}

    def detect_ioc_type(self, indicator: str) -> str:
        """Detect the type of IOC."""
        indicator = indicator.strip()
        for ioc_type, pattern in self.ioc_patterns.items():
            if re.fullmatch(pattern, indicator):
                return {
                    "ip": "Address",
                    "url": "URL",
                    "email": "EmailAddress",
                    "hash": "File",
                    "domain": "Host",
                }.get(ioc_type)
        return "Unknown"

    async def query_threatconnect_api_async(self, indicator: str) -> Dict[str, Any]:
        """Query ThreatConnect API for a single indicator."""
        try:
            endpoint = f"/api/{self.valves.api_version}/indicators"
            params = {
                "resultLimit": self.valves.max_results_per_ioc,
                "tql": f'summary="{indicator}"',
            }
            if self.valves.owner:
                params["owner"] = self.valves.owner
            field_params = await self.build_fields_params_async("indicators")
            params.update(field_params)
            return await self._api_request_async("GET", endpoint, params=params)
        except NotFoundError:
            return {"status": "not_found", "message": "Indicator not found"}
        except APIError as e:
            return {"error": str(e)}

    async def _get_group_details_async(self, group_id: int) -> Dict[str, Any]:
        """Fetch full attributes for a single group."""
        try:
            endpoint = f"/api/{self.valves.api_version}/groups/{group_id}"
            params = {"fields": ["attributes"]}
            response = await self._api_request_async("GET", endpoint, params=params)
            return response.get("data", {}).get("attributes", {})
        except APIError as e:
            return {"error": str(e)}

    async def _enrich_results_async(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enrich a list of API responses with full group attributes concurrently."""
        if not self.valves.enrich_groups_with_attributes:
            return results

        group_ids_to_fetch: Set[int] = set()
        for result in results:
            if result.get("status") == "Success":
                for item in result.get("data", []):
                    for group in item.get("associatedGroups", {}).get("data", []):
                        if group_id := group.get("id"):
                            group_ids_to_fetch.add(group_id)

        if not group_ids_to_fetch:
            return results

        tasks = [self._get_group_details_async(gid) for gid in group_ids_to_fetch]
        group_details_list = await asyncio.gather(*tasks)
        group_cache = dict(zip(group_ids_to_fetch, group_details_list))

        fields_to_remove = [
            f.strip()
            for f in self.valves.fields_to_remove_from_indicator.split(",")
            if f
        ]
        for result in results:
            if result.get("status") == "Success":
                for item in result.get("data", []):
                    for field in fields_to_remove:
                        item.pop(field, None)
                    for group in item.get("associatedGroups", {}).get("data", []):
                        if group_id := group.get("id"):
                            group["attributes"] = group_cache.get(group_id, {})
        return results

    async def _perform_search_async(
        self,
        endpoint_type: str,
        conditions: List[Dict[str, Any]],
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """A generic, private function to perform TQL searches."""
        try:
            tql_query = self._build_tql_from_struct(conditions)
            if not tql_query:
                raise ValueError("Search requires at least one parameter.")

            endpoint = f"/api/{self.valves.api_version}/{endpoint_type}"
            params = {"resultLimit": self.valves.max_search_results, "tql": tql_query}
            if self.valves.owner:
                params["owner"] = self.valves.owner

            if endpoint_type == "indicators":
                field_params = await self.build_fields_params_async("indicators")
                params.update(field_params)

            api_response = await self._api_request_async("GET", endpoint, params=params)

            if (
                api_response.get("status") == "Success"
                and self.valves.enrich_groups_with_attributes
            ):
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"Enriching {len(api_response.get('data',[]))} results...",
                                "done": False,
                            },
                        }
                    )

                if endpoint_type == "indicators":
                    enriched_responses = await self._enrich_results_async(
                        [api_response]
                    )
                    return enriched_responses[0]
                elif endpoint_type == "groups":
                    group_ids = [
                        g["id"] for g in api_response.get("data", []) if "id" in g
                    ]
                    tasks = [self._get_group_details_async(gid) for gid in group_ids]
                    attributes_list = await asyncio.gather(*tasks)
                    attributes_map = dict(zip(group_ids, attributes_list))
                    for group in api_response.get("data", []):
                        if group_id := group.get("id"):
                            group["attributes"] = attributes_map.get(group_id, {})
                    return api_response

            return api_response
        except (ValueError, TypeError, APIError) as e:
            return {"error": str(e)}

    async def lookup_ioc(
        self,
        indicator: str,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> str:
        """
        Look up a single IOC (IP, domain, hash, URL, or email) and get its details.

        :param indicator: The IOC to look up.
        :return: Enriched API response from ThreatConnect as a JSON string.
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Looking up {indicator}...",
                        "done": False,
                    },
                }
            )
        ioc_type = self.detect_ioc_type(indicator)
        if ioc_type == "Unknown":
            return json.dumps(
                {"error": f"Could not determine indicator type: {indicator}"}, indent=2
            )
        api_response = await self.query_threatconnect_api_async(indicator)
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Enriching data for {indicator}...",
                        "done": False,
                    },
                }
            )
        enriched_responses = await self._enrich_results_async([api_response])
        enriched_response = enriched_responses[0]
        enriched_response["_metadata"] = {
            "indicator": indicator,
            "ioc_type": ioc_type,
            "query_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Lookup complete for {indicator}",
                        "done": True,
                    },
                }
            )
        return json.dumps(enriched_response, indent=2)

    async def bulk_lookup_iocs(
        self,
        indicators: str,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> str:
        """
        Look up multiple IOCs (IPs, domains, etc.) concurrently.

        :param indicators: A comma or newline separated list of IOCs to look up.
        :return: Enriched API responses for all indicators.
        """
        indicator_list = [
            i.strip() for i in re.split(r"[,|\s|\n]+", indicators) if i.strip()
        ]
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Querying {len(indicator_list)} indicators...",
                        "done": False,
                    },
                }
            )
        tasks = [self.query_threatconnect_api_async(ind) for ind in indicator_list]
        raw_results = await asyncio.gather(*tasks)
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Enriching results...", "done": False},
                }
            )
        enriched_results = await self._enrich_results_async(raw_results)
        final_results = []
        for i, indicator in enumerate(indicator_list):
            result = enriched_results[i]
            result["_metadata"] = {
                "indicator": indicator,
                "ioc_type": self.detect_ioc_type(indicator),
            }
            final_results.append(result)
        response_payload = {
            "query_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "total_indicators": len(indicator_list),
            "results": final_results,
        }
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "All lookups complete", "done": True},
                }
            )
        return json.dumps(response_payload, indent=2)

    async def search_indicators(
        self,
        keywords: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> str:
        """
        Search for specific technical IOCs like IPs, domains, or file hashes by keyword.
        Use this tool when asked for specific types of indicators.

        :param keywords: REQUIRED. The keywords or values to search for (e.g., "1.2.3.4", "evil.com").
        :param start_date: Optional. The start date for the search (YYYY-MM-DD format or relative like 'TODAY() - 1 DAY').
        :param end_date: Optional. The end date for the search (YYYY-MM-DD format).
        :return: A JSON string containing the search results.
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Searching indicators for '{keywords}'...",
                        "done": False,
                    },
                }
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
        result = await self._perform_search_async(
            "indicators", conditions, __event_emitter__
        )
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Indicator search complete.", "done": True},
                }
            )
        return json.dumps(result, indent=2)

    async def search_groups(
        self,
        keywords: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> str:
        """
        Search for conceptual entities like Threat Actors, Reports, Campaigns, or Malware by keyword.
        Use this for any general question about a threat, malware family, or actor (e.g., "AdaptixC2", "NPM", "APT29").

        :param keywords: REQUIRED. The keywords to search for (e.g., "FIN7", "Cobalt Strike").
        :param start_date: Optional. The start date for the search (YYYY-MM-DD format or relative like 'TODAY() - 1 DAY').
        :param end_date: Optional. The end date for the search (YYYY-MM-DD format).
        :return: A JSON string containing the search results.
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Searching groups for '{keywords}'...",
                        "done": False,
                    },
                }
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
        result = await self._perform_search_async(
            "groups", conditions, __event_emitter__
        )
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Group search complete.", "done": True},
                }
            )
        return json.dumps(result, indent=2)
