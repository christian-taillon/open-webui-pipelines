"""
title: ThreatConnect Agentic Tool
author: christian-taillon
author_url: https://christiant.io/
funding_url: https://github.com/open-webui/open-webui
github_url: https://github.com/christian-taillon/open-webui-pipelines
version: 5.1.0
license: MIT
description: An agentic tool for OpenWebUI that interacts with the ThreatConnect API. Simple entrypoints to fetch IoCs from Groups (by IDs or keyword) with automatic traversal to associated Groups. Includes concurrent API requests, rate limit handling, and caching.
requirements: httpx, tenacity
"""

import re
import hashlib
import hmac
import base64
import time
import json
import asyncio
from typing import Optional, List, Dict, Any, Callable, Awaitable, Set, Tuple
from pydantic import BaseModel, Field
import httpx
from enum import Enum
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


# Exceptions
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
        # Connection
        tc_api_url: str = Field(
            default="https://app.threatconnect.com",
            description="ThreatConnect base URL (e.g., https://actraaz.threatconnect.com).",
        )
        api_version: str = Field(
            default="v3", description="ThreatConnect API version (v2 or v3)."
        )
        owner: str = Field(
            default="",
            description="ThreatConnect owner/source to search (blank for all).",
        )

        # Authentication
        auth_method: AuthMethod = Field(
            default=AuthMethod.token,
            description="Use 'token' (TC 7.7+) or 'access_key' (legacy).",
        )
        tc_api_token: str = Field(
            default="", description="ThreatConnect API Token (auth_method=token)."
        )
        tc_api_access_id: str = Field(
            default="",
            description="ThreatConnect API Access ID (auth_method=access_key).",
        )
        tc_api_secret_key: str = Field(
            default="",
            description="ThreatConnect API Secret Key (auth_method=access_key).",
        )

        # Search / pagination
        max_results_per_ioc: int = Field(default=10)
        max_search_results: int = Field(default=25)
        max_group_assoc_limit: int = Field(
            default=500, description="Page size for group associations."
        )
        max_group_assoc_pages: int = Field(
            default=10, description="Max pages per association list."
        )

        # Fields inclusion
        include_all_fields: bool = Field(default=True)
        custom_fields: str = Field(default="")

        # Enrichment
        enrich_groups_with_attributes: bool = Field(default=True)
        fields_to_remove_from_indicator: str = Field(
            default="investigationLinks,ownerId,trackedUsers"
        )

        # Traversal defaults
        traverse_associated_groups: bool = Field(default=True)
        max_traversal_depth: int = Field(default=2)

    def __init__(self):
        self.valves = self.Valves()
        self.ioc_patterns = {
            "ip": r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
            "domain": r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b",
            "hash": r"\b[a-fA-F0-9]{32}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{64}\b",
            "url": r"https?://[^\s<>\"{}|\\^`\[\]]+",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        }
        self._available_fields_cache: Dict[str, List[str]] = {}

    # --- Internal helpers

    def get_headers(self, path: str, method: str = "GET") -> Dict[str, str]:
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

    def _prepare_params(
        self, params: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not params:
            return params
        out: Dict[str, Any] = {}
        for k, v in params.items():
            if v is None:
                continue
            if isinstance(v, list):
                out[k] = ",".join([str(i) for i in v])
            else:
                out[k] = v
        if self.valves.owner:
            out["owner"] = self.valves.owner
        return out

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RateLimitError),
    )
    async def _api_request_async(
        self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            url = f"{self.valves.tc_api_url}{endpoint}"
            headers = self.get_headers(endpoint, method.upper())
            try:
                response = await client.request(
                    method,
                    url,
                    params=self._prepare_params(params),
                    headers=headers,
                    timeout=30,
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
        s_value = str(value)
        s_value = (
            s_value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("'", "\\'")
            .replace("`", "\\`")
        )
        return f'"{s_value}"'

    def _build_tql_from_struct(self, conditions: List[Dict[str, Any]]) -> str:
        if not conditions:
            return ""
        parts = []
        for cond in conditions:
            field, op, value = cond["field"], cond["operator"], cond["value"]
            parts.append(f"{field} {op} {self.escape_tql_string(value)}")
        return " AND ".join(parts)

    async def build_fields_params_async(
        self, endpoint_type: str = "indicators"
    ) -> Dict[str, List[str]]:
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

    # --- Indicator lookups (kept for completeness)

    async def query_threatconnect_api_async(self, indicator: str) -> Dict[str, Any]:
        try:
            endpoint = f"/api/{self.valves.api_version}/indicators"
            params = {
                "resultLimit": self.valves.max_results_per_ioc,
                "tql": f'summary="{indicator}"',
            }
            params.update(await self.build_fields_params_async("indicators"))
            return await self._api_request_async("GET", endpoint, params=params)
        except NotFoundError:
            return {"status": "not_found", "message": "Indicator not found"}
        except APIError as e:
            return {"error": str(e)}


async def _get_group_associated_indicators_async(
    self, group_id: int
) -> List[Dict[str, Any]]:
    indicators: List[Dict[str, Any]] = []
    start = 0
    limit = min(self.valves.max_group_assoc_limit, 500)
    while True:
        endpoint = f"/api/{self.valves.api_version}/groups/{group_id}"
        params = {
            "fields": ["associatedIndicators"],
            "associatedIndicators.resultStart": start,
            "associatedIndicators.resultLimit": limit,
            # Optional nested fields if supported; otherwise omit:
            # "associatedIndicators.fields": "id,type,summary,dateAdded,lastModified,ownerName,rating,confidence,tags,securityLabels",
        }
        resp = await self._api_request_async("GET", endpoint, params=params)
        data = resp.get("data", {})
        if isinstance(data, list):
            data = data[0] if data else {}
        chunk = data.get("associatedIndicators", {}).get("data", [])
        if not chunk:
            break
        indicators.extend(chunk)
        if len(chunk) < limit:
            break
        start += len(chunk)
    return indicators


async def _get_group_associated_groups_async(
    self, group_id: int
) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    start = 0
    limit = min(self.valves.max_group_assoc_limit, 500)
    while True:
        endpoint = f"/api/{self.valves.api_version}/groups/{group_id}"
        params = {
            "fields": ["associatedGroups"],
            "associatedGroups.resultStart": start,
            "associatedGroups.resultLimit": limit,
            # Optional nested fields if supported; otherwise omit:
            # "associatedGroups.fields": "id,type,name",
        }
        resp = await self._api_request_async("GET", endpoint, params=params)
        data = resp.get("data", {})
        if isinstance(data, list):
            data = data[0] if data else {}
        chunk = data.get("associatedGroups", {}).get("data", [])
        if not chunk:
            break
        groups.extend(chunk)
        if len(chunk) < limit:
            break
        start += len(chunk)
    return groups

    async def _get_group_details_async(self, group_id: int) -> Dict[str, Any]:
        try:
            endpoint = f"/api/{self.valves.api_version}/groups/{group_id}"
            params = {
                "fields": [
                    "id",
                    "name",
                    "type",
                    "ownerName",
                    "attributes",
                    "tags",
                    "securityLabels",
                ]
            }
            response = await self._api_request_async("GET", endpoint, params=params)
            data = response.get("data", {})
            if isinstance(data, list):
                data = data[0] if data else {}
            assoc_inds = await self._get_group_associated_indicators_async(group_id)
            return {
                "id": data.get("id", group_id),
                "name": data.get("name"),
                "type": data.get("type"),
                "ownerName": data.get("ownerName"),
                "attributes": data.get("attributes", {}),
                "tags": data.get("tags", {}),
                "securityLabels": data.get("securityLabels", {}),
                "associatedIndicators": {"data": assoc_inds},
            }
        except APIError as e:
            return {
                "error": str(e),
                "id": group_id,
                "associatedIndicators": {"data": []},
            }

    async def _enrich_results_async(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
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
        try:
            tql_query = self._build_tql_from_struct(conditions)
            if not tql_query:
                raise ValueError("Search requires at least one parameter.")
            endpoint = f"/api/{self.valves.api_version}/{endpoint_type}"
            params = {"resultLimit": self.valves.max_search_results, "tql": tql_query}
            if endpoint_type == "indicators":
                params.update(await self.build_fields_params_async("indicators"))
            response = await self._api_request_async("GET", endpoint, params=params)
            if (
                response.get("status") == "Success"
                and self.valves.enrich_groups_with_attributes
            ):
                if endpoint_type == "indicators":
                    enriched_responses = await self._enrich_results_async([response])
                    return enriched_responses[0]
            return response
        except (ValueError, TypeError, APIError) as e:
            return {"error": str(e)}

    # --- Simple, model-friendly entrypoints ---

    async def get_iocs_for_group_ids(
        self,
        group_ids: str,
        format: str = "json",  # "json" or "csv"
        traverse: Optional[bool] = None,
        max_depth: Optional[int] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
        """
        Fetch IoCs for given group IDs. Traverses associated groups by default.
        """
        traverse = (
            self.valves.traverse_associated_groups if traverse is None else traverse
        )
        max_depth = self.valves.max_traversal_depth if max_depth is None else max_depth

        ids = [int(x) for x in re.split(r"[,\s\n]+", group_ids) if x.strip()]
        seen_groups: Set[int] = set()
        queue: List[Tuple[int, int]] = [(gid, 0) for gid in ids]
        rows: List[Dict[str, Any]] = []
        seen_indicator_keys: Set[str] = set()  # prefer id; fallback to type+summary

        async def collect_for_group(gid: int):
            details = await self._get_group_details_async(gid)
            name = details.get("name") or ""
            gtype = details.get("type") or ""
            for ind in details.get("associatedIndicators", {}).get("data", []):
                key = str(ind.get("id") or f"{ind.get('type')}|{ind.get('summary')}")
                if key in seen_indicator_keys:
                    continue
                seen_indicator_keys.add(key)
                rows.append(
                    {
                        "groupId": gid,
                        "groupName": name,
                        "groupType": gtype,
                        "indicatorId": ind.get("id"),
                        "indicatorType": ind.get("type"),
                        "indicator": ind.get("summary"),
                        "rating": ind.get("rating"),
                        "confidence": ind.get("confidence"),
                        "dateAdded": ind.get("dateAdded"),
                        "lastModified": ind.get("lastModified"),
                        "ownerName": ind.get("ownerName"),
                    }
                )
            return details

        while queue:
            gid, depth = queue.pop(0)
            if gid in seen_groups:
                continue
            seen_groups.add(gid)
            details = await collect_for_group(gid)

            if traverse and depth < max_depth:
                try:
                    assoc_groups = await self._get_group_associated_groups_async(gid)
                    for g in assoc_groups:
                        if "id" in g and g["id"] not in seen_groups:
                            queue.append((g["id"], depth + 1))
                except APIError:
                    pass

        if format.lower() == "csv":
            import io, csv

            output = io.StringIO()
            fieldnames = [
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
            ]
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
            return output.getvalue()

        return json.dumps({"rows": rows}, indent=2)

    async def get_iocs_for_keyword(
        self,
        keywords: str,
        format: str = "json",  # "json" or "csv"
        group_types: Optional[
            str
        ] = None,  # CSV of group types; default searches common types
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        traverse: Optional[bool] = None,
        max_depth: Optional[int] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
        """
        Search Groups by keyword, then fetch IoCs with traversal.
        """
        traverse = (
            self.valves.traverse_associated_groups if traverse is None else traverse
        )
        max_depth = self.valves.max_traversal_depth if max_depth is None else max_depth

        # Build group search TQL (use name for groups; summary is for indicators)
        conditions = [{"field": "name", "operator": "CONTAINS", "value": keywords}]
        if start_date:
            conditions.append(
                {"field": "dateAdded", "operator": ">=", "value": start_date}
            )
        if end_date:
            conditions.append(
                {"field": "dateAdded", "operator": "<=", "value": end_date}
            )
        if group_types:
            # Accept CSV; build (typeName = A OR typeName = B ...) with AND wrapper
            types = [t.strip() for t in group_types.split(",") if t.strip()]
            if types:
                # For simplicity: if only one, add single condition; otherwise run multiple queries and merge
                if len(types) == 1:
                    conditions.append(
                        {"field": "typeName", "operator": "=", "value": types[0]}
                    )
                    group_search = await self._perform_search_async(
                        "groups", conditions, __event_emitter__
                    )
                    groups = group_search.get("data", [])
                else:
                    groups = []
                    for t in types:
                        conds = conditions + [
                            {"field": "typeName", "operator": "=", "value": t}
                        ]
                        r = await self._perform_search_async(
                            "groups", conds, __event_emitter__
                        )
                        groups.extend(r.get("data", []))
        else:
            # Default across common types by running a few targeted queries and merging
            common_types = ["Report", "Threat Actor", "Campaign", "Incident", "Malware"]
            groups = []
            for t in common_types:
                conds = conditions + [
                    {"field": "typeName", "operator": "=", "value": t}
                ]
                r = await self._perform_search_async("groups", conds, __event_emitter__)
                groups.extend(r.get("data", []))

        if not groups:
            return json.dumps({"rows": []}, indent=2)

        id_list = ",".join([str(g["id"]) for g in groups if "id" in g])
        return await self.get_iocs_for_group_ids(
            id_list,
            format=format,
            traverse=traverse,
            max_depth=max_depth,
            __event_emitter__=__event_emitter__,
        )

    # Backwards-compatible names
    async def get_group_iocs(
        self,
        group_ids: str,
        as_csv: bool = False,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
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
        group_type: str = "",  # kept for compatibility; prefer group_types in get_iocs_for_keyword
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
        gt = group_type if group_type else None
        return await self.get_iocs_for_keyword(
            keywords=keywords,
            format="csv",
            group_types=gt,
            traverse=self.valves.traverse_associated_groups,
            max_depth=self.valves.max_traversal_depth,
            __event_emitter__=__event_emitter__,
        )

    # Legacy functions (unchanged signatures) if needed by other pipelines

    async def lookup_ioc(
        self,
        indicator: str,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
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
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
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
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
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
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
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
        # Use 'name' for group queries
        conditions = [{"field": "name", "operator": "CONTAINS", "value": keywords}]
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

