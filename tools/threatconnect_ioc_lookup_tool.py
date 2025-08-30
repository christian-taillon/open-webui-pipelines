"""
title: ThreatConnect IOC Lookup Tool
author: christian-taillon
author_url: https://christiant.io/
funding_url: https://github.com/open-webui/open-webui
github_url: https://github.com/open-webui/open-webui
version: 4.2.0
license: MIT
description: Searches entities through ThreatConnect API integrations with automatic context enrichment for model responses.
requirements: requests
"""

import re
import hashlib
import hmac
import base64
import time
import json
from typing import Optional, List, Dict, Any, Callable, Awaitable
from pydantic import BaseModel, Field
import requests
from urllib.parse import quote, urlencode
from enum import Enum


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

        # Fields to include
        include_all_fields: bool = Field(
            default=True,
            description="Include all available fields in API responses.",
        )

        custom_fields: str = Field(
            default="",
            description="Comma-separated list of specific fields to include (if include_all_fields is False).",
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

    def get_available_fields(self, endpoint_type: str = "indicators") -> List[str]:
        """Get available fields for an endpoint."""
        # Check cache first
        if endpoint_type in self._available_fields_cache:
            return self._available_fields_cache[endpoint_type]

        try:
            endpoint = f"/api/{self.valves.api_version}/{endpoint_type}/fields"
            url = f"{self.valves.tc_api_url}{endpoint}"
            headers = self.get_headers(endpoint, "OPTIONS")

            response = requests.options(url, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                fields = [field["name"] for field in data.get("data", [])]
                # Cache the results
                self._available_fields_cache[endpoint_type] = fields
                return fields
            else:
                # Return default important fields if we can't get the full list
                return [
                    "associatedGroups",
                    "associatedIndicators",
                    "attributes",
                    "tags",
                    "securityLabels",
                    "threatAssess",
                    "observations",
                    "falsePositives",
                    "sightings",
                    "enrichment",
                    "geoLocation",
                    "dnsResolution",
                    "whoIs",
                    "investigationLinks",
                    "fileOccurrences",
                    "fileActions",
                    "trackedUsers",
                    "customAssociations",
                    "externalDates",
                ]
        except Exception:
            # Return default fields on error
            return [
                "associatedGroups",
                "associatedIndicators",
                "attributes",
                "tags",
                "securityLabels",
                "threatAssess",
                "observations",
                "falsePositives",
            ]

    def build_fields_params(self, endpoint_type: str = "indicators") -> str:
        """Build the fields parameter string for the API request."""
        if self.valves.include_all_fields:
            fields = self.get_available_fields(endpoint_type)
        else:
            # Use custom fields if specified
            if self.valves.custom_fields:
                fields = [f.strip() for f in self.valves.custom_fields.split(",")]
            else:
                # Use a sensible default set
                fields = [
                    "associatedGroups",
                    "associatedIndicators",
                    "attributes",
                    "tags",
                    "securityLabels",
                    "threatAssess",
                    "observations",
                    "falsePositives",
                ]

        # Build the fields parameter string
        return "&".join([f"fields={field}" for field in fields])

    def detect_ioc_type(self, indicator: str) -> str:
        """Detect the type of IOC."""
        # Clean the indicator
        indicator = indicator.strip()

        # Check against patterns
        if re.match(self.ioc_patterns["ip"], indicator):
            return "Address"
        elif re.match(self.ioc_patterns["url"], indicator):
            return "URL"
        elif re.match(self.ioc_patterns["email"], indicator):
            return "EmailAddress"
        elif re.match(self.ioc_patterns["hash"], indicator):
            hash_len = len(indicator)
            if hash_len == 32:
                return "File"  # MD5
            elif hash_len == 40:
                return "File"  # SHA1
            elif hash_len == 64:
                return "File"  # SHA256
            else:
                return "File"
        elif re.match(self.ioc_patterns["domain"], indicator):
            # Filter out common non-IOC domains
            excluded = ["example.com", "google.com", "microsoft.com", "amazon.com"]
            if not any(excl in indicator.lower() for excl in excluded):
                return "Host"

        return "Unknown"

    def query_threatconnect_api(
        self, indicator: str, ioc_type: str, include_all_data: bool = True
    ) -> Dict[str, Any]:
        """Query ThreatConnect API for indicator information with all available fields."""
        try:
            fields_param = (
                self.build_fields_params("indicators") if include_all_data else ""
            )

            if self.valves.api_version == "v3":
                endpoint = f"/api/{self.valves.api_version}/indicators"
                params = {
                    "resultLimit": self.valves.max_results_per_ioc,
                    "tql": f'summary="{indicator}"',
                }
                if self.valves.owner:
                    params["owner"] = self.valves.owner

                # Build URL with fields
                url = f"{self.valves.tc_api_url}{endpoint}?{urlencode(params)}"
                if fields_param:
                    url += f"&{fields_param}"
            else:
                type_mapping = {
                    "Address": "addresses",
                    "Host": "hosts",
                    "File": "files",
                    "URL": "urls",
                    "EmailAddress": "emailAddresses",
                }
                endpoint = f"/api/{self.valves.api_version}/indicators/{type_mapping.get(ioc_type, ioc_type.lower())}/{quote(indicator, safe='')}"
                url = f"{self.valves.tc_api_url}{endpoint}"
                if fields_param:
                    url += f"?{fields_param}"

            headers = self.get_headers(endpoint.split("?")[0], "GET")
            response = requests.get(
                url, headers=headers, timeout=30
            )  # Increased timeout for more data

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return {
                    "status": "not_found",
                    "message": "Indicator not found in ThreatConnect",
                }
            else:
                return {
                    "error": f"API returned status {response.status_code}",
                    "response_text": response.text,
                }

        except Exception as e:
            return {"error": str(e)}

    async def lookup_ioc(
        self,
        indicator: str,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> str:
        """
        Look up an Indicator of Compromise (IOC) in ThreatConnect with all available data.

        :param indicator: The IOC to look up (IP, domain, hash, URL, or email)
        :return: Full API response from ThreatConnect including all available fields
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Looking up {indicator} in ThreatConnect...",
                        "done": False,
                    },
                }
            )

        # Clean and validate the indicator
        indicator = indicator.strip()

        # Detect IOC type
        ioc_type = self.detect_ioc_type(indicator)

        if ioc_type == "Unknown":
            return json.dumps(
                {
                    "error": "unknown_type",
                    "message": f"Could not determine the type of indicator: {indicator}",
                    "indicator": indicator,
                },
                indent=2,
            )

        # First, get available fields for this endpoint
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Fetching all available data for {indicator}...",
                        "done": False,
                    },
                }
            )

        # Query ThreatConnect with all fields
        api_response = self.query_threatconnect_api(
            indicator, ioc_type, include_all_data=True
        )

        # Add metadata to response
        api_response["_metadata"] = {
            "indicator": indicator,
            "ioc_type": ioc_type,
            "query_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "api_version": self.valves.api_version,
            "fields_included": (
                "all" if self.valves.include_all_fields else self.valves.custom_fields
            ),
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

        # Return the full, unmodified API response as formatted JSON
        return json.dumps(api_response, indent=2)

    async def bulk_lookup_iocs(
        self,
        indicators: str,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> str:
        """
        Look up multiple IOCs in ThreatConnect with all available data.

        :param indicators: Comma or newline separated list of IOCs to look up
        :return: Full API responses for all indicators including all available fields
        """
        # Parse the input - support both comma and newline separation
        indicator_list = []
        for sep in [",", "\n", " "]:
            if sep in indicators:
                indicator_list = [i.strip() for i in indicators.split(sep) if i.strip()]
                break

        if not indicator_list:
            indicator_list = [indicators.strip()]

        results = {
            "query_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "total_indicators": len(indicator_list),
            "api_version": self.valves.api_version,
            "fields_included": (
                "all" if self.valves.include_all_fields else self.valves.custom_fields
            ),
            "results": [],
        }

        for i, indicator in enumerate(indicator_list, 1):
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Looking up {i}/{len(indicator_list)}: {indicator}",
                            "done": False,
                        },
                    }
                )

            # Get single IOC result (already returns JSON string)
            single_result = await self.lookup_ioc(indicator, __user__, None)

            # Parse it back to add to our results array
            try:
                parsed_result = json.loads(single_result)
                results["results"].append(parsed_result)
            except json.JSONDecodeError:
                results["results"].append(
                    {
                        "error": "parse_error",
                        "indicator": indicator,
                        "raw_response": single_result,
                    }
                )

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "All lookups complete", "done": True},
                }
            )

        # Return all results as formatted JSON
        return json.dumps(results, indent=2)

    async def get_available_fields_list(
        self,
        endpoint_type: str = "indicators",
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> str:
        """
        Get a list of all available fields for a ThreatConnect endpoint.

        :param endpoint_type: The endpoint type (indicators, groups, etc.)
        :return: List of available fields with descriptions
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Fetching available fields for {endpoint_type}...",
                        "done": False,
                    },
                }
            )

        try:
            endpoint = f"/api/{self.valves.api_version}/{endpoint_type}/fields"
            url = f"{self.valves.tc_api_url}{endpoint}"
            headers = self.get_headers(endpoint, "OPTIONS")

            response = requests.options(url, headers=headers, timeout=10)

            if response.status_code == 200:
                result = response.json()
                result["_metadata"] = {
                    "endpoint_type": endpoint_type,
                    "query_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                    "api_version": self.valves.api_version,
                }

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Fields list retrieved",
                                "done": True,
                            },
                        }
                    )

                return json.dumps(result, indent=2)
            else:
                return json.dumps(
                    {
                        "error": f"API returned status {response.status_code}",
                        "endpoint_type": endpoint_type,
                    },
                    indent=2,
                )

        except Exception as e:
            return json.dumps(
                {"error": str(e), "endpoint_type": endpoint_type}, indent=2
            )

    async def search_indicators(
        self,
        query: str,
        limit: int = 10,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> str:
        """
        Search ThreatConnect for indicators using TQL with all available fields.

        :param query: TQL query string (e.g., 'rating > 3', 'confidence >= 80')
        :param limit: Maximum number of results to return
        :return: Full API response with search results including all available fields
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Searching ThreatConnect: {query}",
                        "done": False,
                    },
                }
            )

        try:
            fields_param = self.build_fields_params("indicators")
            endpoint = f"/api/{self.valves.api_version}/indicators"
            params = {"resultLimit": min(limit, 100), "tql": query}
            if self.valves.owner:
                params["owner"] = self.valves.owner

            url = f"{self.valves.tc_api_url}{endpoint}?{urlencode(params)}"
            if fields_param:
                url += f"&{fields_param}"

            headers = self.get_headers(endpoint, "GET")
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200:
                result = response.json()
            else:
                result = {
                    "error": f"API returned status {response.status_code}",
                    "response_text": response.text,
                }

            # Add metadata
            result["_metadata"] = {
                "query": query,
                "limit": limit,
                "query_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "api_version": self.valves.api_version,
                "fields_included": (
                    "all"
                    if self.valves.include_all_fields
                    else self.valves.custom_fields
                ),
            }

        except Exception as e:
            result = {
                "error": str(e),
                "_metadata": {
                    "query": query,
                    "limit": limit,
                    "query_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                },
            }

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Search complete", "done": True},
                }
            )

        return json.dumps(result, indent=2)

