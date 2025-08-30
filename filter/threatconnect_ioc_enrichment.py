"""
title: ThreatConnect IOC Enrichment Filter
author: christian-taillon
author_url: https://christiant.io/
funding_url: https://github.com/christian-taillon
github_url: https://github.com/christian-taillon/open-webui-pipelines
version: 0.7
license: MIT
description: Automatically extracts and enriches indicators of compromise using ThreatConnect's threat intelligence platform
requirements: requests
"""

import re
import json
import hashlib
import hmac
import base64
import time
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
import requests
from urllib.parse import quote, urlencode, urlparse
from enum import Enum


class AuthMethod(str, Enum):
    TOKEN = "token"
    ACCESS_KEY = "access_key"


class Filter:
    """
    ThreatConnect IOC Enrichment Filter
    Extracts indicators from user prompts and enriches them with threat intelligence
    """

    class Valves(BaseModel):
        tc_api_url: str = Field(
            default="https://api.threatconnect.com",
            description="ThreatConnect API base URL (e.g., https://api.threatconnect.com)",
        )
        auth_method: str = Field(
            default="token",
            description="Authentication method: 'token' (TC 7.7+) or 'access_key' (legacy)",
        )
        tc_api_token: str = Field(
            default="",
            description="ThreatConnect API Token (for token auth method, format: APIV2:XXX:XXX:XXX:XXX)",
        )
        tc_api_access_id: str = Field(
            default="",
            description="ThreatConnect API Access ID (for access_key auth method)",
        )
        tc_api_secret_key: str = Field(
            default="",
            description="ThreatConnect API Secret Key (for access_key auth method)",
        )
        enable_auto_enrichment: bool = Field(
            default=True, description="Automatically enrich IOCs found in user messages"
        )
        max_iocs_per_request: int = Field(
            default=10, description="Maximum number of IOCs to enrich per user message"
        )
        ioc_types: str = Field(
            default="Address,Host,File,URL,EmailAddress",
            description="Comma-separated TC indicator types (Address,Host,File,URL,EmailAddress)",
        )
        confidence_threshold: int = Field(
            default=0,
            description="Minimum confidence score to include indicators (0-100)",
        )
        include_tags: bool = Field(
            default=True, description="Include tags in enrichment data"
        )
        include_attributes: bool = Field(
            default=True, description="Include attributes in enrichment data"
        )
        include_associations: bool = Field(
            default=True, description="Include associated indicators"
        )
        owner: str = Field(
            default="",
            description="ThreatConnect Owner to search within (leave empty for all)",
        )
        result_limit: int = Field(
            default=100, description="Maximum results per indicator search"
        )
        timeout_seconds: int = Field(
            default=10, description="API request timeout in seconds"
        )
        api_version: str = Field(
            default="v3", description="ThreatConnect API version (v2 or v3)"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.ioc_patterns = {
            "Address": [
                r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
                r"\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b",  # IPv6
            ],
            "Host": [
                r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b"
            ],
            "URL": [r'https?://[^\s<>"{}|\\^`\[\]]+', r'ftp://[^\s<>"{}|\\^`\[\]]+'],
            "EmailAddress": [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
            "File": [
                r"\b[a-fA-F0-9]{32}\b",  # MD5
                r"\b[a-fA-F0-9]{40}\b",  # SHA1
                r"\b[a-fA-F0-9]{64}\b",  # SHA256
            ],
        }

    def extract_iocs(self, text: str) -> Dict[str, List[str]]:
        """Extract IOCs from text based on configured types"""
        extracted = {}
        enabled_types = [t.strip() for t in self.valves.ioc_types.split(",")]

        for ioc_type in enabled_types:
            if ioc_type not in self.ioc_patterns:
                continue

            found_iocs = set()
            for pattern in self.ioc_patterns[ioc_type]:
                matches = re.findall(pattern, text, re.IGNORECASE)
                found_iocs.update(matches)

            # Filter out false positives
            if ioc_type == "Host":
                # Remove common false positives
                found_iocs = {
                    d
                    for d in found_iocs
                    if not d.endswith(".local")
                    and not d.endswith(".localhost")
                    and not d.endswith(".test")
                    and "example.com" not in d.lower()
                    and "github.com" not in d.lower()
                    and "google.com" not in d.lower()
                    and "openai.com" not in d.lower()
                    and "microsoft.com" not in d.lower()
                }

            elif ioc_type == "Address":
                # Remove private IPs unless specifically wanted
                found_iocs = {
                    ip
                    for ip in found_iocs
                    if not ip.startswith("192.168.")
                    and not ip.startswith("10.")
                    and not ip.startswith("172.")
                    and not ip == "127.0.0.1"
                    and not ip == "0.0.0.0"
                }

            if found_iocs:
                extracted[ioc_type] = list(found_iocs)[
                    : self.valves.max_iocs_per_request
                ]

        return extracted

    def generate_tc_auth_headers(
        self, method: str, path: str, params: Dict = None
    ) -> Dict[str, str]:
        """Generate ThreatConnect API authentication headers based on configured method"""

        # API Token Authentication (TC 7.7+)
        if self.valves.auth_method == AuthMethod.TOKEN:
            if not self.valves.tc_api_token:
                return {}

            return {
                "Authorization": f"TC-Token {self.valves.tc_api_token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }

        # Access ID and Secret Key Authentication (Legacy)
        elif self.valves.auth_method == AuthMethod.ACCESS_KEY:
            if not self.valves.tc_api_access_id or not self.valves.tc_api_secret_key:
                return {}

            timestamp = str(int(time.time()))

            # Build signature string
            signature_parts = []

            # Add path with query string if params exist
            if params:
                query_string = urlencode(sorted(params.items()))
                full_path = f"{path}?{query_string}"
                signature_parts.append(full_path)
            else:
                signature_parts.append(path)

            # Add method and timestamp
            signature_parts.append(method)
            signature_parts.append(timestamp)

            # Join with colons
            signature_string = ":".join(signature_parts)

            # Create HMAC signature
            signature = base64.b64encode(
                hmac.new(
                    self.valves.tc_api_secret_key.encode("utf-8"),
                    signature_string.encode("utf-8"),
                    hashlib.sha256,
                ).digest()
            ).decode("utf-8")

            return {
                "Authorization": f"TC {self.valves.tc_api_access_id}:{signature}",
                "Timestamp": timestamp,
                "Accept": "application/json",
                "Content-Type": "application/json",
            }

        return {}

    def validate_auth_config(self) -> tuple[bool, str]:
        """Validate authentication configuration"""
        if self.valves.auth_method == AuthMethod.TOKEN:
            if not self.valves.tc_api_token:
                return False, "API Token not configured"
            # Basic validation of token format
            if not self.valves.tc_api_token.startswith("APIV2:"):
                return False, "Invalid API Token format (should start with APIV2:)"
            return True, "Token authentication configured"

        elif self.valves.auth_method == AuthMethod.ACCESS_KEY:
            if not self.valves.tc_api_access_id:
                return False, "API Access ID not configured"
            if not self.valves.tc_api_secret_key:
                return False, "API Secret Key not configured"
            return True, "Access ID/Secret Key authentication configured"

        else:
            return False, f"Invalid authentication method: {self.valves.auth_method}"

    def search_indicators(
        self, indicator_value: str, indicator_type: str
    ) -> List[Dict]:
        """Search for indicators using ThreatConnect API"""
        try:
            # Validate auth configuration
            auth_valid, auth_message = self.validate_auth_config()
            if not auth_valid:
                return [{"error": auth_message}]

            # API path based on version
            if self.valves.api_version == "v3":
                path = "/api/v3/indicators"

                # Build TQL query for exact match
                tql = (
                    f'typeName == "{indicator_type}" AND summary == "{indicator_value}"'
                )

                # Build query parameters
                params = {
                    "tql": tql,
                    "resultLimit": str(self.valves.result_limit),
                    "fields": "tags,attributes,associatedIndicators,associatedGroups",
                }
            else:  # v2
                # Map type to v2 endpoint
                type_mapping = {
                    "Address": "addresses",
                    "Host": "hosts",
                    "File": "files",
                    "URL": "urls",
                    "EmailAddress": "emailAddresses",
                }
                tc_type = type_mapping.get(indicator_type, indicator_type.lower())

                # For v2, we query specific indicator directly
                encoded_value = quote(indicator_value, safe="")
                path = f"/api/v2/indicators/{tc_type}/{encoded_value}"
                params = {}

            if self.valves.owner:
                params["owner"] = self.valves.owner

            # Generate auth headers
            headers = self.generate_tc_auth_headers("GET", path, params)
            if not headers:
                return [{"error": "Failed to generate authentication headers"}]

            # Make API request
            url = f"{self.valves.tc_api_url}{path}"

            response = requests.get(
                url,
                headers=headers,
                params=params if params else None,
                timeout=self.valves.timeout_seconds,
            )

            if response.status_code == 200:
                data = response.json()

                if self.valves.api_version == "v3":
                    indicators = data.get("data", [])

                    if not indicators:
                        return [
                            {
                                "indicator": indicator_value,
                                "type": indicator_type,
                                "status": "not_found",
                                "message": "Indicator not found in ThreatConnect",
                            }
                        ]

                    # Format all matching indicators
                    formatted_results = []
                    for ind in indicators:
                        formatted_results.append(
                            self.format_indicator_v3(
                                ind, indicator_value, indicator_type
                            )
                        )

                    return formatted_results
                else:  # v2
                    indicator = data.get("data", {}).get(indicator_type.lower(), {})
                    if indicator:
                        return [
                            self.format_indicator_v2(
                                indicator, indicator_value, indicator_type
                            )
                        ]
                    else:
                        return [
                            {
                                "indicator": indicator_value,
                                "type": indicator_type,
                                "status": "not_found",
                                "message": "Indicator not found in ThreatConnect",
                            }
                        ]

            elif response.status_code == 404:
                return [
                    {
                        "indicator": indicator_value,
                        "type": indicator_type,
                        "status": "not_found",
                        "message": "Indicator not found in ThreatConnect",
                    }
                ]
            elif response.status_code == 401:
                return [
                    {
                        "indicator": indicator_value,
                        "type": indicator_type,
                        "status": "error",
                        "message": f"Authentication failed: {response.text}",
                    }
                ]
            else:
                return [
                    {
                        "indicator": indicator_value,
                        "type": indicator_type,
                        "status": "error",
                        "message": f"API returned status {response.status_code}: {response.text}",
                    }
                ]

        except requests.exceptions.Timeout:
            return [
                {
                    "indicator": indicator_value,
                    "type": indicator_type,
                    "status": "error",
                    "message": "API request timed out",
                }
            ]
        except Exception as e:
            return [
                {
                    "indicator": indicator_value,
                    "type": indicator_type,
                    "status": "error",
                    "message": str(e),
                }
            ]

    def format_indicator_v3(self, data: Dict, indicator: str, ioc_type: str) -> Dict:
        """Format ThreatConnect v3 indicator response for LLM consumption"""
        formatted = {
            "indicator": indicator,
            "type": ioc_type,
            "status": "found",
            "api_version": "v3",
        }

        # Core fields
        if "id" in data:
            formatted["tc_id"] = data["id"]

        if "ownerName" in data:
            formatted["owner"] = data["ownerName"]

        if "dateAdded" in data:
            formatted["first_seen"] = data["dateAdded"]

        if "lastModified" in data:
            formatted["last_modified"] = data["lastModified"]

        if "confidence" in data:
            formatted["confidence"] = data["confidence"]
            if data["confidence"] < self.valves.confidence_threshold:
                formatted["below_threshold"] = True

        if "rating" in data:
            formatted["rating"] = data["rating"]

        if "threatAssessRating" in data:
            formatted["threat_rating"] = data["threatAssessRating"]

        if "threatAssessConfidence" in data:
            formatted["threat_confidence"] = data["threatAssessConfidence"]

        # Include active/inactive status
        if "active" in data:
            formatted["active"] = data["active"]
            if "activeLocked" in data:
                formatted["active_locked"] = data["activeLocked"]

        # Include tags if enabled
        if self.valves.include_tags and "tags" in data and "data" in data["tags"]:
            tag_list = []
            for tag in data["tags"]["data"]:
                tag_name = tag.get("name", "")
                if tag_name:
                    tag_list.append(tag_name)
            if tag_list:
                formatted["tags"] = tag_list

        # Include attributes if enabled
        if (
            self.valves.include_attributes
            and "attributes" in data
            and "data" in data["attributes"]
        ):
            attrs = {}
            for attr in data["attributes"]["data"]:
                attr_type = attr.get("type", "unknown")
                attr_value = attr.get("value", "")
                if attr_type and attr_value:
                    # Group multiple values of same type
                    if attr_type in attrs:
                        if isinstance(attrs[attr_type], list):
                            attrs[attr_type].append(attr_value)
                        else:
                            attrs[attr_type] = [attrs[attr_type], attr_value]
                    else:
                        attrs[attr_type] = attr_value
            if attrs:
                formatted["attributes"] = attrs

        # Include association count if enabled
        if self.valves.include_associations:
            if (
                "associatedIndicators" in data
                and "count" in data["associatedIndicators"]
            ):
                formatted["associated_indicators"] = data["associatedIndicators"][
                    "count"
                ]

            if "associatedGroups" in data and "count" in data["associatedGroups"]:
                formatted["associated_groups"] = data["associatedGroups"]["count"]

        return formatted

    def format_indicator_v2(self, data: Dict, indicator: str, ioc_type: str) -> Dict:
        """Format ThreatConnect v2 indicator response for LLM consumption"""
        formatted = {
            "indicator": indicator,
            "type": ioc_type,
            "status": "found",
            "api_version": "v2",
        }

        # Core fields (v2 format)
        if "id" in data:
            formatted["tc_id"] = data["id"]

        if "owner" in data and "name" in data["owner"]:
            formatted["owner"] = data["owner"]["name"]

        if "dateAdded" in data:
            formatted["first_seen"] = data["dateAdded"]

        if "lastModified" in data:
            formatted["last_modified"] = data["lastModified"]

        if "confidence" in data:
            formatted["confidence"] = data["confidence"]

        if "rating" in data:
            formatted["rating"] = data["rating"]

        if "threatAssessRating" in data:
            formatted["threat_rating"] = data["threatAssessRating"]

        if "threatAssessConfidence" in data:
            formatted["threat_confidence"] = data["threatAssessConfidence"]

        return formatted

    def enrich_iocs(self, iocs: Dict[str, List[str]]) -> List[Dict]:
        """Enrich all extracted IOCs with ThreatConnect data"""
        enriched = []

        for ioc_type, values in iocs.items():
            for value in values:
                results = self.search_indicators(value, ioc_type)
                enriched.extend(results)

        return enriched

    def format_enrichment_for_llm(self, enrichments: List[Dict]) -> str:
        """Format enrichment data as context for the LLM"""
        if not enrichments:
            return ""

        context_parts = ["\n[THREAT INTELLIGENCE ENRICHMENT]"]
        context_parts.append("=" * 50)

        # Add auth method info
        auth_method_display = (
            "API Token"
            if self.valves.auth_method == AuthMethod.TOKEN
            else "Access ID/Secret Key"
        )
        context_parts.append(
            f"Source: ThreatConnect ({auth_method_display} Auth, API {self.valves.api_version})"
        )

        # Group by status
        found_indicators = []
        not_found_indicators = []
        error_indicators = []

        for item in enrichments:
            status = item.get("status", "unknown")
            if status == "found":
                found_indicators.append(item)
            elif status == "not_found":
                not_found_indicators.append(item)
            elif status == "error":
                error_indicators.append(item)

        # Report found indicators with details
        if found_indicators:
            context_parts.append("\n✓ KNOWN INDICATORS:")
            for item in found_indicators:
                # Build detailed summary
                summary_parts = [f"\n• {item['type']}: {item['indicator']}"]

                # Add threat assessment
                threat_level = "Unknown"
                if "rating" in item:
                    rating = item["rating"]
                    if rating >= 4:
                        threat_level = "🔴 HIGH RISK"
                    elif rating >= 2:
                        threat_level = "🟡 MEDIUM RISK"
                    else:
                        threat_level = "🟢 LOW RISK"
                    summary_parts.append(
                        f"  Threat Level: {threat_level} (Rating: {rating}/5)"
                    )

                if "confidence" in item:
                    summary_parts.append(f"  Confidence: {item['confidence']}%")

                if "threat_confidence" in item:
                    summary_parts.append(
                        f"  Threat Assessment Confidence: {item['threat_confidence']}%"
                    )

                if "active" in item:
                    status = "Active" if item["active"] else "Inactive"
                    summary_parts.append(f"  Status: {status}")

                if "owner" in item:
                    summary_parts.append(f"  Source: {item['owner']}")

                if "tags" in item and item["tags"]:
                    tags_str = ", ".join(item["tags"][:10])  # Limit to first 10 tags
                    summary_parts.append(f"  Tags: {tags_str}")

                if (
                    "associated_indicators" in item
                    and item["associated_indicators"] > 0
                ):
                    summary_parts.append(
                        f"  Related Indicators: {item['associated_indicators']}"
                    )

                if "associated_groups" in item and item["associated_groups"] > 0:
                    summary_parts.append(
                        f"  Threat Groups: {item['associated_groups']}"
                    )

                if "attributes" in item:
                    # Show important attributes
                    important_attrs = [
                        "Description",
                        "Malware Family",
                        "Source",
                        "Threat Type",
                        "Country",
                        "Organization",
                    ]
                    for attr_name in important_attrs:
                        if attr_name in item["attributes"]:
                            value = item["attributes"][attr_name]
                            if isinstance(value, list):
                                value = ", ".join(value)
                            summary_parts.append(f"  {attr_name}: {value}")

                context_parts.extend(summary_parts)

        # Report not found indicators
        if not_found_indicators:
            context_parts.append("\n○ UNKNOWN INDICATORS (not in threat database):")
            for item in not_found_indicators:
                context_parts.append(f"• {item['type']}: {item['indicator']}")

        # Report errors if any
        if error_indicators:
            context_parts.append("\n✗ LOOKUP ERRORS:")
            for item in error_indicators:
                context_parts.append(
                    f"• {item['type']}: {item['indicator']} - {item.get('message', 'Unknown error')}"
                )

        context_parts.append("\n" + "=" * 50)
        context_parts.append("[END THREAT INTELLIGENCE ENRICHMENT]\n")

        return "\n".join(context_parts)

    async def inlet(
        self, body: dict, __event_emitter__, __user__: Optional[dict] = None
    ) -> dict:
        """Pre-process user input to add IOC enrichment context"""

        if not self.valves.enable_auto_enrichment:
            return body

        # Validate authentication configuration
        auth_valid, auth_message = self.validate_auth_config()
        if not auth_valid:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"ThreatConnect: {auth_message}",
                        "done": True,
                    },
                }
            )
            return body

        # Get the last user message
        messages = body.get("messages", [])
        if not messages:
            return body

        last_user_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg
                break

        if not last_user_msg:
            return body

        user_content = last_user_msg.get("content", "")

        # Extract IOCs
        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": "Scanning for threat indicators...",
                    "done": False,
                },
            }
        )

        extracted_iocs = self.extract_iocs(user_content)

        if not extracted_iocs:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "No indicators found to enrich",
                        "done": True,
                    },
                }
            )
            return body

        # Count total IOCs
        total_iocs = sum(len(values) for values in extracted_iocs.values())
        ioc_summary = ", ".join([f"{len(v)} {k}" for k, v in extracted_iocs.items()])

        auth_method_display = (
            "Token" if self.valves.auth_method == AuthMethod.TOKEN else "Access Key"
        )
        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": f"Found {total_iocs} indicators ({ioc_summary}). Querying ThreatConnect ({auth_method_display})...",
                    "done": False,
                },
            }
        )

        # Enrich IOCs
        enrichments = self.enrich_iocs(extracted_iocs)

        if enrichments:
            # Count successful enrichments
            found_count = sum(1 for e in enrichments if e.get("status") == "found")

            # Format enrichment as context
            context = self.format_enrichment_for_llm(enrichments)

            if context:
                # Add context as a system message
                context_message = {"role": "system", "content": context}

                # Insert after any existing system messages
                insert_pos = 0
                for i, msg in enumerate(messages):
                    if msg.get("role") != "system":
                        insert_pos = i
                        break
                    insert_pos = i + 1

                messages.insert(insert_pos, context_message)
                body["messages"] = messages

                status_msg = f"Enriched {found_count}/{total_iocs} indicators with threat intelligence"
                if found_count > 0:
                    high_risk = sum(
                        1
                        for e in enrichments
                        if e.get("status") == "found" and e.get("rating", 0) >= 4
                    )
                    if high_risk > 0:
                        status_msg += f" (⚠️ {high_risk} high-risk)"

                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": status_msg, "done": True},
                    }
                )

        return body
