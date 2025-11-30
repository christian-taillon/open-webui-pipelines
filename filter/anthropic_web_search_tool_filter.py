"""
title: Anthropic Web Search Tool Filter
description: Use Anthropic server-side web-search tool instead of Open WebUI built-in feature when enabled.
author: Johan Grande
inspired_by: https://github.com/owndev/Open-WebUI-Functions/blob/main/filters/google_search_tool.py
version: 1.0.0
license: MIT
"""

import logging
import os

from typing import Dict


class Filter:
    def __init__(self):
        logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

    def inlet(self, body: Dict) -> Dict:
        features = body.get("features", {}) or {}

        # When the generic feature flag is set, attach Anthropic's web_search tool
        if features.pop("web_search"):
            logging.debug("Replacing web_search tool with Anthropic's own")
            tools = body.setdefault("tools", [])
            if not any(
                isinstance(t, dict) and t.get("type") == "web_search" for t in tools
            ):
                tools.append(
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        # Optional: Limit the number of searches per request
                        "max_uses": 5,
                        # Optional: Only include results from these domains
                        # "allowed_domains": ["example.com", "trusteddomain.org"],
                        # Optional: Never include results from these domains
                        # "blocked_domains": ["untrustedsource.com"],
                        # Optional: Localize search results
                        # "user_location": {
                        #    "type": "approximate",
                        #    "city": "San Francisco",
                        #    "region": "California",
                        #    "country": "US",
                        #    "timezone": "America/Los_Angeles"
                        # }
                    }
                )

        return body
