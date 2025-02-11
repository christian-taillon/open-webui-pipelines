"""
title: SecureTokenization Filter
author: christian-taillon
author_url: https://github.com/christian-taillon
funding_url: https://github.com/christian-taillon
version: 0.1
license: MIT
description: WIP : Tokenizes sensitive information in prompts and detokenizes in responses
requirements: cryptography,requests
"""

import re
import base64
import json
import requests
from cryptography.fernet import Fernet
from typing import Optional
from pydantic import BaseModel, Field


class Filter:
    # Default patterns as a JSON object
    # Add your own patterns here
    default_patterns = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "host": r"(?i)\b((?:(?!-)[a-zA-Z0-9-]{1,63}(?<!-)\.)+(?:[a-zA-Z]{2,}))\b",
        "ipv4": r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
        "url": r"https?://[^\s<>\"]+|www\.[^\s<>\"]+",
    }

    class Valves(BaseModel):
        priority: int = Field(default=0)
        enabled_for_admins: bool = Field(
            default=True, description="Enable tokenization for admin users"
        )
        enable_detokenization: bool = Field(
            default=True, description="Enable protection of sensitive data"
        )
        github_url: str = Field(
            default="", description="NOT IMPLEMENTED YET - RAW GitHub URL with Custom Regex"
        )

    def __init__(self):
        self.DEBUG_GATEWAY = False
        self.file_handler = False
        self.valves = self.Valves()
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)
        self.token_map = {}
        self.token_count = 0
        self.patterns = {}

        # Load patterns based on whether github_url is provided
        self.load_patterns()

    def load_patterns(self):
        """Load patterns either from github_url or default patterns"""
        if self.valves.github_url:
            try:
                response = requests.get(self.valves.github_url)
                response.raise_for_status()
                custom_patterns = response.json()
                # Compile each pattern from the JSON
                for pattern_name, pattern in custom_patterns.items():
                    try:
                        self.patterns[pattern_name] = re.compile(pattern)
                    except re.error as e:
                        print(f"Error compiling pattern {pattern_name}: {e}")
            except (requests.RequestException, json.JSONDecodeError) as e:
                print(f"Error loading patterns from GitHub: {e}")
                # Fallback to default patterns if GitHub load fails
                self._load_default_patterns()
        else:
            self._load_default_patterns()

    def _load_default_patterns(self):
        """Load the default patterns"""
        for pattern_name, pattern in self.default_patterns.items():
            try:
                self.patterns[pattern_name] = re.compile(pattern)
            except re.error as e:
                print(f"Error compiling default pattern {pattern_name}: {e}")

    def tokenize(self, text: str, pattern_key: str) -> str:
        def replace_match(match):
            sensitive_data = match.group(0)
            token = base64.urlsafe_b64encode(
                self.cipher_suite.encrypt(sensitive_data.encode())
            ).decode()
            self.token_map[token] = sensitive_data
            return f"<TOKEN_{pattern_key}_{token}>"

        return self.patterns[pattern_key].sub(replace_match, text)

    def detokenize(self, text: str) -> str:
        for token, original in self.token_map.items():
            text = text.replace(token, original)
        return text

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        messages = body.get("messages", [])
        for message in messages:
            if message.get("role") == "user":
                content = message["content"]
                matches_found = []

                # Test each pattern
                for pattern_key, pattern in self.patterns.items():
                    matches = pattern.finditer(content)
                    for match in matches:
                        matched_text = match.group(0)
                        matches_found.append(f"{pattern_key}: {matched_text}")

                        self.token_count += 1
                        token = f"<TOKEN_{pattern_key}_{self.token_count}>"
                        self.token_map[token] = matched_text

                        content = content.replace(matched_text, token)

                message["content"] = (
                    "Preserve the formatting of the token values."
                    + content
                    + " [FILTER ENGAGED]"
                )
                if self.DEBUG_GATEWAY:
                    message[
                        "content"
                    ] += f"\nPatterns active: {list(self.patterns.keys())}"
                    message["content"] += f"\nMatches found: {matches_found}"
                    message["content"] += f"\nTokens created: {len(self.token_map)}"

        return body

    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        if self.valves.enable_detokenization:
            if "messages" in body:
                for message in body["messages"]:
                    if "content" in message:
                        message["content"] = self.detokenize(message["content"])
        self.reset_token_map()
        return body

    def reset_token_map(self):
        """Reset the token map and counter"""
        self.token_map = {}
        self.token_count = 0
