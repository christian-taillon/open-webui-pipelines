"""
title: Llama Gaurd Function
author: christian-taillon
author_url: https://github.com/christian-taillon
maintained_url: https://github.com/christian-taillon/open-webui-pipelines
version: 0.2
license: MIT
description: Content filtering using Llama Guard model
requirements: pydantic,requests
"""

from pydantic import BaseModel, Field
from typing import Optional, List
import requests
import json


class Filter:
    class Valves(BaseModel):
        S1: bool = Field(default=True, description="Violent Crimes")
        S2: bool = Field(default=True, description="Non-Violent Crimes")
        S3: bool = Field(default=True, description="Sex-Related Crimes")
        S4: bool = Field(default=True, description="Child Sexual Exploitation")
        S5: bool = Field(default=True, description="Defamation")
        S6: bool = Field(default=True, description="Specialized Advice")
        S7: bool = Field(default=True, description="Privacy")
        S8: bool = Field(default=True, description="Intellectual Property")
        S9: bool = Field(default=True, description="Indiscriminate Weapons")
        S10: bool = Field(default=True, description="Hate")
        S11: bool = Field(default=True, description="Suicide & Self-Harm")
        S12: bool = Field(default=True, description="Sexual Content")
        S13: bool = Field(default=True, description="Elections")
        llama_guard_model: str = Field(
            default="llama-guard3:8b", description="LlamaGuard Model Selection"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.safety_categories = {
            field: self.Valves.__fields__[field].description
            for field in self.Valves.__fields__
            if field.startswith("S")
        }

    def check_content_safety(self, content: str) -> tuple[bool, str]:
        try:
            url = "http://localhost:11434/api/chat"
            data = {
                "model": self.valves.llama_guard_model,
                "messages": [{"role": "user", "content": content}],
                "stream": False,
            }

            response = requests.post(url, json=data)
            response.raise_for_status()

            result = response.json()["message"]["content"]
            safety_status, category = result.split("\n")

            is_safe = safety_status.lower() != "unsafe"
            return is_safe, category.strip()

        except Exception as e:
            return True, f"Error checking content safety: {str(e)}"

    def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        messages = body.get("messages", [])
        if not messages:
            return body

        # Check all user messages in the history for safety
        safe_messages = []
        for msg in messages:
            if msg["role"] == "system":
                safe_messages.append(msg)
                continue

            if msg["role"] == "user":
                is_safe, category = self.check_content_safety(msg["content"])
                if not is_safe:
                    category_code = category.split(":")[0].strip()
                    if hasattr(self.valves, category_code) and getattr(
                        self.valves, category_code
                    ):
                        matched_category = {
                            category_code: self.safety_categories[category_code]
                        }
                        safety_prompt = f"""Message Blocked by LlamaGuard
LlamaGuard Output:
unsafe
{category}

Safety Category Matched:
{json.dumps(matched_category, indent=2)}

Explain conciesly that this message was blocked based on the provided saftey category. """
                        safe_messages = [
                            msg for msg in safe_messages if msg["role"] == "system"
                        ]
                        safe_messages.append({"role": "user", "content": safety_prompt})
                        body["messages"] = safe_messages
                        return body

            # Message is either safe or from assistant
            safe_messages.append(msg)

        body["messages"] = safe_messages
        return body
