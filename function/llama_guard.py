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

        openai_api_base: str = Field(
            default="http://host.docker.internal:11434/v1",
            description="Base URL for Ollama API",
        )
        llama_guard_model: str = Field(
            default="llama-guard3:8b", description="LlamaGuard Model Selection"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.safety_categories = {
            "S1": "Violent Crimes",
            "S2": "Non-Violent Crimes",
            "S3": "Sex-Related Crimes",
            "S4": "Child Sexual Exploitation",
            "S5": "Defamation",
            "S6": "Specialized Advice",
            "S7": "Privacy",
            "S8": "Intellectual Property",
            "S9": "Indiscriminate Weapons",
            "S10": "Hate",
            "S11": "Suicide & Self-Harm",
            "S12": "Sexual Content",
            "S13": "Elections",
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

        last_message = messages[-1]["content"]
        is_safe, category = self.check_content_safety(last_message)

        if not is_safe:
            # Extract category code (S1, S2, etc.) from LlamaGuard output
            category_code = category.split(":")[0].strip()

            # Check if this category is enabled in valves
            if hasattr(self.valves, category_code) and getattr(
                self.valves, category_code
            ):
                safety_prompt = f"""Message Blocked by LlamaGuard
    
    LlamaGuard Output:
    unsafe
    {category}
    
    Safety Categories Reference:
    {json.dumps(self.safety_categories, indent=2)}
    
    Please explain that this message was blocked based on the LlamaGuard output and safety category matched from above."""

                messages[-1]["content"] = safety_prompt
                body["messages"] = messages
            # If category is disabled, treat as safe and proceed with original message

        return body
