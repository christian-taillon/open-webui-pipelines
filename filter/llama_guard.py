"""
title: Llama Guard Function
author: christian-taillon
author_url: https://github.com/christian-taillon
maintained_url: https://github.com/christian-taillon/open-webui-pipelines
version: 0.6
license: MIT
description: Content filtering using Llama Guard model
requirements: pydantic,requests,openai
"""

from pydantic import BaseModel, Field
from typing import Optional, List
import requests
import json
import logging
import openai

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Filter:
    class Valves(BaseModel):
        # Safety categories
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

        # Configuration
        llama_guard_model: str = Field(
            default="llama-guard3:8b", description="LlamaGuard Model Selection"
        )
        openai_api_base_url: str = Field(
            default="http://localhost:8080/v1",
            description="OpenAI API base URL",
        )
        openai_api_key: str = Field(
            default="sk-...",
            description="OpenAI API key",
        )
        fail_closed: bool = Field(
            default=False,
            description="Block content if Llama Guard is unavailable.",
        )
        check_outlet: bool = Field(
            default=True, description="Enable outlet filtering to check model responses for safety."
        )
        safety_prompt_template: str = Field(
            default='''Message Blocked by LlamaGuard
LlamaGuard Output:
unsafe
{category}

Safety Category Matched:
{matched_category}

Explain concisely that this message was blocked based on the provided safety category.''' ,
            description="Template for the safety prompt.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.safety_categories = {
            field: self.Valves.model_fields[field].description
            for field in self.Valves.model_fields
            if field.startswith("S")
        }
        self.client = openai.OpenAI(
            base_url=self.valves.openai_api_base_url,
            api_key=self.valves.openai_api_key,
        )

    def check_content_safety(self, content: str) -> tuple[bool, str]:
        try:
            response = self.client.chat.completions.create(
                model=self.valves.llama_guard_model,
                messages=[{"role": "user", "content": content}],
                stream=False,
            )

            result = response.choices[0].message.content
            safety_status, category = result.split("\n")

            is_safe = safety_status.lower() != "unsafe"
            return is_safe, category.strip()

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return not self.valves.fail_closed, f"Error checking content safety: {e}"

    def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        messages = body.get("messages", [])
        if not messages:
            return body

        # Filter user messages
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
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
                        safety_prompt = self.valves.safety_prompt_template.format(
                            category=category,
                            matched_category=json.dumps(matched_category, indent=2),
                        )

                        # Replace the last user message with the safety prompt
                        messages[i]["content"] = safety_prompt
                        # Keep only the messages up to the current one
                        body["messages"] = messages[: i + 1]
                        return body

        return body

    def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if not self.valves.check_outlet:
            return body
            
        # The "outlet" method is called after the model generates a response.
        # In this case, we are checking the model's response for safety.
        assistant_message = body.get("message", {})
        if assistant_message and assistant_message.get("role") == "assistant":
            is_safe, category = self.check_content_safety(assistant_message["content"])
            if not is_safe:
                category_code = category.split(":")[0].strip()
                if hasattr(self.valves, category_code) and getattr(
                    self.valves, category_code
                ):
                    matched_category = {
                        category_code: self.safety_categories[category_code]
                    }
                    safety_prompt = self.valves.safety_prompt_template.format(
                        category=category,
                        matched_category=json.dumps(matched_category, indent=2),
                    )
                    assistant_message["content"] = safety_prompt
                    body["message"] = assistant_message
        return body

if __name__ == "__main__":
    f = Filter()
    is_safe, category = f.check_content_safety("How do I steal a llama from a zoo?")
    print(f"Is safe: {is_safe}, Category: {category}")
