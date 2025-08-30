"""
title: CloudFlare Manifold Pipe
authors: christian-taillon
author_url: https://christiant.io
funding_url: https://github.com/open-webui
version: 0.1.6
required_open_webui_version: 0.1.6
license: MIT
"""

import os
import requests
import json
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message


class Pipe:
    class Valves(BaseModel):
        CF_ACCOUNT_ID: str = Field(default="")
        CF_API_KEY: str = Field(default="")
        CF_AI_Gateway: str = Field(default="")

    def __init__(self):
        self.type = "manifold"
        self.id = "cloudflare"
        self.name = "cloudflare/"
        self.valves = self.Valves(
            **{
                "CF_ACCOUNT_ID": os.getenv("CF_ACCOUNT_ID", ""),
                "CF_API_KEY": os.getenv("CF_API_KEY", ""),
                "CF_AI_Gateway": os.getenv("CF_AI_Gateway", ""),
            }
        )

    def get_cloudflare_models(self):
        return [
            {"id": "@cf/meta/llama-3.1-8b-instruct", "name": "llama3.1:8b"},
            {"id": "@cf/meta/llama-2-7b-chat-fp16", "name": "llama2:7b"},
            {"id": "@cf/meta/llama-3.3-70b-instruct-fp8-fast", "name": "llama3.3:70b"},
            {"id": "@cf/meta/llama-3.2-11b-vision-instruct", "name": "llama3.2:11b"},
            {
                "id": "@hf/thebloke/deepseek-coder-6.7b-instruct-awq",
                "name": "deepseek:6.7b",
            },
        ]

    def pipes(self) -> List[dict]:
        return self.get_cloudflare_models()

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        system_message, messages = pop_system_message(body["messages"])
        model_id = (
            body["model"].split("cloudfare_workerai.")[-1]
            if "cloudfare_workerai." in body["model"]
            else body["model"]
        )
        url = f"https://gateway.ai.cloudflare.com/v1/{self.valves.CF_ACCOUNT_ID}/{self.valves.CF_AI_Gateway}/workers-ai/{model_id}"
        headers = {
            "Authorization": f"Bearer {self.valves.CF_API_KEY}",
            "Content-Type": "application/json",
        }

        # Prepare the payload according to the API schema
        payload = {
            "messages": messages,
            "stream": body.get("stream", False),
            "max_tokens": body.get("max_tokens", 16384),
            "temperature": body.get("temperature", 0.2),
            "top_p": body.get("top_p", 1),
            "top_k": body.get("top_k", 1),
            "repetition_penalty": body.get("repetition_penalty", 1),
            "frequency_penalty": body.get("frequency_penalty", 0),
            "presence_penalty": body.get("presence_penalty", 0),
        }

        # Add system message if present
        if system_message:
            payload["messages"].insert(
                0, {"role": "system", "content": str(system_message)}
            )

        # Add optional parameters if they are present in the body
        if "seed" in body:
            payload["seed"] = body["seed"]
        if "raw" in body:
            payload["raw"] = body["raw"]

        try:
            if body.get("stream", False):
                return self.stream_response(url, headers, payload)
            else:
                return self.non_stream_response(url, headers, payload)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return f"Error: Request failed: {e}"
        except Exception as e:
            print(f"Error in pipe method: {e}")
            return f"Error: {e}"

    def stream_response(self, url, headers, payload):
        try:
            with requests.post(
                url, headers=headers, json=payload, stream=True
            ) as response:
                if response.status_code != 200:
                    raise Exception(
                        f"HTTP Error {response.status_code}: {response.text}"
                    )

                for chunk in response.iter_lines():
                    if chunk:
                        line = chunk.decode("utf-8")
                        if line.startswith("data: "):
                            if line.strip() == "data: [DONE]":
                                break
                            try:
                                json_str = line[6:]
                                data = json.loads(json_str)
                                if "response" in data:
                                    yield data["response"]
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            yield f"Error: {str(e)}"

    def non_stream_response(self, url, headers, payload):
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["response"]
