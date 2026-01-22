"""
title: Ollama Web Search Tool
author: Christian Taillon  
funding_url: https://github.com/open-webui
version: 0.1.0
license: MIT
description: Search the web using Ollama's Web Search API
"""

import os
import requests
import json
import asyncio
from typing import Optional, Callable, Any
from pydantic import BaseModel, Field


class EventEmitter:
    def __init__(self, event_emitter: Optional[Callable[[dict], Any]] = None):
        self.event_emitter = event_emitter

    async def emit_status(self, description="Unknown State", status="in_progress", done=False):
        if self.event_emitter:
            await self.event_emitter({
                "type": "status",
                "data": {
                    "status": status,
                    "description": description,
                    "done": done,
                },
            })


class Tools:
    class Valves(BaseModel):
        OLLAMA_API_KEY: str = Field(
            default="",
            description="Ollama API key for web search access",
        )
        MAX_RESULTS: int = Field(
            default=5,
            description="Maximum number of search results to return (max 10)",
            ge=1,
            le=10,
        )
        TIMEOUT: int = Field(
            default=30,
            description="Request timeout in seconds",
            ge=1,
        )
        CITATION_LINKS: bool = Field(
            default=False,
            description="If True, send custom citations with links",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.base_url = "https://ollama.com"
        self.citation = False  # Required when using custom citations

    def _get_headers(self):
        """Get headers for API requests"""
        return {
            "Authorization": f"Bearer {self.valves.OLLAMA_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }

    async def search_web(
        self,
        query: str,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Search the web using Ollama's Web Search API and return relevant results.
        :params query: Web Query used in search engine.
        :return: The search results in JSON format.
        """
        emitter = EventEmitter(__event_emitter__)

        await emitter.emit_status(f"Initiating web search for: {query}")

        if not self.valves.OLLAMA_API_KEY:
            error_msg = "Ollama API key is required. Please set the OLLAMA_API_KEY in the valves."
            await emitter.emit_status(
                status="error",
                description=error_msg,
                done=True,
            )
            return json.dumps({"error": error_msg})

        try:
            await emitter.emit_status("Sending request to Ollama web search API")

            payload = {
                "query": query,
                "max_results": self.valves.MAX_RESULTS
            }

            response = await asyncio.to_thread(
                requests.post,
                f"{self.base_url}/api/web_search",
                headers=self._get_headers(),
                json=payload,
                timeout=self.valves.TIMEOUT
            )
            response.raise_for_status()

            data = await asyncio.to_thread(response.json)
            await emitter.emit_status(f"Retrieved {len(data.get('results', []))} search results")

            results_json = []
            results = data.get("results", [])

            for result in results:
                if isinstance(result, dict) and all(key in result for key in ["title", "url", "content"]):
                    results_json.append({
                        "title": result["title"],
                        "url": result["url"],
                        "content": result["content"],
                        "snippet": result.get("content", "")[:200] + "..." if len(result.get("content", "")) > 200 else result.get("content", "")
                    })

                    # Send citation if enabled
                    if self.valves.CITATION_LINKS and __event_emitter__:
                        await __event_emitter__({
                            "type": "citation",
                            "data": {
                                "document": [result["content"]],
                                "metadata": [{"source": result["url"]}],
                                "source": {"name": result["title"]},
                            },
                        })

            await emitter.emit_status(
                status="complete",
                description=f"Web search completed. Retrieved {len(results_json)} results",
                done=True,
            )

            return json.dumps(results_json, ensure_ascii=False)

        except requests.exceptions.RequestException as e:
            error_msg = f"Error during web search: {str(e)}"
            await emitter.emit_status(
                status="error",
                description=error_msg,
                done=True,
            )
            return json.dumps({"error": error_msg})
        except json.JSONDecodeError as e:
            error_msg = f"Error parsing API response: {str(e)}"
            await emitter.emit_status(
                status="error",
                description=error_msg,
                done=True,
            )
            return json.dumps({"error": error_msg})
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            await emitter.emit_status(
                status="error",
                description=error_msg,
                done=True,
            )
            return json.dumps({"error": error_msg})

    async def fetch_website(
        self,
        url: str,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Fetch a single web page by URL using Ollama's Web Fetch API.
        :params url: The URL to fetch.
        :return: The webpage content in JSON format.
        """
        emitter = EventEmitter(__event_emitter__)

        await emitter.emit_status(f"Fetching content from URL: {url}")

        if not self.valves.OLLAMA_API_KEY:
            error_msg = "Ollama API key is required. Please set the OLLAMA_API_KEY in the valves."
            await emitter.emit_status(
                status="error",
                description=error_msg,
                done=True,
            )
            return json.dumps({"error": error_msg})

        if not url.startswith(('http://', 'https://')):
            url = f"https://{url}"

        try:
            await emitter.emit_status("Sending request to Ollama web fetch API")

            payload = {"url": url}

            response = await asyncio.to_thread(
                requests.post,
                f"{self.base_url}/api/web_fetch",
                headers=self._get_headers(),
                json=payload,
                timeout=self.valves.TIMEOUT
            )
            response.raise_for_status()

            data = await asyncio.to_thread(response.json)
            await emitter.emit_status("Website content retrieved successfully")

            result_json = {
                "title": data.get("title", "No title found"),
                "url": url,
                "content": data.get("content", ""),
                "links": data.get("links", [])
            }

            # Send citation if enabled
            if self.valves.CITATION_LINKS and __event_emitter__:
                await __event_emitter__({
                    "type": "citation",
                    "data": {
                        "document": [data.get("content", "")],
                        "metadata": [{"source": url}],
                        "source": {"name": data.get("title", "No title found")},
                    },
                })

            await emitter.emit_status(
                status="complete",
                description="Website content retrieved and processed successfully",
                done=True,
            )

            return json.dumps([result_json], ensure_ascii=False)

        except requests.exceptions.RequestException as e:
            error_msg = f"Error fetching website content: {str(e)}"
            results_json = [
                {
                    "url": url,
                    "content": f"Failed to retrieve the page. Error: {str(e)}",
                }
            ]
            await emitter.emit_status(
                status="error",
                description=error_msg,
                done=True,
            )
            return json.dumps(results_json, ensure_ascii=False)
        except json.JSONDecodeError as e:
            error_msg = f"Error parsing API response: {str(e)}"
            await emitter.emit_status(
                status="error",
                description=error_msg,
                done=True,
            )
            return json.dumps({"error": error_msg})
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            await emitter.emit_status(
                status="error",
                description=error_msg,
                done=True,
            )
            return json.dumps({"error": error_msg})