"""
title: Enhanced Web Scrape
author: ekatiyar and christian-taillon
author_url: https://github.com/ekatiyar and https://github.com/christian-taillon/
git_url: https://github.com/christian-taillon/open-webui-pipelines
description: An improved-improved web scraping tool that extracts text content using Jina Reader, now with better filtering, user-configuration, UI feedback using emitters, retries, and batch scraping.
original_author: Pyotr Growpotkin
original_author_url: https://github.com/christ-offer/
original_git_url: https://github.com/christ-offer/open-webui-tools
funding_url: https://github.com/open-webui
version: 0.1.2
license: MIT
"""

import asyncio
import random
import re
import urllib.parse
from typing import Callable, Any, List, Dict, Optional

import requests
from pydantic import BaseModel, Field


def extract_title(text: str) -> Optional[str]:
    match = re.search(r"Title: (.*)\n", text)
    return match.group(1).strip() if match else None


def clean_urls(text: str) -> str:
    return re.sub(r"\((http[^)]+)\)", "", text)


class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def progress_update(self, description: str):
        await self.emit(description)

    async def error_update(self, description: str):
        await self.emit(description, "error", True)

    async def success_update(self, description: str):
        await self.emit(description, "success", True)

    async def emit(
        self,
        description: str = "Unknown State",
        status: str = "in_progress",
        done: bool = False,
    ):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )


class Tools:
    class Valves(BaseModel):
        DISABLE_CACHING: bool = Field(
            default=False, description="Bypass Jina Cache when scraping"
        )
        GLOBAL_JINA_API_KEY: str = Field(
            default="",
            description="(Optional) Jina API key. Allows a higher rate limit when scraping. Used when a User-specific API key is not available.",
        )
        CITATION: bool = Field(default=True, description="True or false for citation")

    class UserValves(BaseModel):
        CLEAN_CONTENT: bool = Field(
            default=True,
            description="Remove links and image urls from scraped content. This reduces the number of tokens.",
        )
        JINA_API_KEY: str = Field(
            default="",
            description="(Optional) Jina API key. Allows a higher rate limit when scraping.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.citation = self.valves.CITATION

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _normalize_url(self, u: str) -> str:
        return urllib.parse.urldefrag((u or "").strip())[0]

    def _concurrency(self, __user__: Dict) -> int:
        key_present = False
        try:
            key_present = bool(
                __user__.get("valves", self.UserValves()).JINA_API_KEY
            )  # pydantic model
        except AttributeError:
            # If valves is a pydantic model already
            key_present = bool(
                getattr(__user__.get("valves", self.UserValves()), "JINA_API_KEY", "")
            )
        key_present = key_present or bool(self.valves.GLOBAL_JINA_API_KEY)
        return 5 if key_present else 2

    async def _http_get(
        self, url: str, headers: Dict[str, str], timeout=(5, 30), attempts: int = 3
    ) -> requests.Response:
        last_exc: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                resp: requests.Response = await asyncio.to_thread(
                    requests.get, url, headers=headers, timeout=timeout
                )
                if resp.status_code in (429, 503):
                    # Respect Retry-After if present
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = 0.5 * (2 ** (attempt - 1)) + random.random() * 0.2
                    else:
                        delay = 0.5 * (2 ** (attempt - 1)) + random.random() * 0.2
                    await asyncio.sleep(delay)
                    last_exc = requests.HTTPError(f"HTTP {resp.status_code} for {url}")
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                last_exc = e
                if attempt == attempts:
                    break
                delay = 0.5 * (2 ** (attempt - 1)) + random.random() * 0.2
                await asyncio.sleep(delay)
        raise last_exc  # type: ignore

    def _build_headers(self, __user__: Dict) -> Dict[str, str]:
        headers = {
            "X-No-Cache": "true" if self.valves.DISABLE_CACHING else "false",
            "X-With-Generated-Alt": "true",
        }

        user_valves = __user__.get("valves", self.UserValves())
        api_key = ""
        try:
            api_key = user_valves.JINA_API_KEY
        except AttributeError:
            api_key = getattr(user_valves, "JINA_API_KEY", "")

        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        elif self.valves.GLOBAL_JINA_API_KEY:
            headers["Authorization"] = f"Bearer {self.valves.GLOBAL_JINA_API_KEY}"
        return headers

    async def _scrape_one(
        self, url: str, __event_emitter__: Callable[[dict], Any], __user__: Dict
    ) -> Dict[str, Optional[str]]:
        emitter = EventEmitter(__event_emitter__)

        if "valves" not in __user__ or not isinstance(
            __user__["valves"], self.UserValves
        ):
            __user__["valves"] = self.UserValves.parse_obj(
                getattr(__user__.get("valves", {}), "dict", lambda: {})()
            )

        url = self._normalize_url(url)
        await emitter.progress_update(f"Scraping {url}")

        jina_url = f"https://r.jina.ai/{url}"
        headers = self._build_headers(__user__)

        try:
            response = await self._http_get(jina_url, headers=headers)
            content = response.text

            should_clean = __user__["valves"].CLEAN_CONTENT
            if should_clean:
                await emitter.progress_update("Received content, cleaning up ...")
                content = clean_urls(content)

            title = extract_title(content)
            await emitter.success_update(
                f"Successfully Scraped {title if title else url}"
            )

            return {
                "url": url,
                "title": title or None,
                "content": content,
                "error": None,
            }

        except requests.RequestException as e:
            error_message = f"Error scraping web page: {str(e)}"
            await emitter.error_update(error_message)
            return {
                "url": url,
                "title": None,
                "content": None,
                "error": error_message,
            }

    # ----------------------------
    # Public Tool methods
    # ----------------------------
    async def web_scrape(
        self,
        url: str,
        __event_emitter__: Callable[[dict], Any] = None,
        __user__: dict = {},
    ) -> str:
        """
        Scrape and process a web page using r.jina.ai
        """
        result = await self._scrape_one(url, __event_emitter__, __user__)
        return (
            result["content"]
            if result["content"] is not None
            else (result["error"] or "Unknown error")
        )

    async def web_scrape_many(
        self,
        urls: List[str],
        __event_emitter__: Callable[[dict], Any] = None,
        __user__: dict = {},
    ) -> List[Dict[str, Optional[str]]]:
        """
        Scrape multiple web pages. Emits per-URL status updates and a final summary.
        Returns a list of results in the same order as input.
        """
        emitter = EventEmitter(__event_emitter__)

        if "valves" not in __user__ or not isinstance(
            __user__["valves"], self.UserValves
        ):
            __user__["valves"] = self.UserValves.parse_obj(
                getattr(__user__.get("valves", {}), "dict", lambda: {})()
            )

        limit = self._concurrency(__user__)
        sem = asyncio.Semaphore(limit)

        results: List[Optional[Dict[str, Optional[str]]]] = [None] * len(urls)

        async def run_one(i: int, raw_url: str):
            async with sem:
                res = await self._scrape_one(raw_url, __event_emitter__, __user__)
                results[i] = res

        await asyncio.gather(*(run_one(i, u) for i, u in enumerate(urls)))
        await emitter.success_update(f"Finished scraping {len(urls)} pages")

        return [r for r in results if r is not None]


if __name__ == "__main__":
    # Example usage (manual run):
    # asyncio.run(Tools().web_scrape_many(["https://example.com", "https://httpbin.org/html"]))
    pass

