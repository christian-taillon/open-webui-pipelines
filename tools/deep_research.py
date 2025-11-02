"""
title: Deep Research Tool with Jina Reranker
author: Christian Taillon
author_url: https://github.com/christian-taillon/
git_url: https://github.com/open-webui/open-webui/tree/main/src/lib/tools
description: >
    Performs deep research by scraping a list of URLs using Jina Reader API, then uses Jina's advanced reranker models to filter, rank, and select the top K most relevant pages based on a research topic. This ensures the final output is concise and highly relevant, optimizing the context provided to the main LLM.
requirements: requests, pydantic
version: 5.1.0
license: MIT
"""

import asyncio
import re
import urllib.parse
from typing import Any, Callable, List, Dict, Optional
import json
import os

import requests
from pydantic import BaseModel, Field

# Helper functions
def extract_title(text: str) -> Optional[str]:
    match = re.search(r"Title: (.*)\n", text)
    return match.group(1).strip() if match else None

def clean_urls(text: str) -> str:
    return re.sub(r"\((http[^)]+)\"", "", text)

def chunk_text(text: str, max_size: int) -> List[str]:
    """Split text into chunks while trying to preserve sentence boundaries."""
    if len(text) <= max_size:
        return [text]
    
    # Try to split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_size:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

class EventEmitter:
    """A simple class for emitting status and citation events."""
    def __init__(self, event_emitter: Optional[Callable[[dict], Any]]):
        self.event_emitter = event_emitter

    async def emit_status(self, description: str, status: str = "in_progress", done: bool = False):
        if self.event_emitter:
            await self.event_emitter({
                "type": "status",
                "data": {"status": status, "description": description, "done": done},
            })

    async def emit_citation(self, title: str, url: str, content: str, metadata: Optional[Dict] = None):
        if self.event_emitter:
            citation_data = {
                "document": [content],
                "metadata": [{"source": url, **(metadata or {})}],
                "source": {"name": title, "url": url},
            }
            await self.event_emitter({"type": "citation", "data": citation_data})

# Main Tool Class for OpenWebUI
class Tools:
    class Valves(BaseModel):
        """Configuration settings for the Deep Research Tool. These can be adjusted by the user or system administrator to control the tool's behavior.
        """
        RELEVANCE_THRESHOLD: float = Field(
            default=0.4,
            description="The minimum reranker score (0.0-1.0) for a page to be considered relevant. Pages below this threshold are discarded."
        )
        TOP_K_RESULTS: int = Field(
            default=5,
            description="The maximum number of the most relevant pages to return. This helps to keep the final output concise."
        )
        JINA_GLOBAL_API_KEY: str = Field(
            default="",
            description="A global Jina API key for Reader/Reranker. Used if a user-specific key is not provided.",
        )
        RERANKER_MODEL: str = Field(
            default="jina-reranker-v2-base-multilingual",
            description="Jina reranker model to use. Options: jina-reranker-v2-base-multilingual, jina-colbert-v2, jina-reranker-m0"
        )
        MAX_CHUNK_SIZE: int = Field(
            default=8000,
            description="Maximum characters per chunk when splitting long documents for reranking."
        )
        CITATION_LINKS: bool = Field(
            default=True, description="If True, a citation event is sent for each successfully processed and filtered URL."
        )

    class UserValves(BaseModel):
        """User-specific settings that can override the global tool configurations.
        """
        CLEAN_CONTENT: bool = Field(
            default=True,
            description="If enabled, removes links and image URLs from scraped content to reduce token count and improve clarity.",
        )
        JINA_API_KEY: str = Field(
            default="",
            description="A personal Jina API key to ensure the highest possible scraping rate limits.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.citation = False # We handle our own citations

    def _normalize_url(self, u: str) -> str:
        return urllib.parse.urldefrag((u or "").strip())[0]

    def _get_concurrency(self, __user__: Dict) -> int:
        try:
            user_valves = self.UserValves.model_validate(__user__.get("valves", {}))
            key_present = bool(user_valves.JINA_API_KEY) or bool(self.valves.JINA_GLOBAL_API_KEY)
            return 5 if key_present else 2
        except Exception:
            return 2

    def _build_headers(self, __user__: Dict) -> Dict[str, str]:
        headers = {}
        try:
            user_valves = self.UserValves.model_validate(__user__.get("valves", {}))
            api_key = user_valves.JINA_API_KEY
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            elif self.valves.JINA_GLOBAL_API_KEY:
                headers["Authorization"] = f"Bearer {self.valves.JINA_GLOBAL_API_KEY}"
        except Exception:
            if self.valves.JINA_GLOBAL_API_KEY:
                headers["Authorization"] = f"Bearer {self.valves.JINA_GLOBAL_API_KEY}"
        return headers

    async def deep_research(
        self,
        urls: List[str],
        research_topic: str,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None
    ) -> str:
        """
        Performs deep research on a list of URLs, filtering them for relevance to a specific topic.

        This tool scrapes the content of each URL using Jina Reader API, then uses Jina's Reranker API to score and rank the relevance of each page to the research topic. The tool intelligently chunks long documents to stay within token limits while preserving context. It returns a concatenated string of the top K most relevant pages, ensuring the output is both concise and highly relevant.

        Args:
            urls: A list of URLs to scrape, process, and filter.
            research_topic: The central topic of the research. This is used to calculate the relevance of each scraped page.
            __user__: User information, injected by OpenWebUI.
            __event_emitter__: An optional callback to emit status updates and citations.

        Returns:
            A formatted string containing the content from the top K most relevant pages, or an error message if no content could be processed.
        """
        emitter = EventEmitter(__event_emitter__)
        jina_headers = self._build_headers(__user__)
        concurrency_limit = self._get_concurrency(__user__)
        semaphore = asyncio.Semaphore(concurrency_limit)
        valves = self.valves

        try:
            user_valves = self.UserValves.model_validate(__user__.get("valves", {}))
            should_clean = user_valves.CLEAN_CONTENT
        except Exception:
            should_clean = True


        async def _scrape_one_url_nested(url: str) -> Optional[Dict[str, Any]]:
            normalized_url = self._normalize_url(url)
            await emitter.emit_status(f"Scraping: {normalized_url}")
            jina_url = f"https://r.jina.ai/{normalized_url}"

            try:
                response = await asyncio.to_thread(requests.get, jina_url, headers=jina_headers, timeout=120)
                response.raise_for_status()
                content = response.text

                if should_clean:
                    content = clean_urls(content)

                title = extract_title(content) or normalized_url
                return {"url": normalized_url, "title": title, "content": content}
            except requests.RequestException as e:
                error_str = str(e)
                if "400" in error_str and "Bad Request" in error_str:
                    await emitter.emit_status(f"Skipping invalid URL (Jina 400 Error): {normalized_url}")
                else:
                    await emitter.emit_status(f"Error scraping {normalized_url}: {e}", status="error")
                return None

        async def run_scrape_with_semaphore(url: str):
            async with semaphore:
                return await _scrape_one_url_nested(url)

        scrape_tasks = [asyncio.create_task(run_scrape_with_semaphore(url)) for url in urls]
        results = await asyncio.gather(*scrape_tasks, return_exceptions=True)
        
        successful_scrapes = []
        for r in results:
            if isinstance(r, Exception):
                await emitter.emit_status(f"An unexpected error occurred in a scraping task: {r}", status="error")
            elif r:
                successful_scrapes.append(r)

        if not successful_scrapes:
            return "No content could be scraped from the provided URLs."

        # Always attempt reranking using Jina
        await emitter.emit_status("Starting relevance filtering with Jina Reranker...")

        # Check if API key is available
        has_api_key = bool(jina_headers.get("Authorization"))

        if has_api_key:
            await emitter.emit_status(f"Using Jina Reranker ({valves.RERANKER_MODEL}) for relevance scoring...")

            # Prepare documents for reranking
            documents_to_rerank: List[str] = []
            index_to_original: List[int] = []  # Map rerank index -> original scrape index

            for idx, r in enumerate(successful_scrapes):
                content = r["content"]
                # If content is too long, chunk it
                if len(content) > valves.MAX_CHUNK_SIZE:
                    chunks = chunk_text(content, valves.MAX_CHUNK_SIZE)
                    for _chunk_idx, chunk in enumerate(chunks):
                        documents_to_rerank.append(chunk)
                        index_to_original.append(idx)
                else:
                    documents_to_rerank.append(content)
                    index_to_original.append(idx)

            # Call Jina Reranker API
            rerank_url = "https://api.jina.ai/v1/rerank"
            rerank_payload = {
                "model": valves.RERANKER_MODEL,
                "query": research_topic,
                "documents": documents_to_rerank,
                "top_n": min(len(documents_to_rerank), valves.TOP_K_RESULTS * 2),  # get a bit more to account for chunks
                "return_documents": False
            }

            try:
                rerank_response = await asyncio.to_thread(
                    requests.post,
                    rerank_url,
                    headers=jina_headers,
                    json=rerank_payload,
                    timeout=60
                )
                rerank_response.raise_for_status()
                rerank_data = rerank_response.json()

                # Process reranking results: keep highest score per original document
                doc_scores: Dict[int, float] = {}
                for result in rerank_data.get("results", []):
                    idx_in_batch = result.get("index")
                    if idx_in_batch is None or idx_in_batch >= len(index_to_original):
                        continue
                    original_idx = index_to_original[idx_in_batch]
                    score = float(result.get("relevance_score", 0.0))
                    if original_idx not in doc_scores or score > doc_scores[original_idx]:
                        doc_scores[original_idx] = score

                # Apply threshold and sort
                ranked_results: List[Dict[str, Any]] = []
                for idx, score in doc_scores.items():
                    if score >= valves.RELEVANCE_THRESHOLD:
                        r = successful_scrapes[idx].copy()
                        r["reranker_score"] = score
                        ranked_results.append(r)
                        await emitter.emit_status(f"URL: {r['url']} | Reranker Score: {score:.4f}")

                ranked_results.sort(key=lambda x: x.get("reranker_score", 0.0), reverse=True)
                final_results = ranked_results[:valves.TOP_K_RESULTS]

                await emitter.emit_status(f"Reranking complete. Kept {len(final_results)} of {len(successful_scrapes)} pages.")
                successful_scrapes = final_results

            except Exception as e:
                await emitter.emit_status(f"Reranker API error: {e}. Skipping relevance filter.", status="error")
                # Best-effort fallback: keep first TOP_K_RESULTS without scores
                successful_scrapes = successful_scrapes[:valves.TOP_K_RESULTS]
        else:
            await emitter.emit_status("No Jina API key provided; cannot call Reranker API. Skipping relevance filter.", status="error")
            # Best-effort fallback: keep first TOP_K_RESULTS without scores
            successful_scrapes = successful_scrapes[:valves.TOP_K_RESULTS]

        for r in successful_scrapes:
            if valves.CITATION_LINKS:
                score = r.get('reranker_score')
                score_meta = f"{score:.4f}" if isinstance(score, float) else "N/A"
                await emitter.emit_citation(r["title"], r["url"], r["content"], metadata={"reranker_score": score_meta})

        await emitter.emit_status(f"Deep research complete. Processed {len(successful_scrapes)} pages.", status="complete", done=True)
        
        if not successful_scrapes:
            return "No relevant content found based on the filtering criteria."

        combined_content = [f"## {r['title']}\nURL: {r['url']}\n\n{r['content']}" for r in successful_scrapes]
        return "\n\n---\n\n".join(combined_content)

if __name__ == '__main__':
    async def my_event_handler(event: dict):
        if event["type"] == "status":
            print(f"EVENT: {event['data']['description']}")
        elif event["type"] == "citation":
            print(f"EVENT: Emitting citation for '{event['data']['source']['name']}'")
            print(f"  -> CITATION METADATA: {event['data']['metadata']}")

    async def main():
        print("--- Starting Deep Research Tool Test v5.1 ---")
        tools_instance = Tools()

        # For local testing, set your Jina API key if present
        tools_instance.valves.JINA_GLOBAL_API_KEY = os.getenv("JINA_API_KEY", "")
        if tools_instance.valves.JINA_GLOBAL_API_KEY:
            print("Using Jina API key from environment.")
        else:
            print("WARNING: No Jina API key set. Reranker will be skipped.")

        test_urls = [
            "https://blog.langchain.dev/open-deep-research/",
            "https://docs.openwebui.com/getting-started/env-variables/",
            "https://this-is-a-fake-url-that-will-404.com/some-page",
            "https://www.jina.ai/news/jina-reader-the-easiest-way-to-read-any-url/"
        ]
        
        try:
            findings = await tools_instance.deep_research(
                urls=test_urls,
                research_topic="Open Source AI and WebUI",
                __event_emitter__=my_event_handler
            )
            print("\n--- FINAL FINDINGS ---")
            print(findings)
        except Exception as e:
            print(f"\nAn error occurred during the test: {e}")

    asyncio.run(main())

