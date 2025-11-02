"""
title: Person/Entity Research Tool with Jina
author: Christian Taillon
author_url: https://github.com/christian-taillon/
description: >
    Specialized tool for researching individuals or organizations. Automatically searches for various aspects like professional background, social media, publications, etc. Uses Jina's Reader and Search APIs for comprehensive information gathering.
requirements: requests, pydantic
version: 1.0.0
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


def clean_for_search(text: str) -> str:
    """Clean text for use in search queries."""
    # Remove problematic characters
    cleaned = text.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
    cleaned = cleaned.replace('{', '').replace('}', '').replace('"', '').replace("'", '')
    # Limit length
    if len(cleaned) > 150:
        cleaned = cleaned[:150]
    return cleaned.strip()


async def emit_status(event_emitter: Optional[Callable[[dict], Any]], description: str, status: str = "in_progress", done: bool = False):
    """Emit a status event."""
    if event_emitter:
        await event_emitter({
            "type": "status",
            "data": {"status": status, "description": description, "done": done},
        })


async def emit_citation(event_emitter: Optional[Callable[[dict], Any]], title: str, url: str, content: str, metadata: Optional[Dict] = None):
    """Emit a citation event."""
    if event_emitter:
        citation_data = {
            "document": [content],
            "metadata": [{"source": url, **(metadata or {})}],
            "source": {"name": title, "url": url},
        }
        await event_emitter({"type": "citation", "data": citation_data})


class Tools:
    class Valves(BaseModel):
        """Configuration for Person/Entity Research Tool."""
        JINA_GLOBAL_API_KEY: str = Field(
            default="",
            description="Global Jina API key for Reader/Search APIs.",
        )
        SEARCH_ASPECTS: List[str] = Field(
            default=[
                "professional background career",
                "social media profiles LinkedIn Twitter",
                "publications articles authored",
                "GitHub open source projects",
                "education university degrees",
                "news mentions press releases",
                "contact information email",
                "company affiliations organizations"
            ],
            description="Aspects to search for each person/entity"
        )
        MAX_RESULTS_PER_ASPECT: int = Field(
            default=2,
            description="Maximum results to keep per aspect"
        )
        CITATION_LINKS: bool = Field(
            default=True,
            description="Emit citations for sources"
        )

    class UserValves(BaseModel):
        """User settings."""
        JINA_API_KEY: str = Field(
            default="",
            description="Personal Jina API key.",
        )
        ADDITIONAL_ASPECTS: str = Field(
            default="",
            description="Additional search aspects (comma-separated)",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.citation = False

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

    async def _search_jina(
        self,
        query: str,
        jina_headers: Dict[str, str],
        __event_emitter__: Optional[Callable[[dict], Any]]
    ) -> str:
        """Search using Jina Search API and return the content."""
        clean_query = clean_for_search(query)
        search_query = urllib.parse.quote(clean_query, safe='')
        search_url = f"https://s.jina.ai/{search_query}"
        
        await emit_status(__event_emitter__, f"Searching: {clean_query[:80]}...")
        
        try:
            response = await asyncio.to_thread(
                requests.get,
                search_url,
                headers=jina_headers,
                timeout=30
            )
            response.raise_for_status()
            return response.text
        except Exception as e:
            await emit_status(__event_emitter__, f"Search error: {e}", status="error")
            return ""

    async def _extract_urls_from_search(self, content: str) -> List[str]:
        """Extract URLs from Jina search results."""
        urls = []
        
        # Extract URLs from search results
        url_pattern = r'URL Source: (https?://[^\s]+)'
        found_urls = re.findall(url_pattern, content)
        urls.extend(found_urls)
        
        # Also extract from markdown links
        md_link_pattern = r'\[([^\]]+)\]\((https?://[^\)]+)\)'
        md_urls = [url for _, url in re.findall(md_link_pattern, content)]
        urls.extend(md_urls)
        
        # Remove duplicates and limit
        unique_urls = list(dict.fromkeys(urls))
        return unique_urls[:self.valves.MAX_RESULTS_PER_ASPECT]

    async def _scrape_url(
        self,
        url: str,
        jina_headers: Dict[str, str],
        __event_emitter__: Optional[Callable[[dict], Any]]
    ) -> Optional[Dict[str, str]]:
        """Scrape a URL using Jina Reader."""
        jina_url = f"https://r.jina.ai/{url}"
        
        try:
            response = await asyncio.to_thread(
                requests.get,
                jina_url,
                headers=jina_headers,
                timeout=30
            )
            response.raise_for_status()
            
            content = response.text
            # Extract title
            title_match = re.search(r"Title: (.*)\n", content)
            title = title_match.group(1).strip() if title_match else url
            
            return {
                "url": url,
                "title": title,
                "content": content
            }
        except Exception as e:
            await emit_status(__event_emitter__, f"Scraping error for {url}: {e}")
            return None

    async def person_entity_research(
        self,
        entity_name: str,
        additional_context: str = "",
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None
    ) -> str:
        """
        Research a person or entity comprehensively.
        
        Args:
            entity_name: Name of the person or entity to research
            additional_context: Additional context (location, company, etc.)
            __user__: User context
            __event_emitter__: Event handler
            
        Returns:
            Comprehensive research report
        """
        jina_headers = self._build_headers(__user__)
        
        # Check API key
        if not jina_headers.get("Authorization"):
            await emit_status(__event_emitter__, "ERROR: No Jina API key provided", status="error")
            return "Cannot perform research without Jina API key."
        
        await emit_status(__event_emitter__, f"Starting comprehensive research on: {entity_name}")
        
        # Get user-specific aspects
        try:
            user_valves = self.UserValves.model_validate(__user__.get("valves", {}))
            additional_aspects = [a.strip() for a in user_valves.ADDITIONAL_ASPECTS.split(",") if a.strip()]
        except:
            additional_aspects = []
        
        # Combine all search aspects
        all_aspects = self.valves.SEARCH_ASPECTS + additional_aspects
        
        # Build base query
        base_query = f"{entity_name} {additional_context}".strip()
        
        all_findings = []
        aspect_results = {}
        
        # Search for each aspect
        for i, aspect in enumerate(all_aspects, 1):
            await emit_status(__event_emitter__, f"Researching ({i}/{len(all_aspects)}): {aspect}")
            
            # Create targeted search query
            search_query = f"{base_query} {aspect}"
            
            # Search using Jina
            search_content = await self._search_jina(search_query, jina_headers, __event_emitter__)
            
            if search_content:
                # Extract URLs from search results
                urls = await self._extract_urls_from_search(search_content)
                
                # Scrape each URL
                aspect_findings = []
                for url in urls:
                    result = await self._scrape_url(url, jina_headers, __event_emitter__)
                    if result:
                        result["aspect"] = aspect
                        aspect_findings.append(result)
                        all_findings.append(result)
                        
                        # Emit citation
                        if self.valves.CITATION_LINKS:
                            await emit_citation(
                                __event_emitter__,
                                result["title"],
                                result["url"],
                                result["content"],
                                metadata={"aspect": aspect}
                            )
                
                if aspect_findings:
                    aspect_results[aspect] = aspect_findings
                    await emit_status(__event_emitter__, f"Found {len(aspect_findings)} sources for: {aspect}")
                else:
                    # If no URLs found, use the search content itself
                    if "Title:" in search_content:
                        all_findings.append({
                            "title": f"Search results for {aspect}",
                            "content": search_content,
                            "aspect": aspect,
                            "url": "Jina Search"
                        })
        
        # Generate report
        await emit_status(__event_emitter__, "Generating comprehensive report...", status="complete", done=True)
        
        if not all_findings:
            return f"No information found for {entity_name}"
        
        # Build structured report
        report = []
        report.append(f"# Research Report: {entity_name}")
        if additional_context:
            report.append(f"**Context:** {additional_context}")
        report.append(f"\n**Total Sources Found:** {len(all_findings)}")
        report.append(f"**Aspects Researched:** {len(aspect_results)}/{len(all_aspects)}")
        report.append("\n---\n")
        
        # Organize by aspect
        report.append("## Research Findings by Category\n")
        
        for aspect, findings in aspect_results.items():
            report.append(f"### {aspect.title()}\n")
            for finding in findings:
                report.append(f"**Source:** {finding['title']}")
                report.append(f"**URL:** {finding['url']}")
                report.append("")
                
                # Truncate content for report (full content in citations)
                content = finding['content']
                if len(content) > 1500:
                    content = content[:1500] + "..."
                report.append(content)
                report.append("\n---\n")
        
        # Add summary of aspects with no findings
        missing_aspects = [a for a in all_aspects if a not in aspect_results]
        if missing_aspects:
            report.append("## Aspects with No Results Found\n")
            for aspect in missing_aspects:
                report.append(f"- {aspect}")
        
        return "\n".join(report)


# Test
if __name__ == '__main__':
    async def test():
        tools = Tools()
        tools.valves.JINA_GLOBAL_API_KEY = os.getenv("JINA_API_KEY", "")
        
        async def event_handler(event):
            if event["type"] == "status":
                print(f"[STATUS] {event['data']['description']}")
        
        result = await tools.person_entity_research(
            entity_name="Christian Taillon",
            additional_context="Phoenix Arizona cybersecurity GCU",
            __event_emitter__=event_handler
        )
        print("\n=== REPORT ===")
        print(result)
    
    asyncio.run(test())