"""
title: Iterative Deep Research Tool with Jina
author: Christian Taillon
author_url: https://github.com/christian-taillon/
description: >
    An advanced research tool that performs iterative, multi-round searches until all aspects of a research question are thoroughly addressed. Uses Jina's Reader and Reranker APIs to ensure comprehensive, high-quality results. The tool analyzes gaps in knowledge and automatically performs follow-up searches.
requirements: requests, pydantic
version: 1.0.0
license: MIT
"""

import asyncio
import re
import urllib.parse
from typing import Any, Callable, List, Dict, Optional, Set, Tuple
import json
import os

import requests
from pydantic import BaseModel, Field


# Helper functions
def extract_title(text: str) -> Optional[str]:
    match = re.search(r"Title: (.*)\\n", text)
    return match.group(1).strip() if match else None


def clean_urls(text: str) -> str:
    return re.sub(r"\\((http[^)]+)\\)", "", text)


def chunk_text(text: str, max_size: int) -> List[str]:
    """Split text into chunks while trying to preserve sentence boundaries."""
    if len(text) <= max_size:
        return [text]
    
    sentences = re.split(r'(?<=[.!?])\\s+', text)
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


def extract_questions_from_topic(topic: str) -> List[str]:
    """Extract key questions that need to be answered from the research topic."""
    questions = []
    
    # Direct questions in the topic (ending with ?)
    direct_questions = re.findall(r'[^.!?]*\?', topic)
    questions.extend([q.strip() for q in direct_questions if q.strip()])
    
    # Check if this is a person/entity research request
    topic_lower = topic.lower()
    if any(phrase in topic_lower for phrase in ['information on', 'research on', 'find information about', 'who is']):
        # Extract what information is being requested
        if 'personal background' in topic_lower:
            questions.append("What is their personal background?")
        if 'professional' in topic_lower or 'work' in topic_lower or 'career' in topic_lower:
            questions.append("What is their professional background and current work?")
        if 'social media' in topic_lower:
            questions.append("What are their social media profiles?")
        if 'location' in topic_lower or 'where' in topic_lower:
            questions.append("Where are they located or have lived?")
        if 'news' in topic_lower or 'article' in topic_lower:
            questions.append("What news articles or publications mention them?")
        if 'written by' in topic_lower or 'authored' in topic_lower:
            questions.append("What have they written or authored?")
        if 'blog' in topic_lower:
            questions.append("What blog posts have they written?")
        if 'open source' in topic_lower or 'github' in topic_lower or 'project' in topic_lower:
            questions.append("What open source projects are they involved with?")
        if 'education' in topic_lower or 'college' in topic_lower or 'university' in topic_lower:
            questions.append("What is their educational background?")
        if 'contact' in topic_lower:
            questions.append("What is their contact information?")
        if 'family' in topic_lower:
            questions.append("What is known about their family?")
    
    # Look for comparison requests
    elif any(word in topic_lower for word in ['compare', 'versus', 'vs', 'difference between']):
        questions.append("What are the key differences?")
        questions.append("What are the similarities?")
        questions.append("Which is better and why?")
    
    # Look for "how" questions
    elif 'how to' in topic_lower or 'how do' in topic_lower:
        questions.append("What are the step-by-step instructions?")
        questions.append("What are the prerequisites?")
        questions.append("What are common pitfalls to avoid?")
    
    # Look for "best" or optimization queries
    elif any(word in topic_lower for word in ['best', 'top', 'optimal', 'recommended']):
        questions.append("What are the evaluation criteria?")
        questions.append("What are the top options?")
        questions.append("What are the pros and cons of each?")
    
    # Look for implementation or setup queries
    elif any(word in topic_lower for word in ['implement', 'setup', 'install', 'deploy']):
        questions.append("What are the requirements?")
        questions.append("What is the implementation process?")
        questions.append("How to verify it's working correctly?")
    
    # Default questions if none were found
    if not questions:
        questions = [
            "What is the overview of this topic?",
            "What are the key details?",
            "What are the important considerations?"
        ]
    
    return questions


def analyze_research_gaps(topic: str, current_findings: str, questions: List[str]) -> Tuple[List[str], List[str]]:
    """Analyze what questions remain unanswered and suggest follow-up searches."""
    unanswered = []
    follow_up_searches = []
    
    findings_lower = current_findings.lower()
    
    # Extract the main subject from the topic (usually the name or main entity)
    # This helps create better targeted searches
    main_subject = topic.split('.')[0].split('?')[0].strip()
    if len(main_subject) > 100:
        main_subject = main_subject[:100]
    
    for question in questions:
        question_keywords = set(re.findall(r'\b[a-z]+\b', question.lower()))
        question_keywords.discard('the')
        question_keywords.discard('what')
        question_keywords.discard('are')
        question_keywords.discard('is')
        question_keywords.discard('how')
        question_keywords.discard('their')
        question_keywords.discard('they')
        
        # Check if keywords from the question appear in findings
        keywords_found = sum(1 for kw in question_keywords if kw in findings_lower)
        coverage = keywords_found / len(question_keywords) if question_keywords else 0
        
        if coverage < 0.5:  # Less than 50% of keywords found
            unanswered.append(question)
            
            # Generate targeted search query
            # Use main subject + specific aspect from question
            key_terms = ' '.join(list(question_keywords)[:3])  # Limit keywords
            search_query = f"{main_subject} {key_terms}"
            follow_up_searches.append(search_query)
    
    return unanswered, follow_up_searches


def generate_search_urls(query: str, round_num: int) -> List[str]:
    """Generate search URLs using Jina Search API."""
    # Use Jina's search endpoint
    encoded_query = urllib.parse.quote(query)
    search_url = f"https://s.jina.ai/{encoded_query}"
    
    # For subsequent rounds, we might want to add modifiers
    if round_num > 1:
        # Add terms to get different results
        modifiers = ["detailed", "comprehensive", "technical", "practical", "examples"]
        modifier = modifiers[min(round_num - 2, len(modifiers) - 1)]
        modified_query = f"{query} {modifier}"
        encoded_modified = urllib.parse.quote(modified_query)
        search_url = f"https://s.jina.ai/{encoded_modified}"
    
    return [search_url]


# Helper functions for event emission
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
        """Configuration settings for the Iterative Deep Research Tool."""
        MAX_RESEARCH_ROUNDS: int = Field(
            default=5,
            description="Maximum number of research rounds to perform. Tool will stop earlier if all questions are answered."
        )
        URLS_PER_ROUND: int = Field(
            default=5,
            description="Number of URLs to analyze per research round."
        )
        RELEVANCE_THRESHOLD: float = Field(
            default=0.35,
            description="Minimum reranker score (0.0-1.0) for relevance. Lower for iterative research to gather more context."
        )
        TOP_K_PER_ROUND: int = Field(
            default=3,
            description="Maximum number of relevant pages to keep per round."
        )
        JINA_GLOBAL_API_KEY: str = Field(
            default="",
            description="Global Jina API key for Reader/Reranker/Search APIs.",
        )
        RERANKER_MODEL: str = Field(
            default="jina-reranker-v2-base-multilingual",
            description="Jina reranker model. Options: jina-reranker-v2-base-multilingual, jina-colbert-v2, jina-reranker-m0"
        )
        MAX_CHUNK_SIZE: int = Field(
            default=6000,
            description="Maximum characters per chunk when splitting documents."
        )
        ENABLE_SEARCH_API: bool = Field(
            default=True,
            description="Use Jina Search API to find relevant URLs automatically."
        )
        COMPLETENESS_THRESHOLD: float = Field(
            default=0.8,
            description="Fraction of questions that must be answered to consider research complete."
        )
        CITATION_LINKS: bool = Field(
            default=True,
            description="Emit citation events for sources."
        )

    class UserValves(BaseModel):
        """User-specific settings."""
        CLEAN_CONTENT: bool = Field(
            default=True,
            description="Remove links and image URLs from content.",
        )
        JINA_API_KEY: str = Field(
            default="",
            description="Personal Jina API key.",
        )
        VERBOSE_MODE: bool = Field(
            default=True,
            description="Show detailed progress including unanswered questions.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.citation = False  # We handle our own citations

    def _normalize_url(self, u: str) -> str:
        return urllib.parse.urldefrag((u or "").strip())[0]

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

    async def _search_for_urls(
        self, 
        query: str, 
        jina_headers: Dict[str, str],
        __event_emitter__: Optional[Callable[[dict], Any]]
    ) -> List[str]:
        """Search for relevant URLs using Jina Search API."""
        # Clean up the query to avoid encoding issues
        # Remove parentheses and other problematic characters
        clean_query = query.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
        # Limit query length to avoid overly long URLs
        if len(clean_query) > 200:
            clean_query = clean_query[:200]
        
        search_query = urllib.parse.quote(clean_query, safe='')
        search_url = f"https://s.jina.ai/{search_query}"
        
        await emit_status(__event_emitter__, f"Searching for: {clean_query[:100]}..." if len(clean_query) > 100 else f"Searching for: {clean_query}")
        
        try:
            response = await asyncio.to_thread(
                requests.get, 
                search_url, 
                headers=jina_headers,
                timeout=30
            )
            response.raise_for_status()
            
            # Parse URLs from search results
            content = response.text
            # Extract URLs from the search results
            # Jina Search returns results in markdown format with URLs
            url_pattern = r'URL Source: (https?://[^\\s]+)'
            found_urls = re.findall(url_pattern, content)
            
            # Also try to extract from markdown links
            md_link_pattern = r'\\[([^\\]]+)\\]\\((https?://[^\\)]+)\\)'
            md_urls = [url for _, url in re.findall(md_link_pattern, content)]
            
            all_urls = list(set(found_urls + md_urls))[:self.valves.URLS_PER_ROUND]
            
            if all_urls:
                await emit_status(__event_emitter__, f"Found {len(all_urls)} URLs from search")
            else:
                await emit_status(__event_emitter__, "No URLs found from search, using search result as content")
                
            return all_urls
            
        except Exception as e:
            await emit_status(__event_emitter__, f"Search error: {e}", status="error")
            return []

    async def _research_round(
        self,
        round_num: int,
        urls: List[str],
        research_topic: str,
        jina_headers: Dict[str, str],
        __event_emitter__: Optional[Callable[[dict], Any]],
        should_clean: bool
    ) -> List[Dict[str, Any]]:
        """Perform a single round of research on the given URLs."""
        
        await emit_status(__event_emitter__, f"=== Research Round {round_num} ===")
        await emit_status(__event_emitter__, f"Analyzing {len(urls)} URLs...")
        
        # Scrape URLs
        successful_scrapes = []
        for url in urls:
            normalized_url = self._normalize_url(url)
            if not normalized_url:
                continue
                
            await emit_status(__event_emitter__, f"Scraping: {normalized_url}")
            jina_url = f"https://r.jina.ai/{normalized_url}"
            
            try:
                response = await asyncio.to_thread(
                    requests.get, 
                    jina_url, 
                    headers=jina_headers, 
                    timeout=60
                )
                response.raise_for_status()
                content = response.text
                
                if should_clean:
                    content = clean_urls(content)
                
                title = extract_title(content) or normalized_url
                successful_scrapes.append({
                    "url": normalized_url,
                    "title": title,
                    "content": content,
                    "round": round_num
                })
                
            except Exception as e:
                await emit_status(__event_emitter__, f"Error scraping {normalized_url}: {e}")
                continue
        
        if not successful_scrapes:
            return []
        
        # Rerank for relevance
        await emit_status(__event_emitter__, f"Reranking {len(successful_scrapes)} pages for relevance...")
        
        documents_to_rerank = []
        index_to_original = []
        
        for idx, r in enumerate(successful_scrapes):
            content = r["content"]
            if len(content) > self.valves.MAX_CHUNK_SIZE:
                chunks = chunk_text(content, self.valves.MAX_CHUNK_SIZE)
                for chunk in chunks[:3]:  # Limit chunks per document in iterative mode
                    documents_to_rerank.append(chunk)
                    index_to_original.append(idx)
            else:
                documents_to_rerank.append(content)
                index_to_original.append(idx)
        
        if not documents_to_rerank:
            return successful_scrapes[:self.valves.TOP_K_PER_ROUND]
        
        # Call Jina Reranker
        rerank_url = "https://api.jina.ai/v1/rerank"
        rerank_payload = {
            "model": self.valves.RERANKER_MODEL,
            "query": research_topic,
            "documents": documents_to_rerank,
            "top_n": min(len(documents_to_rerank), self.valves.TOP_K_PER_ROUND * 2),
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
            
            # Process results
            doc_scores = {}
            for result in rerank_data.get("results", []):
                idx_in_batch = result.get("index")
                if idx_in_batch is None or idx_in_batch >= len(index_to_original):
                    continue
                original_idx = index_to_original[idx_in_batch]
                score = float(result.get("relevance_score", 0.0))
                if original_idx not in doc_scores or score > doc_scores[original_idx]:
                    doc_scores[original_idx] = score
            
            # Filter and sort
            relevant_results = []
            for idx, score in doc_scores.items():
                if score >= self.valves.RELEVANCE_THRESHOLD:
                    r = successful_scrapes[idx].copy()
                    r["reranker_score"] = score
                    relevant_results.append(r)
                    await emit_status(__event_emitter__, f"Relevant: {r['title'][:50]}... (score: {score:.3f})")
            
            relevant_results.sort(key=lambda x: x.get("reranker_score", 0.0), reverse=True)
            return relevant_results[:self.valves.TOP_K_PER_ROUND]
            
        except Exception as e:
            await emit_status(__event_emitter__, f"Reranker error: {e}, using all results")
            return successful_scrapes[:self.valves.TOP_K_PER_ROUND]

    async def iterative_deep_research(
        self,
        initial_urls: List[str],
        research_topic: str,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None
    ) -> str:
        """
        Performs iterative deep research until the research topic is thoroughly addressed.
        
        This tool:
        1. Analyzes the research topic to extract key questions
        2. Performs initial research on provided URLs
        3. Identifies gaps in the research
        4. Automatically searches for additional sources
        5. Continues researching until questions are answered or max rounds reached
        
        Args:
            initial_urls: Starting URLs to research (can be empty if ENABLE_SEARCH_API is True)
            research_topic: The comprehensive research question or topic
            __user__: User context from OpenWebUI
            __event_emitter__: Event handler for status updates
            
        Returns:
            Comprehensive research findings with citations
        """
        jina_headers = self._build_headers(__user__)
        valves = self.valves
        
        try:
            user_valves = self.UserValves.model_validate(__user__.get("valves", {}))
            should_clean = user_valves.CLEAN_CONTENT
            verbose = user_valves.VERBOSE_MODE
        except Exception:
            should_clean = True
            verbose = True
        
        # Check API key
        has_api_key = bool(jina_headers.get("Authorization"))
        if not has_api_key:
            await emit_status(__event_emitter__, "WARNING: No Jina API key provided. Limited functionality.", status="error")
            return "Cannot perform iterative research without Jina API key."
        
        # Extract questions from topic
        await emit_status(__event_emitter__, "Analyzing research requirements...")
        questions = extract_questions_from_topic(research_topic)
        
        if verbose:
            await emit_status(__event_emitter__, f"Identified {len(questions)} key questions to answer:")
            for i, q in enumerate(questions, 1):
                await emit_status(__event_emitter__, f"  {i}. {q}")
        
        # Initialize research tracking
        all_findings = []
        researched_urls = set()
        research_rounds = []
        
        # Start with initial URLs or search for them
        current_urls = initial_urls.copy() if initial_urls else []
        
        if not current_urls and valves.ENABLE_SEARCH_API:
            await emit_status(__event_emitter__, "No initial URLs provided, searching for relevant sources...")
            current_urls = await self._search_for_urls(research_topic, jina_headers, __event_emitter__)
        
        # Main research loop
        for round_num in range(1, valves.MAX_RESEARCH_ROUNDS + 1):
            if not current_urls:
                if valves.ENABLE_SEARCH_API:
                    await emit_status(__event_emitter__, f"Round {round_num}: Searching for more sources...")
                    # Generate search query based on unanswered questions
                    if round_num > 1 and all_findings:
                        combined_content = "\n".join([f["content"] for f in all_findings])
                        unanswered, search_queries = analyze_research_gaps(
                            research_topic, 
                            combined_content, 
                            questions
                        )
                        if search_queries:
                            query = search_queries[0]  # Use most important gap
                        else:
                            query = research_topic
                    else:
                        query = research_topic
                    
                    current_urls = await self._search_for_urls(query, jina_headers, __event_emitter__)
                else:
                    await emit_status(__event_emitter__, "No more URLs to research.")
                    break
            
            # Filter out already researched URLs
            new_urls = [url for url in current_urls if url not in researched_urls]
            if not new_urls:
                await emit_status(__event_emitter__, f"Round {round_num}: No new URLs to research.")
                if valves.ENABLE_SEARCH_API and round_num < valves.MAX_RESEARCH_ROUNDS:
                    # Try searching for more
                    combined_content = "\n".join([f["content"] for f in all_findings])
                    unanswered, search_queries = analyze_research_gaps(
                        research_topic, 
                        combined_content, 
                        questions
                    )
                    if search_queries:
                        await emit_status(__event_emitter__, f"Searching for information on: {search_queries[0][:100]}...")
                        current_urls = await self._search_for_urls(search_queries[0], jina_headers, __event_emitter__)
                        continue
                break
            
            # Perform research round
            round_findings = await self._research_round(
                round_num,
                new_urls[:valves.URLS_PER_ROUND],
                research_topic,
                jina_headers,
                __event_emitter__,
                should_clean
            )
            
            if round_findings:
                all_findings.extend(round_findings)
                researched_urls.update([f["url"] for f in round_findings])
                
                # Track round info
                round_info = {
                    "round_num": round_num,
                    "urls": [f["url"] for f in round_findings],
                    "content": "\n".join([f["content"] for f in round_findings])
                }
                research_rounds.append(round_info)
                
                # Emit citations for this round
                if valves.CITATION_LINKS:
                    for f in round_findings:
                        await emit_citation(
                            __event_emitter__,
                            f["title"],
                            f["url"],
                            f["content"],
                            metadata={
                                "round": round_num,
                                "reranker_score": f.get("reranker_score", "N/A")
                            }
                        )
            
            # Check if we've answered enough questions
            if round_num > 1:
                combined_content = "\n".join([f["content"] for f in all_findings])
                unanswered, _ = analyze_research_gaps(research_topic, combined_content, questions)
                
                answered_ratio = 1 - (len(unanswered) / len(questions) if questions else 0)
                
                if verbose:
                    await emit_status(
                        __event_emitter__,
                        f"Progress: {answered_ratio:.0%} of questions addressed ({len(questions) - len(unanswered)}/{len(questions)})"
                    )
                    if unanswered and round_num < valves.MAX_RESEARCH_ROUNDS:
                        await emit_status(__event_emitter__, "Still need information on:")
                        for q in unanswered[:3]:  # Show top 3 unanswered
                            await emit_status(__event_emitter__, f"  - {q}")
                
                if answered_ratio >= valves.COMPLETENESS_THRESHOLD:
                    await emit_status(
                        __event_emitter__,
                        f"Research complete! Addressed {answered_ratio:.0%} of identified questions after {round_num} rounds."
                    )
                    break
            
            # Prepare URLs for next round
            if valves.ENABLE_SEARCH_API and round_num < valves.MAX_RESEARCH_ROUNDS:
                combined_content = "\n".join([f["content"] for f in all_findings])
                unanswered, search_queries = analyze_research_gaps(
                    research_topic, 
                    combined_content, 
                    questions
                )
                
                if search_queries:
                    # Get new URLs for the next round based on gaps
                    current_urls = []
                    for query in search_queries[:2]:  # Search for top 2 gaps
                        urls = await self._search_for_urls(query, jina_headers, __event_emitter__)
                        current_urls.extend(urls)
                else:
                    current_urls = []
            else:
                current_urls = []
        
        # Generate final report
        await emit_status(__event_emitter__, "Generating comprehensive research report...", status="complete")
        
        if not all_findings:
            return "No relevant content found during research."
        
        # Organize findings by round for better structure
        report_sections = []
        
        # Executive Summary
        report_sections.append(f"# Research Report: {research_topic}\n")
        report_sections.append(f"**Research Rounds Completed:** {len(research_rounds)}")
        report_sections.append(f"**Total Sources Analyzed:** {len(all_findings)}")
        report_sections.append(f"**Questions Identified:** {len(questions)}\n")
        
        # Key Questions Section
        if questions:
            report_sections.append("## Key Questions Addressed\n")
            combined_content = "\n".join([f["content"] for f in all_findings])
            unanswered, _ = analyze_research_gaps(research_topic, combined_content, questions)
            
            for q in questions:
                status = "❌" if q in unanswered else "✅"
                report_sections.append(f"- {status} {q}")
            report_sections.append("")
        
        # Findings by relevance (not by round, for coherence)
        report_sections.append("## Research Findings\n")
        
        # Sort all findings by relevance score
        all_findings.sort(key=lambda x: x.get("reranker_score", 0), reverse=True)
        
        for finding in all_findings:
            score = finding.get("reranker_score", 0)
            round_num = finding.get("round", 1)
            report_sections.append(f"### {finding['title']}")
            report_sections.append(f"**Source:** {finding['url']}")
            report_sections.append(f"**Relevance Score:** {score:.3f} | **Round:** {round_num}")
            report_sections.append(f"\n{finding['content']}\n")
            report_sections.append("---\n")
        
        return "\n".join(report_sections)


# Test function
if __name__ == '__main__':
    async def my_event_handler(event: dict):
        if event["type"] == "status":
            print(f"[STATUS] {event['data']['description']}")
        elif event["type"] == "citation":
            print(f"[CITATION] {event['data']['source']['name']}")

    async def main():
        print("=== Iterative Deep Research Tool Test ===\n")
        tools = Tools()
        
        # Set API key from environment
        tools.valves.JINA_GLOBAL_API_KEY = os.getenv("JINA_API_KEY", "")
        if not tools.valves.JINA_GLOBAL_API_KEY:
            print("WARNING: No JINA_API_KEY in environment")
        
        # Test with a complex, multi-faceted research topic
        research_topic = """
        Compare OpenWebUI and LangChain for building AI applications. 
        What are their architectures? How do they handle tool integration? 
        Which is better for production use? What are the deployment options?
        Include code examples and best practices for each.
        """
        
        # Start with no URLs to test search capability
        result = await tools.iterative_deep_research(
            initial_urls=[],  # Will search automatically
            research_topic=research_topic,
            __event_emitter__=my_event_handler
        )
        
        print("\n=== FINAL REPORT ===")
        print(result)

    asyncio.run(main())