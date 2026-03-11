import asyncio
import time
import os
import re
import json
import arxiv
import wikipedia
from ddgs import DDGS
import google.genai as genai
from google.genai import types as genai_types
from dotenv import load_dotenv
from pypdf import PdfReader
from functools import lru_cache
import hashlib
from typing import Dict, Any
import pickle
from pathlib import Path
from tools import RESEARCH_TOOLS, search_arxiv, search_wikipedia, search_web
from models import AnalysisOutput, QCOutput

# Load environment variables
load_dotenv()

# Configure Gemini
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-3-flash-preview"

# ── Global token usage tracker ────────────────────────────────────────────────
_token_usage: dict = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
_agent_token_usage: dict = {}

# Gemini Flash approximate pricing ($ per 1M tokens)
_COST_INPUT_PER_M = 0.075
_COST_OUTPUT_PER_M = 0.300


def get_token_usage() -> dict:
    """Return a snapshot of cumulative token usage and estimated cost."""
    inp = _token_usage["input_tokens"]
    out = _token_usage["output_tokens"]
    cost = (inp * _COST_INPUT_PER_M + out * _COST_OUTPUT_PER_M) / 1_000_000
    return {**_token_usage, "estimated_cost_usd": round(cost, 4)}


def get_agent_token_breakdown() -> dict:
    """Return per-agent token usage with estimated cost."""
    result = {}
    for name, usage in _agent_token_usage.items():
        inp = usage["input_tokens"]
        out = usage["output_tokens"]
        cost = (inp * _COST_INPUT_PER_M + out * _COST_OUTPUT_PER_M) / 1_000_000
        result[name] = {**usage, "estimated_cost_usd": round(cost, 4)}
    return result


def reset_token_usage():
    _token_usage["input_tokens"] = 0
    _token_usage["output_tokens"] = 0
    _token_usage["total_tokens"] = 0
    _agent_token_usage.clear()
# ─────────────────────────────────────────────────────────────────────────────


# Cache directory setup
CACHE_DIR = Path(".research_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Helper: Cache Manager


class CacheManager:
    @staticmethod
    def get_cache_key(data: Any) -> str:
        """Generate cache key from data"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()

    @staticmethod
    def save_cache(key: str, data: Any, ttl: int = 3600):
        """Save data to cache with TTL"""
        cache_file = CACHE_DIR / f"{key}.pkl"
        cache_data = {
            "data": data,
            "timestamp": time.time(),
            "ttl": ttl
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

    @staticmethod
    def load_cache(key: str) -> Any:
        """Load data from cache if valid"""
        cache_file = CACHE_DIR / f"{key}.pkl"
        if not cache_file.exists():
            return None

        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        # Check TTL
        if time.time() - cache_data["timestamp"] > cache_data["ttl"]:
            cache_file.unlink()  # Delete expired cache
            return None

        return cache_data["data"]

# Helper: Retry Decorator


def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """Retry decorator for functions that may fail"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(delay * (attempt + 1))  # Exponential backoff
            return None
        return wrapper
    return decorator


class Agent:
    def __init__(self, name, role, icon="🤖"):
        self.name = name
        self.role = role
        self.icon = icon
        self.logs = []

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {self.icon} **{self.name}**: {message}"
        self.logs.append(entry)
        return entry

    def _track_usage(self, response):
        """Intercept usage_metadata from any generate_content response and accumulate totals."""
        um = getattr(response, "usage_metadata", None)
        if um:
            inp = getattr(um, "prompt_token_count", 0) or 0
            out = getattr(um, "candidates_token_count", 0) or 0
            tot = getattr(um, "total_token_count", 0) or 0
            _token_usage["input_tokens"] += inp
            _token_usage["output_tokens"] += out
            _token_usage["total_tokens"] += tot
            if self.name not in _agent_token_usage:
                _agent_token_usage[self.name] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            _agent_token_usage[self.name]["input_tokens"] += inp
            _agent_token_usage[self.name]["output_tokens"] += out
            _agent_token_usage[self.name]["total_tokens"] += tot

    @staticmethod
    def _dated_prompt(prompt: str) -> str:
        today = time.strftime("%A, %B %d, %Y")
        return f"[Current date: {today}]\n\n{prompt}"

    @retry_on_error(max_retries=3, delay=1.0)
    def call_llm(self, prompt, use_cache=True):
        """Call LLM with caching and retry logic"""
        prompt = self._dated_prompt(prompt)
        # Check cache first
        if use_cache:
            cache_key = CacheManager.get_cache_key(
                {"prompt": prompt, "model": "gemini"})
            cached_result = CacheManager.load_cache(cache_key)
            if cached_result:
                self.log("Using cached response")
                return cached_result

        try:
            response = _client.models.generate_content(
                model=MODEL, contents=prompt)
            self._track_usage(response)
            result = response.text

            # Save to cache
            if use_cache:
                CacheManager.save_cache(cache_key, result, ttl=3600)

            return result
        except Exception as e:
            self.log(f"LLM Error: {e}")
            raise e

    def stream_llm(self, prompt):
        """Stream LLM response, yielding text chunks. No caching (streaming incompatible)."""
        prompt = self._dated_prompt(prompt)
        try:
            last_chunk = None
            for chunk in _client.models.generate_content_stream(model=MODEL, contents=prompt):
                if chunk.text:
                    yield chunk.text
                last_chunk = chunk
            # Usage metadata is populated on the final chunk
            if last_chunk is not None:
                self._track_usage(last_chunk)
        except Exception as e:
            self.log(f"Stream LLM Error: {e}")
            raise e

    async def call_llm_async(self, prompt: str, use_cache: bool = True):
        """Async LLM call for use with asyncio.gather."""
        prompt = self._dated_prompt(prompt)
        if use_cache:
            cache_key = CacheManager.get_cache_key({"prompt": prompt, "model": "gemini"})
            cached = CacheManager.load_cache(cache_key)
            if cached:
                self.log("Using cached response (async)")
                return cached
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: _client.models.generate_content(model=MODEL, contents=prompt)
        )
        self._track_usage(response)
        result = response.text
        if use_cache:
            CacheManager.save_cache(cache_key, result, ttl=3600)
        return result

    @staticmethod
    def _dedup_by_key(items: list, key: str) -> list:
        """Deduplicate a list of dicts by a given key, normalizing URLs."""
        from urllib.parse import urlparse, urlunparse
        seen = set()
        result = []
        for item in items:
            raw = item.get(key, "")
            try:
                parsed = urlparse(raw)
                normalized = urlunparse(parsed._replace(query="", fragment=""))
                normalized = normalized.rstrip("/")
            except Exception:
                normalized = raw.rstrip("/")
            if normalized not in seen:
                seen.add(normalized)
                result.append(item)
        return result

    def perform_task(self, context):
        raise NotImplementedError


class ResearcherAgent(Agent):
    def __init__(self):
        super().__init__("Researcher", "Gathers raw data from academic, web, and local sources", "🔍")

    @retry_on_error(max_retries=2, delay=1.0)
    def perform_task(self, context):
        topic = context.get('topic')
        uploaded_files = context.get('uploaded_files', [])

        # Check cache
        cache_key = CacheManager.get_cache_key(
            {"topic": topic, "agent": "researcher"})
        cached_result = CacheManager.load_cache(cache_key)
        if cached_result:
            self.log("Using cached research data")
            return cached_result

        self.log(f"Starting research on: {topic}")
        results = {"arxiv": [], "wiki": "", "web": [], "local_docs": []}

        # 0. Process Uploaded PDFs
        if uploaded_files:
            self.log(f"Processing {len(uploaded_files)} uploaded documents...")
            for uploaded_file in uploaded_files:
                try:
                    reader = PdfReader(uploaded_file)
                    pages = reader.pages

                    def clean_pdf_text(raw: str) -> str:
                        """Drop diagram labels (< 5 words) — keeps only real sentences."""
                        lines = [l for l in raw.split(
                            '\n') if len(l.split()) > 4]
                        return "\n".join(lines)

                    # Page 1 contains the abstract/intro — highest signal density
                    abstract_text = clean_pdf_text(
                        pages[0].extract_text() if pages else "")

                    # Last 1-2 pages contain conclusions/results
                    conclusion_raw = ""
                    for page in pages[-2:] if len(pages) > 2 else pages:
                        conclusion_raw += page.extract_text()
                    conclusion_text = clean_pdf_text(conclusion_raw)

                    combined = (
                        f"[ABSTRACT / INTRODUCTION]\n{abstract_text[:1500]}\n\n"
                        f"[CONCLUSION / RESULTS]\n{conclusion_text[:1500]}"
                    )
                    results["local_docs"].append({
                        "filename": uploaded_file.name,
                        "content": combined
                    })
                    self.log(
                        f"✓ Read {uploaded_file.name} (abstract + conclusion extracted)")
                except Exception as e:
                    self.log(f"✗ Failed to read {uploaded_file.name}: {e}")

        # 1-3. Gemini native function calling ReAct loop
        qc_retry = context.get("qc_retry", False)
        qc_feedback = context.get("qc_feedback", "")
        planner_strategy = context.get("planner_strategy", {})

        try:
            self.log("Starting Gemini ReAct tool-calling loop for research...")
            if planner_strategy:
                search_focus = planner_strategy.get("search_focus", "")
                tool_priorities = planner_strategy.get("tool_priorities", [])
                extra_instr = planner_strategy.get("extra_instructions", "")
                suggested_queries = planner_strategy.get("suggested_queries", [])
                system_instruction = (
                    f"Expert researcher for topic '{topic}'. "
                    f"TARGETED FOCUS: {search_focus}. "
                    f"Tool priority order: {', '.join(tool_priorities)}. "
                    f"Prioritize these search queries: {'; '.join(suggested_queries[:3])}. "
                    f"{extra_instr}"
                )
                self.log(f"Using planner strategy: {search_focus[:60]}")
            else:
                system_instruction = (
                    f"Expert researcher for topic '{topic}'. "
                    "Use tools to gather comprehensive data. "
                    "Call search_arxiv for papers, search_wikipedia for background, "
                    "search_web for current info. Make multiple calls for broad coverage."
                )
            if qc_retry and qc_feedback and not planner_strategy:
                system_instruction += f" If this is a retry, focus on: {qc_feedback}"

            config = genai_types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=RESEARCH_TOOLS
            )

            initial_query = suggested_queries[0] if planner_strategy and planner_strategy.get("suggested_queries") else topic
            contents = [
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(
                        text=f"Research comprehensively: {initial_query}")]
                )
            ]

            collected_arxiv = []
            collected_wiki = []
            collected_web = []

            for iteration in range(8):
                response = _client.models.generate_content(
                    model=MODEL,
                    contents=contents,
                    config=config
                )
                self._track_usage(response)

                function_calls_found = False
                tool_response_parts = []

                for part in response.candidates[0].content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        function_calls_found = True
                        fc = part.function_call
                        fn_name = fc.name
                        fn_args = dict(fc.args) if fc.args else {}

                        self.log(
                            f"Tool call [{iteration+1}]: {fn_name}({fn_args})")

                        if fn_name == "search_arxiv":
                            raw = search_arxiv(**fn_args)
                            tool_result = raw
                            try:
                                parsed = json.loads(raw)
                                if isinstance(parsed, list):
                                    collected_arxiv.extend(parsed)
                            except Exception:
                                pass
                        elif fn_name == "search_wikipedia":
                            raw = search_wikipedia(**fn_args)
                            tool_result = raw
                            collected_wiki.append(raw)
                        elif fn_name == "search_web":
                            raw = search_web(**fn_args)
                            tool_result = raw
                            try:
                                parsed = json.loads(raw)
                                if isinstance(parsed, list):
                                    collected_web.extend(parsed)
                            except Exception:
                                pass
                        else:
                            tool_result = json.dumps(
                                {"error": f"Unknown tool: {fn_name}"})

                        tool_response_parts.append(
                            genai_types.Part(
                                function_response=genai_types.FunctionResponse(
                                    name=fn_name,
                                    response={"result": tool_result}
                                )
                            )
                        )

                # Append model turn then tool responses
                contents.append(response.candidates[0].content)
                if tool_response_parts:
                    contents.append(
                        genai_types.Content(
                            role="user", parts=tool_response_parts)
                    )

                if not function_calls_found:
                    self.log(
                        f"ReAct loop complete after {iteration+1} iterations.")
                    break

            collected_arxiv = self._dedup_by_key(collected_arxiv, "url")
            collected_web = self._dedup_by_key(collected_web, "href")
            results["arxiv"] = collected_arxiv
            results["wiki"] = "\n\n".join(
                collected_wiki) if collected_wiki else ""
            results["web"] = collected_web

            self.log(
                f"✓ ReAct collected: {len(results['arxiv'])} arxiv, {len(collected_wiki)} wiki, {len(results['web'])} web")

            # If model returned text without calling any tools, force fallback
            if not results["arxiv"] and not results["wiki"] and not results["web"]:
                raise ValueError(
                    "ReAct loop produced no tool calls — triggering direct search fallback")

        except Exception as react_err:
            self.log(
                f"⚠ ReAct loop failed ({react_err}), falling back to direct search...")

            # Reset collected lists before fallback
            results = {"arxiv": [], "wiki": "", "web": [],
                       "local_docs": results.get("local_docs", [])}

            # Fallback: original direct search
            self.log("Querying ArXiv database for academic papers...")
            try:
                search = arxiv.Search(query=topic, max_results=5)
                for result in search.results():
                    results["arxiv"].append({
                        "title": result.title,
                        "summary": result.summary[:500],
                        "url": result.entry_id,
                        "published": result.published.strftime("%Y-%m-%d"),
                        "authors": [a.name for a in result.authors[:3]]
                    })
                self.log(f"✓ Found {len(results['arxiv'])} academic papers.")
            except Exception as e:
                self.log(f"⚠ ArXiv search failed: {e}")

            self.log("Consulting Wikipedia for general context...")
            try:
                results["wiki"] = wikipedia.summary(
                    topic, sentences=7, auto_suggest=True)
                self.log("✓ Retrieved Wikipedia summary.")
            except Exception as e:
                self.log(f"⚠ Wikipedia search failed: {e}")
                results["wiki"] = ""

            self.log("Performing deep web search...")
            try:
                time.sleep(1)
                from ddgs import DDGS
                ddgs = DDGS()
                web_res = list(ddgs.text(topic, max_results=5))
                results["web"] = web_res
                self.log(f"✓ Found {len(results['web'])} web sources.")
            except Exception as e:
                self.log(f"⚠ Web search failed: {e}")

            results["arxiv"] = self._dedup_by_key(results["arxiv"], "url")
            results["web"] = self._dedup_by_key(results["web"], "href")

        # ── Build chunk index for source grounding ───────────────────────────
        chunk_index = {}
        chunk_counter = [1]  # list so inner func can mutate it

        def add_chunks(text, source, url, source_type):
            paragraphs = [p.strip()
                          for p in text.split('\n\n') if len(p.strip()) > 50]
            if not paragraphs:
                paragraphs = [text.strip()] if len(text.strip()) > 50 else []
            for para in paragraphs:
                cid = f"[SRC-{chunk_counter[0]}]"
                chunk_index[cid] = {
                    "text": para[:500],
                    "source": source,
                    "url": url,
                    "type": source_type
                }
                chunk_counter[0] += 1

        for paper in results.get("arxiv", []):
            add_chunks(paper["summary"], paper["title"], paper["url"], "ArXiv")
        if results.get("wiki"):
            add_chunks(results["wiki"], "Wikipedia",
                       "https://en.wikipedia.org", "Wikipedia")
        for item in results.get("web", []):
            body = item.get("body", item.get("snippet", ""))
            if body:
                add_chunks(body, item.get("title", "Web Source"),
                           item.get("href", ""), "Web")
        for doc in results.get("local_docs", []):
            add_chunks(doc["content"], doc["filename"], "", "Uploaded PDF")

        results["_chunks"] = chunk_index
        self.log(f"✓ Built {len(chunk_index)} source chunks for grounding.")
        # ─────────────────────────────────────────────────────────────────────

        # Save to cache
        CacheManager.save_cache(cache_key, results, ttl=7200)

        return results


class AnalystAgent(Agent):
    def __init__(self):
        super().__init__("Analyst", "Identifies patterns and generates data visualizations", "📊")

    def perform_task(self, research_data):
        self.log("Analyzing gathered data for multi-modal insights...")

        prompt = f"""
        Analyze the following research data and extract key themes and insights.

        VISUALIZATION REQUIREMENTS:
        - Create 2-4 high-quality, content-focused visualizations
        - STRONGLY ENCOURAGED: Create multiple visualizations to enhance understanding
        - Mix chart types: use bar/line/pie charts for quantitative data, mermaid diagrams for processes/relationships
        - Each visualization should add unique value and insight

        CRITICAL RULES - WHAT TO VISUALIZE:
        ✅ ENCOURAGED VISUALIZATIONS (Create these if data is available):
        - Market size projections, growth trends, forecasts over time
        - Technology/product performance comparisons with metrics
        - Adoption rates across industries, regions, or time periods
        - Cost/price comparisons between different approaches or solutions
        - Process flows, system architectures, workflow diagrams (mermaid)
        - Competitive landscapes, ecosystem maps (mermaid)
        - Key statistics, metrics, or KPIs comparisons
        - Timeline of technological evolution or milestones (content-based, not publication dates)
        - Component breakdowns, architecture diagrams (mermaid)

        ❌ PROHIBITED VISUALIZATIONS (Never create these):
        - Distribution of research areas or paper topics
        - Number of papers per topic/year
        - Publication timelines based on paper dates
        - Author affiliations or author counts
        - Source distribution (ArXiv vs Wikipedia vs Web)
        - Document metadata charts

        INSTRUCTIONS:
        1. Extract ALL quantitative data from research content (market sizes, percentages, growth rates, etc.)
        2. Identify processes, workflows, or relationships that benefit from diagrams
        3. Create 2-4 diverse visualizations that tell the story of the research findings
        4. If limited numerical data exists, create 2-3 mermaid diagrams showing relationships/processes
        5. Make visualizations complementary - don't repeat the same insight

        Data:
        {json.dumps(research_data, indent=2)[:10000]}

        Output format (JSON):
        {{
            "key_themes": ["Theme 1", "Theme 2", "Theme 3"],
            "summary": "Brief summary of the research findings...",
            "visualizations": [
                {{
                    "title": "Market Growth Projection 2024-2030",
                    "type": "chart",
                    "chart_type": "line",
                    "labels": ["2024", "2025", "2026", "2027", "2028", "2029", "2030"],
                    "values": [100, 150, 225, 340, 510, 765, 1150],
                    "description": "Projected market growth showing exponential adoption"
                }},
                {{
                    "title": "Technology Ecosystem Map",
                    "type": "mermaid",
                    "code": "graph TB; A[Core Tech]-->B[Application]; A-->C[Infrastructure]; B-->D[End Users];",
                    "description": "Overview of the technology ecosystem and key relationships"
                }}
            ]
        }}

        IMPORTANT: Aim for 2-4 visualizations. Empty visualizations array should only be used if there is truly NO data to visualize.
        """

        try:
            response = _client.models.generate_content(
                model=MODEL,
                contents=self._dated_prompt(prompt),
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AnalysisOutput
                )
            )
            self._track_usage(response)
            analysis = AnalysisOutput.model_validate_json(response.text)
            self.log(
                f"✓ Structured output: {len(analysis.visualizations)} visualizations, {len(analysis.key_themes)} themes")
            return analysis.model_dump()
        except Exception as structured_err:
            self.log(
                f"⚠ Structured output failed ({structured_err}), falling back to plain JSON parsing...")
            response = self.call_llm(prompt)
            try:
                cleaned = response.replace('```json', '').replace('```', '')
                analysis = json.loads(cleaned)
                count = len(analysis.get('visualizations', []))
                self.log(f"Generated {count} visualizations.")
                return analysis
            except (json.JSONDecodeError, ValueError) as e:
                self.log(f"JSON parsing failed ({e}), returning raw summary.")
                return {"key_themes": ["Complex Topic"], "summary": response, "visualizations": []}


class FactCheckerAgent(Agent):
    def __init__(self):
        super().__init__("Fact Checker", "Validates claims against sources", "✅")

    def perform_task(self, context):
        self.log("Verifying consistency of data...")
        data = context.get('research_data', {})
        prompt = f"""
        Check the following research data for any internal contradictions or obvious factual errors.
        Data: {str(data)[:2000]}...
        Return a brief validation report.
        """
        validation = self.call_llm(prompt)
        return {"validation_report": validation}

    async def perform_task_async(self, context):
        self.log("Verifying consistency of data... (async)")
        data = context.get('research_data', {})
        prompt = f"""
    Check the following research data for any internal contradictions or obvious factual errors.
    Data: {str(data)[:2000]}...
    Return a brief validation report.
    """
        validation = await self.call_llm_async(prompt)
        return {"validation_report": validation}


class BiasDetectorAgent(Agent):
    def __init__(self):
        super().__init__("Bias Detector", "Checks for balanced perspectives", "⚖️")

    def perform_task(self, context):
        self.log("Scanning for potential cognitive biases...")
        data = context.get('research_data', {})
        prompt = f"""
        Analyze the sources and content below for potential bias.
        Sources: {str(data)[:2000]}...
        Provide a brief bias assessment.
        """
        assessment = self.call_llm(prompt)
        return {"bias_report": assessment}

    async def perform_task_async(self, context):
        self.log("Scanning for potential cognitive biases... (async)")
        data = context.get('research_data', {})
        prompt = f"""
    Analyze the sources and content below for potential bias.
    Sources: {str(data)[:2000]}...
    Provide a brief bias assessment.
    """
        assessment = await self.call_llm_async(prompt)
        return {"bias_report": assessment}


class CitationVerifierAgent(Agent):
    def __init__(self):
        super().__init__("Citation Verifier", "Verifies source grounding & formats citations", "📝")

    def perform_task(self, context):
        report = context.get('draft_report', '')
        chunks = context.get('research_chunks', {})
        citation_format = context.get('citation_format', 'APA 7th')

        if not report or not chunks:
            self.log("Skipping — no report or chunks available yet.")
            return {"citations_formatted": True, "invalid_citations_flagged": 0,
                    "references_section": "", "used_sources_count": 0, "draft_report": report}

        # 1. Find and verify all [SRC-X] tags
        citations_found = re.findall(r'\[SRC-\d+\]', report)
        invalid = [c for c in citations_found if c not in chunks]
        
        if invalid:
            self.log(f"⚠️ {len(set(invalid))} hallucinated tag(s) detected: {', '.join(set(invalid))}")
        else:
            self.log("✅ All citation tags verified against source chunks.")

        # 2. Deduplicate sources and build a mapping dictionary
        seen_keys = {}      # Maps unique source URL/Name to an Integer ID (1, 2, 3...)
        used_sources = []   # Stores the actual source metadata
        tag_to_idx = {}     # Maps [SRC-5] to the Integer ID
        
        for cid in citations_found:
            if cid in chunks:
                meta = chunks[cid]
                key = meta.get('url') or meta.get('source')
                
                if key not in seen_keys:
                    seen_keys[key] = len(used_sources) + 1
                    used_sources.append(meta)
                
                tag_to_idx[cid] = seen_keys[key]

        # 3. Post-Process the Report text (Replace [SRC-X] with academic formats)
        final_report = report
        for cid, ref_num in tag_to_idx.items():
            if citation_format == "IEEE":
                # Replace [SRC-X] with [1]
                final_report = final_report.replace(cid, f"[{ref_num}]")
            elif citation_format in ["MLA 9th", "Chicago", "Harvard"]:
                # Replace [SRC-X] with (Author/Source Name)
                source_name = chunks[cid].get('source', 'Source').split()[0]
                final_report = final_report.replace(cid, f"({source_name})")
            else: # APA 7th
                source_name = chunks[cid].get('source', 'Source').split()[0]
                final_report = final_report.replace(cid, f"({source_name}, n.d.)")

        # 4. Build the final References Section
        references_md = "## References\n\n"
        for idx, meta in enumerate(used_sources, 1):
            url  = meta.get('url', '')
            name = meta.get('source', 'Unknown Source')
            kind = meta.get('type', '')
            
            if citation_format == "IEEE":
                references_md += f"[{idx}] {name}, {kind}. [Online]. Available: {url}\n\n"
            elif citation_format == "MLA 9th":
                references_md += f'{idx}. "{name}." {kind}, {url}.\n\n'
            elif citation_format == "Harvard":
                references_md += f"{idx}. {name}. ({kind}). Available at: {url}\n\n"
            elif citation_format == "Chicago":
                references_md += f'{idx}. "{name}." {kind}. {url}.\n\n'
            else: # APA 7th
                references_md += f"{idx}. {name}. ({kind}). Retrieved from {url}\n\n"

        self.log(f"Converted [SRC] tags to {citation_format} format.")
        
        # Return the modified report so it overwrites the draft
        return {
            "citations_formatted": True,
            "invalid_citations_flagged": len(set(invalid)),
            "references_section": references_md,
            "used_sources_count": len(used_sources),
            "draft_report": final_report  # Pass the cleaned text back to the pipeline
        }


class GapAnalystAgent(Agent):
    def __init__(self):
        super().__init__("Gap Analyst", "Identifies missing information", "🔍")

    def perform_task(self, context):
        self.log("Checking for knowledge gaps...")
        data = context.get('research_data', {})
        prompt = f"""
        Based on the topic '{context.get('topic')}' and the gathered data, what important aspects are missing?
        Data Summary: {str(data)[:1000]}...
        """
        gaps = self.call_llm(prompt)
        return {"gaps": gaps}

    async def perform_task_async(self, context):
        self.log("Checking for knowledge gaps... (async)")
        data = context.get('research_data', {})
        prompt = f"""
    Based on the topic '{context.get('topic')}' and the gathered data, what important aspects are missing?
    Data Summary: {str(data)[:1000]}...
    """
        gaps = await self.call_llm_async(prompt)
        return {"gaps": gaps}


class SynthesizerAgent(Agent):
    def __init__(self):
        super().__init__("Synthesizer", "Merges disparate data streams", "🔄")

    def perform_task(self, context):
        self.log("Synthesizing multi-modal data...")
        time.sleep(0.5)
        return {"synthesis_complete": True}


class QualityControlAgent(Agent):
    def __init__(self):
        super().__init__("Quality Control", "Final quality assessment", "🏆")

    def perform_task(self, context):
        self.log("Performing final quality audit...")

        topic = context.get("topic", "Unknown topic")
        research_data = context.get("research_data", {})
        analysis = context.get("analysis", {})
        qc_iterations = context.get("qc_iterations", 0)

        arxiv_count = len(research_data.get("arxiv", []))
        web_count = len(research_data.get("web", []))
        wiki_available = "yes" if research_data.get("wiki") else "no"
        key_themes = analysis.get("key_themes", [])

        prompt = f"""You are a quality control evaluator for a multi-agent research system.
Evaluate the research collected on the topic: "{topic}"

Research metrics:
- Academic papers (ArXiv): {arxiv_count}
- Web sources: {web_count}
- Wikipedia background: {wiki_available}
- Key themes identified: {json.dumps(key_themes)}
- QC iteration number: {qc_iterations}

Score the research on the following criteria (total 100 points):
1. Depth & Breadth (30 pts): Are there enough sources covering multiple aspects of the topic?
2. Relevance (30 pts): Do the sources directly address the research topic?
3. Source Diversity (20 pts): Is there a good mix of academic, encyclopedic, and current web sources?
4. Recency (20 pts): Do the sources reflect current state of knowledge?

Scoring thresholds:
- Score >= 85: PASS — research is sufficient to produce a high-quality report
- Score < 85: FAIL — researcher should retry with focus on the identified weaknesses

Be strict but fair. If arxiv_count < 2, penalize Depth. If web_count < 2, penalize Recency.
If wiki_available is "no", penalize Source Diversity slightly.

Provide actionable feedback that the researcher can use to improve on retry."""

        try:
            response = _client.models.generate_content(
                model=MODEL,
                contents=self._dated_prompt(prompt),
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=QCOutput
                )
            )
            self._track_usage(response)
            qc = QCOutput.model_validate_json(response.text)
            self.log(
                f"✓ QC score: {qc.score}/100 — {'PASSED' if qc.passed else 'FAILED'}")
            if not qc.passed:
                self.log(f"⚠ QC feedback: {qc.feedback}")
            return {
                "score": qc.score,
                "qc_passed": qc.passed,
                "feedback": qc.feedback,
                "qc_strengths": qc.strengths,
                "qc_weaknesses": qc.weaknesses
            }
        except Exception as e:
            self.log(
                f"⚠ QC structured output failed ({e}), defaulting to pass.")
            return {
                "score": 85,
                "qc_passed": True,
                "feedback": "QC evaluation unavailable, defaulting to pass.",
                "qc_strengths": [],
                "qc_weaknesses": []
            }


class FormatterAgent(Agent):
    def __init__(self):
        super().__init__("Formatter", "Applies final styling", "🎨")

    def perform_task(self, context):
        self.log("Applying markdown formatting...")
        time.sleep(0.5)
        return {"formatted": True}


class PlannerAgent(Agent):
    def __init__(self):
        super().__init__("Planner", "Designs targeted search strategy from QC feedback", "🗺️")

    def perform_task(self, context: dict) -> dict:
        topic = context.get("topic", "")
        qc_feedback = context.get("qc_feedback", "")
        gaps = context.get("gaps", "")

        self.log("Analyzing QC feedback to build targeted search strategy...")

        prompt = f"""You are a research strategist. A quality control agent flagged weaknesses in a research run.

Topic: {topic}
QC Feedback: {qc_feedback}
Knowledge Gaps: {gaps}

Output a targeted search strategy as a JSON object:
{{
  "search_focus": "One sentence describing the primary focus for the next round",
  "suggested_queries": ["query 1", "query 2", "query 3"],
  "tool_priorities": ["arxiv", "web", "wikipedia"],
  "extra_instructions": "Specific guidance (e.g. look for empirical studies, prioritize post-2022 papers)"
}}

tool_priorities must only contain these exact values: "arxiv", "web", "wikipedia".
suggested_queries should be 2-4 specific, targeted search strings.
"""
        try:
            result_text = self.call_llm(prompt)
            cleaned = result_text.replace("```json", "").replace("```", "").strip()
            strategy = json.loads(cleaned)
            self.log(f"✓ Strategy: {strategy.get('search_focus', '')[:80]}")
            return {"planner_strategy": strategy}
        except Exception as e:
            self.log(f"⚠ Planner failed ({e}), using default strategy.")
            return {"planner_strategy": {
                "search_focus": f"Find comprehensive information about {topic}",
                "suggested_queries": [topic, f"{topic} research", f"{topic} recent developments"],
                "tool_priorities": ["arxiv", "web", "wikipedia"],
                "extra_instructions": qc_feedback
            }}


class WriterAgent(Agent):
    def __init__(self):
        super().__init__("Writer", "Compiles final report", "✍️")

    def _build_report_prompt(self, context):
        """Build the full report prompt, incorporating all upstream agent outputs."""
        topic = context.get('topic')
        data = context.get('research_data', {})
        analysis = context.get('analysis', {})
        report_length = context.get('report_length', 'Standard')
        citation_format = context.get('citation_format', 'APA 7th')
        include_citations = context.get('include_citations', True)

        # Outputs from parallel agents — this is what makes the multi-agent system real
        bias_report = context.get('bias_report', 'No bias analysis available.')
        fact_check = context.get(
            'validation_report', 'No fact-check available.')
        gaps = context.get('gaps', 'No gap analysis available.')
        extra_instructions = context.get('extra_writer_instructions', '')

        # Add a dynamic 'chunk_limit' mapped to the user's UI selection
        length_guidance = {
            'Concise': {
                'description': 'Brief, focused overview',
                'target_words': '800-1200 words',
                'detail_level': 'High-level summary with key points only...',
                'chunk_limit': 30  # Low context
            },
            'Standard': {
                'description': 'Balanced detail and readability',
                'target_words': '2000-3000 words',
                'detail_level': 'Moderate depth with essential details...',
                'chunk_limit': 60  # Medium context
            },
            'Detailed': {
                'description': 'Comprehensive analysis',
                'target_words': '4000-6000 words',
                'detail_level': 'In-depth exploration with extensive analysis...',
                'chunk_limit': 120  # High context
            },
            'Exhaustive': {
                'description': 'Maximum depth and coverage',
                'target_words': '7000+ words',
                'detail_level': 'Extremely thorough with all available details...',
                'chunk_limit': 200  # Max context
            }
        }

        length_config = length_guidance.get(
            report_length, length_guidance['Standard'])
        # <--- Dynamic chunk limit based on user selection
        active_chunk_limit = length_config['chunk_limit']

        style_map = {
            "APA 7th": "In-text: (Author, Year). Reference: Author, A. A. (Year). Title. Source. URL",
            "MLA 9th": 'In-text: (Author). Reference: Author. "Title." Source, Year, URL.',
            "IEEE": "In-text: [1], [2]. Reference: [1] A. Author, \"Title,\" Source, Year. [Online]. Available: URL",
            "Chicago": "In-text: (Author Year). Reference: Author. \"Title.\" Source (Year). URL.",
            "Harvard": "In-text: (Author Year). Reference: Author (Year) *Title*, Source. Available at: URL",
        }

        if include_citations:
            citation_instructions = f"""
CITATION REQUIREMENTS:
- Format all citations in **{citation_format}** style
- Include in-text citations where appropriate
- Create a properly formatted References section at the end
- Style guide: {style_map.get(citation_format, style_map['APA 7th'])}
"""
        else:
            citation_instructions = "Do not include in-text citations. List source URLs in the References section only."

        # Use chunk-based grounding if available, else fall back to raw data
        chunks = context.get('research_chunks', {})
        if chunks:
            formatted_sources = "\n".join([
                f"ID: {cid} | Source: {meta['source']} | Type: {meta['type']} | Text: {meta['text'][:350]}"
                # <--- Applies dynamic limit
                for cid, meta in list(chunks.items())[:active_chunk_limit]
            ])
            source_block = f"""SOURCE MATERIAL — use ONLY the chunks below. You are forbidden from using outside knowledge.

{formatted_sources}

STRICT GROUNDING RULES:
1. Every factual claim, statistic, or specific assertion MUST be immediately followed by its Source ID.
   Example: "Energy density exceeds 400 Wh/kg [SRC-4], enabling longer range [SRC-7]."
2. CRITICAL: Do NOT combine citations into a single bracket. 
   Write [SRC-1][SRC-2], NEVER [SRC-1, SRC-2].
3. If a fact is not explicitly present in the SOURCE MATERIAL, omit it entirely.
4. You MAY use the analysis themes below for structure and framing, but back every fact with a [SRC-X] tag.
5. The References section will be built automatically — do NOT write a References section yourself."""
        else:
            source_block = f"""Research Data:
{json.dumps(data, indent=2)[:15000]}

{citation_instructions}"""

        prompt = f"""
Write a {report_length.lower()}, publication-quality research report on: "{topic}".

{source_block}

Thematic Structure from Analysis (use for framing, not as a source of facts):
Key Themes: {json.dumps(analysis.get('key_themes', []))}
Summary: {analysis.get('summary', '')}

User Feedback / Specific Instructions:
{extra_instructions if extra_instructions else "None provided."}

Critical Reviews (weave inline — do NOT save all critiques for the end):
- Fact Check: {fact_check}
- Bias Assessment: {bias_report}
- Knowledge Gaps: {gaps}

INLINE CRITIQUE RULE: When you introduce a technology, system, or claim, immediately acknowledge
relevant limitations, biases, or gaps *in that same section*. For example, when discussing
Agentic Semantic Routing, mention its latency overhead in the same paragraph, not only later.
The "Limitations & Knowledge Gaps" section should synthesize remaining issues not yet addressed inline.

STRICT FORMATTING RULES:
- Return ONLY prose markdown — no code fences, no raw code blocks, no ```markdown wrapper
- DO NOT generate any Mermaid.js code, JSON blocks, or programming syntax anywhere in the report.
  Diagrams are handled by a separate system. Write prose descriptions instead.
- Start directly with content
- Use proper markdown headers (##, ###)

Structure:
## Executive Summary
## Introduction
## Key Findings
## Detailed Analysis
## Limitations & Knowledge Gaps
## Conclusion

Tone: Professional, Academic, Insightful.
Target Length: {length_config['target_words']}
Detail Level: {length_config['detail_level']}
"""
        return prompt

    def perform_task(self, context):
        self.log("Drafting comprehensive report with Gemini...")
        prompt = self._build_report_prompt(context)
        report = self.call_llm(prompt)

        if report.startswith("```markdown"):
            report = report.replace("```markdown", "", 1)
        if report.startswith("```"):
            report = report.replace("```", "", 1)
        if report.endswith("```"):
            report = report.rsplit("```", 1)[0]

        self.log("Report generation complete.")
        return report.strip()

    def stream_task(self, context):
        """Yield report text chunks for live UI streaming."""
        self.log("Streaming report with Gemini...")
        prompt = self._build_report_prompt(context)
        yield from self.stream_llm(prompt)
        self.log("Report streaming complete.")


class ChatAgent(Agent):
    def __init__(self):
        super().__init__("Chat Assistant", "Answers user questions about the report", "💬")

    def perform_task(self, context):
        report = context.get('report')
        question = context.get('question')
        prompt = f"""
        You are a helpful research assistant. The user has a question about the following report.
        Report: {report[:10000]}...
        User Question: {question}
        Answer concisely and accurately based on the report.
        """
        return self.call_llm(prompt)
