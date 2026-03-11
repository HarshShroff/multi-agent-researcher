# 🔬 MARS — Multi-Agent Research System

> An autonomous research pipeline that orchestrates 11 specialized LLM agents via a LangGraph state machine to produce citation-grounded reports on any topic — with a self-correcting QC loop, async parallel critics, and optional human review.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-green.svg)](https://github.com/langchain-ai/langgraph)
[![Gemini Flash](https://img.shields.io/badge/Gemini-2.5--flash-orange.svg)](https://aistudio.google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What it does

You give MARS a topic. It retrieves sources from ArXiv, the web, and Wikipedia via a Model Context Protocol (MCP) server, runs three parallel critic agents, scores the research quality, and — if the score is too low — generates a targeted search strategy and tries again. When quality passes, it streams a fully cited report to the UI.

Every factual claim in the final report is tied to a specific retrieved chunk via an `[SRC-X]` tag. A Citation Verifier scans for broken tags before the report is finalized.

**Preliminary evaluation across 15 runs:**
| Metric | Value |
|---|---|
| Mean cost per report | $0.0028 |
| Mean latency | 130s |
| QC first-pass rate | 87% (13/15) |
| Citation tag error rate **†** | 0% |
| Mean sources cited | 10.5 |

> **†** The "0% citation tag error rate" measures *referential integrity* — whether the LLM correctly formatted `[SRC-X]` tags that point to real chunks in the index. It does **not** measure semantic faithfulness (whether the claim actually matches the chunk content). Full faithfulness evaluation via RAGAS or LLM-as-judge is a [planned addition](#roadmap). This distinction matters; see [Limitations](#limitations).

---

## Architecture

```
User Input & PDFs
       │
       ▼
  ┌─────────────────────────────────────────┐
  │  Researcher (ReAct + MCP Tools)         │
  │  ArXiv · DuckDuckGo · Wikipedia         │
  │  URL deduplication · PDF parsing        │
  └──────────────┬──────────────────────────┘
                 │ indexed chunks [SRC-0..N]
                 ▼
  ┌──────────────────────────────────────────┐
  │  Analyst  (Pydantic structured output)   │
  │  Plotly charts · Mermaid diagrams        │
  └──────────────┬───────────────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Parallel Critics  (asyncio.gather)                     │
  │  Gap Analyst · Bias Detector · Fact Checker             │
  └──────────────────────┬──────────────────────────────────┘
                         │
                         ▼
                    Synthesizer
                         │
                         ▼
              ┌─────────────────┐
              │   QC Agent      │  score 0–100
              │   threshold: 85 │
              └───────┬─────────┘
                      │
          ┌───────────┴────────────┐
          │ FAIL                   │ PASS
          ▼                        ▼
    ┌───────────┐          ┌──────────────────┐
    │  Planner  │          │  HITL Gate       │  (toggleable)
    │  JSON     │          │  approve/inject  │
    │  strategy │          └────────┬─────────┘
    └─────┬─────┘                   │
          │ targeted queries        ▼
          └──────► Researcher  Formatter → Writer (streaming)
                                         │
                                         ▼
                              Citation Verifier
                              (regex · tag integrity)
                                         │
                                         ▼
                               Final Report + Visualizations
                               PDF · Markdown · JSON · TXT
```

<!-- ```mermaid
graph TD
    %% Node Definitions
    User([User Input & PDFs])
    R["Researcher<br/>(ReAct + MCP Tools)<br/>ArXiv · DuckDuckGo · Wikipedia<br/>URL dedupe · PDF parsing"]
    A["Analyst<br/>(Pydantic Output)<br/>Plotly charts · Mermaid diagrams"]
    S["Synthesizer"]
    QC{"QC Agent<br/>Score 0-100<br/>Threshold: 85"}
    P["Planner<br/>(JSON strategy)"]
    HITL{{"HITL Gate<br/>(Toggleable)<br/>Approve / Inject"}}
    FMT["Formatter"]
    W["Writer<br/>(Streaming)"]
    CV["Citation Verifier<br/>(Regex · Tag integrity)"]
    Final(["Final Report + Visualizations<br/>PDF · Markdown · JSON · TXT"])

    %% Flow Routing
    User --> R
    R -- "indexed chunks [SRC-0..N]" --> A
    
    A --> Critics
    
    subgraph Critics [Parallel Critics]
        direction LR
        G["Gap Analyst"]
        B["Bias Detector"]
        F["Fact Checker"]
    end
    
    Critics --> S
    S --> QC
    
    %% Cyclic Retry Loop
    QC -- "FAIL" --> P
    P -- "targeted queries" --> R
    
    %% Forward Progress
    QC -- "PASS" --> HITL
    HITL --> FMT
    FMT --> W
    W --> CV
    CV --> Final
``` -->

### The QC retry loop

When the QC agent scores research below 85/100, it doesn't just re-run the same search. A dedicated **Planner agent** reads the QC feedback and outputs a structured JSON strategy:

```json
{
  "search_focus": "missing: longitudinal clinical trial data",
  "suggested_queries": ["psilocybin RCT 2022 NEJM", "MDMA PTSD phase 3 trial"],
  "tool_priorities": ["arxiv", "web"]
}
```

The Researcher uses this on the next iteration instead of a generic retry. Two runs in the evaluation set triggered this path — both on topics with contested or thin primary literature — and neither reached the 85-point threshold within 2 iterations, which we treat as correct behavior: the system surfaced uncertainty rather than generating an overconfident report.

---

## Agent roster

| Agent | Role | Notes |
|---|---|---|
| **Researcher** | ReAct loop over MCP tools | URL deduplication, PDF parsing, planner-strategy-aware |
| **Analyst** | Pydantic-typed visualization generation | Plotly + Mermaid.js |
| **Gap Analyst** | Identifies missing coverage | Async |
| **Bias Detector** | Flags epistemic/perspective bias | Async |
| **Fact Checker** | Flags internal contradictions | Async |
| **Synthesizer** | Merges critic outputs into context | — |
| **QC Agent** | Scores 0–100, triggers retry or pass | Threshold: 85 |
| **Planner** | On QC fail: produces structured search strategy | JSON, Pydantic schema |
| **Formatter** | Final style constraints for Writer | — |
| **Writer** | Streams prose with `[SRC-X]` grounding tags | Depth-scaled context window |
| **Citation Verifier** | Regex integrity scan + citation formatting | IEEE / APA / MLA / Harvard / Chicago |

---

## Setup

**Prerequisites:** Python 3.11+, a [Gemini API key](https://aistudio.google.com/app/apikey), optionally a [LangSmith key](https://smith.langchain.com) for tracing.

```bash
git clone https://github.com/HarshShroff/multi-agent-researcher
cd multi-agent-researcher

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file:

```env
GEMINI_API_KEY=your_key_here

# Optional — LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=mars-researcher
```

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

To run the standalone MCP server for external clients (Claude Desktop, Cursor):
```bash
python mcp_server.py
```

---

## Usage

1. **Set depth** — Concise / Standard / Detailed / Exhaustive in the sidebar. This controls the Writer's context window, not retrieval volume (see Limitations).
2. **Toggle HITL** — when enabled, the system pauses before writing. You see the QC score, token cost, and source list. Approve or inject feedback to force a re-research.
3. **Upload PDFs** — drop local papers into the Knowledge Base; they're parsed into indexed chunks and included in the RAG pool.
4. **Run** — enter a topic and watch the agents execute. The final report streams live.
5. **Export** — PDF, Markdown, JSON, or TXT from the results panel.
6. **Cost breakdown** — expand the 🪙 Token Cost Breakdown panel to see per-agent spend.

---

## Limitations

These are real and worth knowing before you build on this.

**Retrieval cap (known engineering limitation)**
The Researcher fetches `max_results=5` per tool call, yielding 10–12 chunks regardless of depth setting. "Exhaustive" depth tells the Writer to produce a longer report from the same data — which risks verbosity and unsupported elaboration at higher word counts. Depth-responsive retrieval scaling is on the roadmap.

**QC is self-assessed**
The QC agent uses the same foundation model (Gemini Flash) that generated the research. LLM sycophancy is a documented issue; scores of 91/100 reflect the model's self-evaluation, not a ground-truth quality measure. A future version will run a stronger, independent model as judge or incorporate human-annotated baselines.

**Citation tag integrity ≠ semantic faithfulness**
The Citation Verifier checks that `[SRC-X]` tags point to real indexed chunks. It does not verify that the claim adjacent to the tag is actually supported by that chunk's content. Full faithfulness evaluation (e.g., RAGAS Context-Answer Relevance, TruLens) is planned but not yet implemented.

**N=15 evaluation set**
The preliminary metrics (87% QC first-pass, $0.0028/report) are from 15 runs across 8 topic domains. This is too small for statistical claims; treat them as indicative, not benchmarks.

**HITL latency includes human review time**
The reported 2.8× HITL overhead includes the time it took a human to read the dashboard and click Approve. The true computational overhead is a second Writer invocation (~1.9× token cost). Separating these cleanly requires instrumentation changes that are on the roadmap.

---

## Roadmap

- [ ] Depth-responsive retrieval (`max_results` scales with depth setting)
- [ ] RAGAS / TruLens semantic faithfulness evaluation
- [ ] Independent judge model for QC (GPT-4o or Claude Sonnet)
- [ ] Planner ablation vs. raw-feedback retry baseline
- [ ] HITL instrumentation: log interrupt entry/exit times separately
- [ ] Structured data source support via additional MCP servers (SQL, APIs)
- [ ] Multi-model critic ensemble

---

## Experiment logging

MARS ships with `experiment_logger.py` — a lightweight JSONL logger that records one record per run:

```python
# Fields logged per run
{
  "run_id", "timestamp_utc", "topic", "report_depth", "hitl_enabled",
  "qc_events",           # [{iteration, score, passed}]
  "total_retries",
  "wall_time_s",
  "tokens",              # {input, output, total, cost_usd}
  "per_agent_tokens",    # per-agent breakdown
  "grounding",           # {total_chunks, unique_sources_cited, hallucinated_tags, hallucination_rate}
  "source_yield"         # {arxiv_chunks, web_chunks, wiki_chunks, pdf_chunks}
}
```

Run `analyze_runs.py` to generate summary stats and four figures (QC scores, cost vs. depth, hallucination rate, source yield) from any accumulated JSONL log.

---

## Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph (StateGraph + MemorySaver) |
| LLM | Gemini 2.5 Flash Preview |
| Tools | MCP (ArXiv, DuckDuckGo, Wikipedia) |
| Structured output | Pydantic |
| Visualizations | Plotly, Mermaid.js |
| UI | Streamlit |
| Tracing | LangSmith |
| PDF export | xhtml2pdf + CairoSVG (Kaleido fallback) |

---

## Citation

If you use MARS in your work:

```bibtex
@misc{shroff2026mars,
  title   = {MARS: A Multi-Agent Research System with Cyclic Quality Control
             and Chunk-Level Hallucination Grounding},
  author  = {Shroff, Harsh},
  year    = {2026},
  url     = {https://github.com/HarshShroff/multi-agent-researcher}
}
```

---

## License

MIT. See [LICENSE](LICENSE).