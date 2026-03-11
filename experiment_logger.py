"""
experiment_logger.py
────────────────────
Lightweight run-level telemetry for the Multi-Agent Research System.
Writes one JSON record per research run to `experiment_log.jsonl` (append-only).
Designed for zero coupling — import and call from app.py only.

Usage (3 insertions in app.py):
    1. On research start:  experiment_logger.start_run(topic, config)
    2. On QC update:       experiment_logger.record_qc_event(score, passed, iteration)
    3. On report finish:   experiment_logger.finish_run(token_stats, grounding_stats, agent_breakdown)
"""

import json
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone

LOG_FILE = Path("experiment_log.jsonl")
_current_run: dict = {}


# ── Public API ────────────────────────────────────────────────────────────────

def start_run(topic: str, config: dict) -> str:
    """Call once when the user clicks Start Research. Returns a run_id."""
    global _current_run
    run_id = str(uuid.uuid4())[:8]
    _current_run = {
        "run_id":         run_id,
        "timestamp_utc":  datetime.now(timezone.utc).isoformat(),
        "topic":          topic,
        "report_depth":   config.get("report_length", "Standard"),
        "citation_fmt":   config.get("citation_format", "APA 7th"),
        "hitl_enabled":   config.get("enable_hitl", True),
        "qc_events":      [],          # [{iteration, score, passed}]
        "total_retries":  0,
        "wall_time_s":    None,
        "tokens": {
            "input":  0,
            "output": 0,
            "total":  0,
            "cost_usd": 0.0
        },
        "per_agent_tokens": {},        # {agent_name: {total, cost_usd}}
        "grounding": {
            "total_chunks":        0,
            "unique_sources_cited": 0,
            "hallucinated_tags":   0,
            "hallucination_rate":  0.0  # hallucinated / total citations found
        },
        "source_yield": {
            "arxiv_chunks":  0,
            "web_chunks":    0,
            "wiki_chunks":   0,
            "pdf_chunks":    0
        },
        "qc_final_score":  None,
        "qc_passed":       None,
    }
    return run_id


def record_qc_event(score: int, passed: bool, iteration: int):
    """Call each time the QC node emits a result (including retries)."""
    global _current_run
    if not _current_run:
        return
    _current_run["qc_events"].append({
        "iteration": iteration,
        "score":     score,
        "passed":    passed,
    })
    if not passed:
        _current_run["total_retries"] += 1
    _current_run["qc_final_score"] = score
    _current_run["qc_passed"]      = passed


def finish_run(
    token_stats: dict,
    grounding_stats: dict,
    agent_breakdown: dict,
    research_chunks: dict,
    wall_time_s: float,
):
    """
    Call once the CitationVerifier finishes and the report is displayed.

    token_stats      — from agents.get_token_usage()
    grounding_stats  — {"invalid": int, "used": int, "total": int}
    agent_breakdown  — from agents.get_agent_token_breakdown()
    research_chunks  — the research_chunks dict from ResearchState (for source_yield)
    wall_time_s      — time.time() - research_start_time
    """
    global _current_run
    if not _current_run:
        return

    # Token stats
    _current_run["tokens"] = {
        "input":    token_stats.get("input_tokens", 0),
        "output":   token_stats.get("output_tokens", 0),
        "total":    token_stats.get("total_tokens", 0),
        "cost_usd": token_stats.get("estimated_cost_usd", 0.0),
    }

    # Per-agent breakdown (condensed)
    _current_run["per_agent_tokens"] = {
        name: {
            "total_tokens": v.get("total_tokens", 0),
            "cost_usd":     v.get("estimated_cost_usd", 0.0),
        }
        for name, v in (agent_breakdown or {}).items()
    }

    # Grounding stats + hallucination rate
    total_chunks  = grounding_stats.get("total", 0)
    invalid_tags  = grounding_stats.get("invalid", 0)
    used_sources  = grounding_stats.get("used", 0)

    # Approximate total citations found = used_sources (unique) + invalid (hallucinated)
    total_cite_attempts = used_sources + invalid_tags
    hallucination_rate  = (
        round(invalid_tags / total_cite_attempts, 4)
        if total_cite_attempts > 0 else 0.0
    )

    _current_run["grounding"] = {
        "total_chunks":         total_chunks,
        "unique_sources_cited": used_sources,
        "hallucinated_tags":    invalid_tags,
        "hallucination_rate":   hallucination_rate,
    }

    # Source yield — count chunk types from research_chunks
    arxiv_n = wiki_n = web_n = pdf_n = 0
    for meta in (research_chunks or {}).values():
        t = meta.get("type", "")
        if t == "ArXiv":      arxiv_n += 1
        elif t == "Wikipedia": wiki_n  += 1
        elif t == "Web":       web_n   += 1
        elif t == "Uploaded PDF": pdf_n += 1

    _current_run["source_yield"] = {
        "arxiv_chunks": arxiv_n,
        "web_chunks":   web_n,
        "wiki_chunks":  wiki_n,
        "pdf_chunks":   pdf_n,
    }

    _current_run["wall_time_s"] = round(wall_time_s, 2)

    # Append to JSONL log
    _append_to_log(dict(_current_run))
    _current_run = {}


def get_summary_stats() -> dict:
    """
    Read all logged runs and return aggregate stats for display or paper tables.
    Returns empty dict if no runs logged yet.
    """
    if not LOG_FILE.exists():
        return {}

    runs = _load_all_runs()
    if not runs:
        return {}

    n = len(runs)

    def avg(key_path):
        vals = []
        for r in runs:
            v = r
            for k in key_path:
                v = v.get(k, {}) if isinstance(v, dict) else None
                if v is None:
                    break
            if isinstance(v, (int, float)):
                vals.append(v)
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def pct_passed():
        passed = sum(1 for r in runs if r.get("qc_passed") is True)
        return round(passed / n * 100, 1)

    def avg_retries():
        return avg(["total_retries"])

    return {
        "total_runs":              n,
        "avg_wall_time_s":         avg(["wall_time_s"]),
        "avg_cost_usd":            avg(["tokens", "cost_usd"]),
        "avg_total_tokens":        avg(["tokens", "total"]),
        "avg_qc_score":            avg(["qc_final_score"]),
        "qc_first_pass_rate_pct":  pct_passed(),
        "avg_retries_per_run":     avg_retries(),
        "avg_hallucination_rate":  avg(["grounding", "hallucination_rate"]),
        "avg_unique_sources":      avg(["grounding", "unique_sources_cited"]),
        "avg_total_chunks":        avg(["grounding", "total_chunks"]),
        "avg_arxiv_chunks":        avg(["source_yield", "arxiv_chunks"]),
        "avg_web_chunks":          avg(["source_yield", "web_chunks"]),
        "avg_wiki_chunks":         avg(["source_yield", "wiki_chunks"]),
    }


def load_runs_as_table() -> list[dict]:
    """Return all runs as a flat list of dicts — ready for pd.DataFrame()."""
    runs = _load_all_runs()
    flat = []
    for r in runs:
        flat.append({
            "run_id":            r.get("run_id"),
            "timestamp":         r.get("timestamp_utc"),
            "topic":             r.get("topic"),
            "report_depth":      r.get("report_depth"),
            "wall_time_s":       r.get("wall_time_s"),
            "qc_final_score":    r.get("qc_final_score"),
            "qc_passed":         r.get("qc_passed"),
            "total_retries":     r.get("total_retries"),
            "cost_usd":          r.get("tokens", {}).get("cost_usd"),
            "total_tokens":      r.get("tokens", {}).get("total"),
            "hallucination_rate":r.get("grounding", {}).get("hallucination_rate"),
            "hallucinated_tags": r.get("grounding", {}).get("hallucinated_tags"),
            "unique_sources":    r.get("grounding", {}).get("unique_sources_cited"),
            "total_chunks":      r.get("grounding", {}).get("total_chunks"),
            "arxiv_chunks":      r.get("source_yield", {}).get("arxiv_chunks"),
            "web_chunks":        r.get("source_yield", {}).get("web_chunks"),
            "wiki_chunks":       r.get("source_yield", {}).get("wiki_chunks"),
        })
    return flat


# ── Private helpers ───────────────────────────────────────────────────────────

def _append_to_log(record: dict):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _load_all_runs() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    runs = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return runs