"""
analyze_runs.py
───────────────
Run this after collecting 10+ research runs to generate paper-ready tables and plots.
Usage:  python analyze_runs.py

Outputs:
  - Prints aggregate stats table to terminal
  - Saves runs_table.csv  (all runs, flat)
  - Saves figures/qc_scores.png
  - Saves figures/cost_vs_depth.png
  - Saves figures/hallucination_rate.png
  - Saves figures/source_yield.png
"""

import json
import sys
from pathlib import Path

# ── Try imports gracefully ────────────────────────────────────────────────────
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
except ImportError:
    sys.exit("Install deps first:  pip install pandas matplotlib numpy")

LOG_FILE = Path("experiment_log.jsonl")
FIG_DIR  = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

STYLE = {
    "bg":    "#0a0e27",
    "fg":    "#e8eaf6",
    "blue":  "#667eea",
    "purple":"#764ba2",
    "green": "#4ade80",
    "red":   "#f87171",
    "grid":  "#1a1d3a",
}

plt.rcParams.update({
    "figure.facecolor": STYLE["bg"],
    "axes.facecolor":   STYLE["bg"],
    "axes.edgecolor":   STYLE["grid"],
    "axes.labelcolor":  STYLE["fg"],
    "xtick.color":      STYLE["fg"],
    "ytick.color":      STYLE["fg"],
    "text.color":       STYLE["fg"],
    "grid.color":       STYLE["grid"],
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "sans-serif",
    "figure.dpi":       150,
})


# ── Load data ─────────────────────────────────────────────────────────────────

def load_runs() -> pd.DataFrame:
    if not LOG_FILE.exists():
        sys.exit(f"No log file found at {LOG_FILE}. Run the system first.")
    
    rows = []
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                rows.append({
                    "run_id":             r.get("run_id"),
                    "timestamp":          r.get("timestamp_utc"),
                    "topic":              r.get("topic", "")[:40],
                    "report_depth":       r.get("report_depth", "Standard"),
                    "hitl_enabled":       r.get("hitl_enabled", True),
                    "wall_time_s":        r.get("wall_time_s"),
                    "qc_final_score":     r.get("qc_final_score"),
                    "qc_passed":          r.get("qc_passed"),
                    "total_retries":      r.get("total_retries", 0),
                    "cost_usd":           r.get("tokens", {}).get("cost_usd", 0),
                    "total_tokens":       r.get("tokens", {}).get("total", 0),
                    "input_tokens":       r.get("tokens", {}).get("input", 0),
                    "output_tokens":      r.get("tokens", {}).get("output", 0),
                    "hallucinated_tags":  r.get("grounding", {}).get("hallucinated_tags", 0),
                    "hallucination_rate": r.get("grounding", {}).get("hallucination_rate", 0.0),
                    "unique_sources":     r.get("grounding", {}).get("unique_sources_cited", 0),
                    "total_chunks":       r.get("grounding", {}).get("total_chunks", 0),
                    "arxiv_chunks":       r.get("source_yield", {}).get("arxiv_chunks", 0),
                    "web_chunks":         r.get("source_yield", {}).get("web_chunks", 0),
                    "wiki_chunks":        r.get("source_yield", {}).get("wiki_chunks", 0),
                    "pdf_chunks":         r.get("source_yield", {}).get("pdf_chunks", 0),
                })
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line: {e}")
    
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


# ── Print aggregate table ─────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    n = len(df)
    print(f"\n{'═'*60}")
    print(f"  Multi-Agent Research System — Experiment Summary")
    print(f"  {n} runs logged")
    print(f"{'═'*60}")

    metrics = {
        "Avg Wall Time (s)":          df["wall_time_s"].mean(),
        "Avg Cost (USD)":             df["cost_usd"].mean(),
        "Avg Total Tokens":           df["total_tokens"].mean(),
        "Avg QC Score":               df["qc_final_score"].mean(),
        "QC First-Pass Rate (%)":     df["qc_passed"].mean() * 100,
        "Avg Retries / Run":          df["total_retries"].mean(),
        "Avg Hallucination Rate":     df["hallucination_rate"].mean(),
        "Avg Unique Sources Cited":   df["unique_sources"].mean(),
        "Avg Total Chunks Indexed":   df["total_chunks"].mean(),
        "Avg ArXiv Chunks":           df["arxiv_chunks"].mean(),
        "Avg Web Chunks":             df["web_chunks"].mean(),
        "Avg Wiki Chunks":            df["wiki_chunks"].mean(),
    }

    for k, v in metrics.items():
        if "Rate" in k:
            print(f"  {k:<35} {v:.4f}")
        elif "%" in k:
            print(f"  {k:<35} {v:.1f}%")
        elif "Tokens" in k:
            print(f"  {k:<35} {v:,.0f}")
        elif "USD" in k:
            print(f"  {k:<35} ${v:.4f}")
        else:
            print(f"  {k:<35} {v:.2f}")
    print(f"{'═'*60}\n")

    # Per-depth breakdown
    if df["report_depth"].nunique() > 1:
        print("  Per-Depth Breakdown:")
        print(f"  {'Depth':<12} {'N':>4} {'Avg QC':>8} {'Avg Cost':>10} {'Avg Time(s)':>12}")
        print(f"  {'-'*50}")
        for depth, grp in df.groupby("report_depth"):
            print(f"  {depth:<12} {len(grp):>4} {grp['qc_final_score'].mean():>8.1f} "
                  f"${grp['cost_usd'].mean():>9.4f} {grp['wall_time_s'].mean():>12.1f}")
        print()


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_qc_scores(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("QC Score Distribution", fontsize=14, fontweight="bold", color=STYLE["fg"])

    # Distribution histogram
    ax = axes[0]
    ax.hist(df["qc_final_score"].dropna(), bins=10, color=STYLE["blue"], edgecolor=STYLE["bg"], alpha=0.9)
    ax.axvline(85, color=STYLE["red"], linestyle="--", linewidth=1.5, label="Pass threshold (85)")
    ax.set_xlabel("QC Score")
    ax.set_ylabel("Runs")
    ax.set_title("Score Distribution")
    ax.legend(fontsize=9)
    ax.grid(True)

    # Score over runs
    ax = axes[1]
    ax.plot(range(len(df)), df["qc_final_score"], marker="o", color=STYLE["blue"],
            linewidth=1.5, markersize=5)
    ax.axhline(85, color=STYLE["red"], linestyle="--", linewidth=1.5, label="Pass threshold")
    ax.fill_between(range(len(df)), df["qc_final_score"], 85,
                    where=df["qc_final_score"] < 85, alpha=0.2, color=STYLE["red"], label="Below threshold")
    ax.set_xlabel("Run #")
    ax.set_ylabel("QC Score")
    ax.set_title("Score Over Time")
    ax.legend(fontsize=9)
    ax.grid(True)

    plt.tight_layout()
    path = FIG_DIR / "qc_scores.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_cost_vs_depth(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Cost & Latency by Report Depth", fontsize=14, fontweight="bold", color=STYLE["fg"])

    depth_order = ["Concise", "Standard", "Detailed", "Exhaustive"]
    present = [d for d in depth_order if d in df["report_depth"].values]

    colors = [STYLE["blue"], STYLE["purple"], STYLE["green"], STYLE["red"]][:len(present)]

    for i, (ax, col, ylabel) in enumerate(zip(
        axes,
        ["cost_usd", "wall_time_s"],
        ["Avg Cost (USD)", "Avg Wall Time (s)"]
    )):
        means = [df[df["report_depth"] == d][col].mean() for d in present]
        stds  = [df[df["report_depth"] == d][col].std() for d in present]
        bars  = ax.bar(present, means, color=colors, edgecolor=STYLE["bg"],
                       yerr=[s if not pd.isna(s) else 0 for s in stds],
                       capsize=4, error_kw={"color": STYLE["fg"], "linewidth": 1.2})
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, axis="y")
        for bar, val in zip(bars, means):
            if not pd.isna(val):
                label = f"${val:.4f}" if "USD" in ylabel else f"{val:.0f}s"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                        label, ha="center", va="bottom", fontsize=9, color=STYLE["fg"])

    plt.tight_layout()
    path = FIG_DIR / "cost_vs_depth.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_hallucination(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Citation Grounding & Hallucination Analysis", fontsize=14,
                 fontweight="bold", color=STYLE["fg"])

    # Hallucination rate per run
    ax = axes[0]
    c = [STYLE["green"] if r == 0 else STYLE["red"] for r in df["hallucinated_tags"]]
    ax.bar(range(len(df)), df["hallucination_rate"], color=c, edgecolor=STYLE["bg"])
    ax.set_xlabel("Run #")
    ax.set_ylabel("Hallucination Rate")
    ax.set_title("Hallucination Rate per Run\n(green = 0 hallucinations)")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.grid(True, axis="y")

    # Sources cited vs chunks available
    ax = axes[1]
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w/2, df["total_chunks"],   width=w, label="Chunks Available",
           color=STYLE["blue"],   edgecolor=STYLE["bg"])
    ax.bar(x + w/2, df["unique_sources"], width=w, label="Unique Sources Cited",
           color=STYLE["purple"], edgecolor=STYLE["bg"])
    ax.set_xlabel("Run #")
    ax.set_ylabel("Count")
    ax.set_title("Source Utilization")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y")

    plt.tight_layout()
    path = FIG_DIR / "hallucination_rate.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_source_yield(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle("Source Yield by Type Across Runs", fontsize=14,
                 fontweight="bold", color=STYLE["fg"])

    x = np.arange(len(df))
    w = 0.2
    offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]
    sources = [
        ("arxiv_chunks",  "ArXiv",     STYLE["blue"]),
        ("web_chunks",    "Web",       STYLE["purple"]),
        ("wiki_chunks",   "Wikipedia", STYLE["green"]),
        ("pdf_chunks",    "PDF",       "#fbbf24"),
    ]

    for (col, label, color), offset in zip(sources, offsets):
        ax.bar(x + offset, df[col], width=w, label=label,
               color=color, edgecolor=STYLE["bg"], alpha=0.9)

    ax.set_xlabel("Run #")
    ax.set_ylabel("Chunks")
    ax.set_title("Chunks per Source Type")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y")

    plt.tight_layout()
    path = FIG_DIR / "source_yield.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\nLoading experiment log...")
    df = load_runs()
    print(f"  {len(df)} runs found.\n")

    if len(df) == 0:
        print("No completed runs in log yet. Run the system on some topics first.")
        return

    # Save CSV
    csv_path = Path("runs_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved raw data: {csv_path}")

    print_summary(df)

    print("Generating figures...")
    plot_qc_scores(df)
    if df["report_depth"].nunique() > 1:
        plot_cost_vs_depth(df)
    else:
        print("  (Skipping cost_vs_depth — only one depth setting used so far)")
    plot_hallucination(df)
    plot_source_yield(df)

    print(f"\nDone. Figures in ./{FIG_DIR}/")
    print("Use runs_table.csv directly for LaTeX tables.\n")


if __name__ == "__main__":
    main()