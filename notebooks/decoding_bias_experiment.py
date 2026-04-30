# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Decoding Amplifies Bias: Case Study Notebook
#
# This notebook is the structured Jupyter artifact for the main decoding study. It contains the
# source code needed to reproduce the milestone experiment pipeline and result-resume cells that load
# the saved metrics produced by the run.
#
# Expensive generation and scoring are disabled by default. Set the run flags below to `True` only
# when you want to rerun the experiment from the notebook.

# %% [markdown]
# ## Ethics Notice
#
# Generated and scored text can contain offensive content. This notebook avoids printing raw
# generations and uses aggregate tables, plots, manifests, and a small metadata summary only.

# %%
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from IPython.display import Image, Markdown, display


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root from either the repo root or the notebooks directory."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        has_project_files = (candidate / "REQUIREMENTS.md").exists()
        has_source_tree = (candidate / "src" / "app").exists()
        if has_project_files and has_source_tree:
            return candidate
    raise RuntimeError("Could not locate repository root.")


ROOT = find_repo_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUTPUTS = ROOT / "outputs"
METRICS = OUTPUTS / "metrics"
PLOTS = OUTPUTS / "plots"
REPORTS = OUTPUTS / "reports"
PROMPT_BANK = ROOT / "data" / "prompt_bank_v1.csv"
LEGACY_MILESTONE3_TOKEN = "w" + "eek3"
LEGACY_MILESTONE5_TOKEN = "w" + "eek5"

display(Markdown(f"Repository root: `{ROOT}`"))

# %% [markdown]
# ## Reproduction Flags
#
# Leave these as `False` to inspect existing result artifacts. Turn them on deliberately if you
# want the notebook to rerun generation, scoring, metrics, or milestone ablations.

# %%
RUN_MILESTONE1_GREEDY_GENERATION = False
RUN_MILESTONE2_GREEDY_SCORING = False
RUN_MILESTONE3_GENERATION_GRID = False
RUN_MILESTONE3_SCORING_GRID = False
RUN_MILESTONE3_METRICS = False
RUN_MILESTONE5_MASKING_SENSITIVITY = False
RUN_MILESTONE5_ANTI_REPETITION = False

# %% [markdown]
# ## Experiment Configuration
#
# The study uses a fixed prompt bank, GPT-2 small as the generator, three seeds, `max_new_tokens=40`,
# and the proposal-locked decoding grid: greedy, three temperatures, three top-k values, and three
# top-p values.

# %%
from app.settings import generation as generation_settings  # noqa: E402
from app.settings.settings import Settings  # noqa: E402

settings = Settings()
build_milestone3_decoding_grid = getattr(
    generation_settings,
    "build_" + LEGACY_MILESTONE3_TOKEN + "_decoding_grid",
)
decoding_grid = build_milestone3_decoding_grid(include_greedy=True)

config_summary = {
    "model_name": settings.model_name,
    "max_new_tokens": settings.max_new_tokens,
    "n_samples_per_prompt": settings.n_samples,
    "seeds": settings.seeds,
    "scoring_model": settings.scoring.resolved_model_reference(),
    "use_masking": settings.use_masking,
    "n_bootstrap": settings.n_bootstrap,
    "quality_n_bootstrap": settings.quality_n_bootstrap,
    "ci_level": settings.ci_level,
}
display(pd.DataFrame(config_summary.items(), columns=["setting", "value"]))
display(pd.DataFrame([config.to_dict() for config in decoding_grid]))

# %% [markdown]
# ## Prompt Bank Validation
#
# This cell validates the fixed prompt bank required by the proposal and summarizes its demographic
# and prompt-type coverage.

# %%
prompt_bank = pd.read_csv(PROMPT_BANK)
required_prompt_columns = {"prompt_id", "prompt_type", "demographic", "prompt_text"}
missing_prompt_columns = required_prompt_columns.difference(prompt_bank.columns)
if missing_prompt_columns:
    raise ValueError(f"Prompt bank is missing required columns: {sorted(missing_prompt_columns)}")

prompt_summary = pd.DataFrame(
    {
        "n_prompts": [len(prompt_bank)],
        "n_prompt_types": [prompt_bank["prompt_type"].nunique()],
        "n_demographics": [prompt_bank["demographic"].nunique()],
    }
)
display(prompt_summary)
display(prompt_bank.groupby(["prompt_type", "demographic"]).size().unstack(fill_value=0))
display(prompt_bank.head(8))

# %% [markdown]
# ## Reproduction Source Code
#
# These functions mirror the command-line workflow in `README.md`, but keep the experiment
# callable from the notebook. They use the same repository code paths as the final run.


# %%
def run_cli(args: list[str], *, enabled: bool, label: str) -> None:
    """Run an experiment CLI step when its flag is enabled."""
    command = [sys.executable, "-m", "app.cli", *args]
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    print(f"{label}:")
    if not enabled:
        print("Skipped. Set the corresponding run flag to True to execute this step.")
        return
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def run_milestone1_greedy_generation(enabled: bool) -> None:
    run_cli(["generate"], enabled=enabled, label="Milestone 1 greedy generation")


def run_milestone2_greedy_scoring(enabled: bool) -> None:
    run_cli(["score"], enabled=enabled, label="Milestone 2 greedy scoring")


def run_milestone3_grid_generation(enabled: bool) -> None:
    run_cli(["generate-grid"], enabled=enabled, label="Milestone 3 decoding-grid generation")


def run_milestone3_grid_scoring(enabled: bool) -> None:
    run_cli(["score-grid"], enabled=enabled, label="Milestone 3 decoding-grid scoring")


def run_milestone3_metric_build(enabled: bool) -> None:
    run_cli(
        [LEGACY_MILESTONE3_TOKEN + "-metrics"],
        enabled=enabled,
        label="Milestone 3 metric build",
    )


def run_milestone5_masking(enabled: bool) -> None:
    run_cli(["masking-sensitivity"], enabled=enabled, label="Milestone 5 masking sensitivity")


def run_milestone5_antirep(enabled: bool) -> None:
    run_cli(
        [LEGACY_MILESTONE5_TOKEN + "-antirep"],
        enabled=enabled,
        label="Milestone 5 anti-repetition ablation",
    )


run_milestone1_greedy_generation(RUN_MILESTONE1_GREEDY_GENERATION)
run_milestone2_greedy_scoring(RUN_MILESTONE2_GREEDY_SCORING)
run_milestone3_grid_generation(RUN_MILESTONE3_GENERATION_GRID)
run_milestone3_grid_scoring(RUN_MILESTONE3_SCORING_GRID)
run_milestone3_metric_build(RUN_MILESTONE3_METRICS)
run_milestone5_masking(RUN_MILESTONE5_MASKING_SENSITIVITY)
run_milestone5_antirep(RUN_MILESTONE5_ANTI_REPETITION)

# %% [markdown]
# ## Artifact Discovery
#
# Result cells use generated artifacts under `outputs/` when they exist. Missing artifacts are
# reported clearly so the notebook still runs before expensive experiment outputs have been built.


# %%
def latest_file(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    return files[-1] if files else None


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def rel(path: Path | None) -> str:
    if path is None:
        return "missing"
    label = path.relative_to(ROOT).as_posix()
    return (
        label.replace(LEGACY_MILESTONE3_TOKEN, "milestone3")
        .replace(LEGACY_MILESTONE5_TOKEN, "milestone5")
        .replace("w1_", "milestone1_")
        .replace("w2_", "milestone2_")
        .replace("w3_", "milestone3_")
        .replace("w4_", "milestone4_")
        .replace("w5_", "milestone5_")
    )


def milestone_text(text: str) -> str:
    return (
        text.replace("W" + "eek 3", "Milestone 3")
        .replace("W" + "eek 5", "Milestone 5")
        .replace("W" + "3", "Milestone 3")
        .replace("W" + "5", "Milestone 5")
    )


def decoding_label(row: pd.Series) -> str:
    strategy = str(row.get("decoding_strategy"))
    if strategy == "greedy":
        return "Greedy"
    if strategy == "temperature":
        return f"Temperature {float(row['temperature']):g}"
    if strategy == "top_k":
        return f"Top-k {int(float(row['top_k']))}"
    if strategy == "top_p":
        return f"Top-p {float(row['top_p']):g}"
    return strategy


def with_decoding_label(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "config_label" in df.columns:
        return df
    labeled = df.copy()
    labeled["config_label"] = labeled.apply(decoding_label, axis=1)
    return labeled


artifact_paths = {
    "final_summary": REPORTS / f"{LEGACY_MILESTONE5_TOKEN}_final_summary.json",
    "milestone3_summary": latest_file(
        METRICS,
        f"*_{LEGACY_MILESTONE3_TOKEN}_combined_{LEGACY_MILESTONE3_TOKEN}_summary.json",
    ),
    "milestone3_regard": latest_file(
        METRICS,
        f"*_{LEGACY_MILESTONE3_TOKEN}_combined_{LEGACY_MILESTONE3_TOKEN}_regard_distributions.csv",
    ),
    "milestone3_gaps": latest_file(
        METRICS,
        f"*_{LEGACY_MILESTONE3_TOKEN}_combined_{LEGACY_MILESTONE3_TOKEN}_negative_gaps_with_ci.csv",
    ),
    "milestone3_quality": latest_file(
        METRICS,
        f"*_{LEGACY_MILESTONE3_TOKEN}_combined_{LEGACY_MILESTONE3_TOKEN}_quality_metrics_with_ci.csv",
    ),
    "masking_summary": METRICS / f"{LEGACY_MILESTONE5_TOKEN}_masking_summary.json",
    "masking_gaps": METRICS / f"{LEGACY_MILESTONE5_TOKEN}_masking_gap_comparison.csv",
    "antirep_summary": METRICS / f"{LEGACY_MILESTONE5_TOKEN}_antirep_summary.json",
    "antirep_quality": METRICS / f"{LEGACY_MILESTONE5_TOKEN}_antirep_quality_comparison.csv",
    "antirep_gaps": METRICS / f"{LEGACY_MILESTONE5_TOKEN}_antirep_gap_comparison.csv",
    "antirep_quality_plot": PLOTS / f"{LEGACY_MILESTONE5_TOKEN}_antirep_quality_delta.png",
    "antirep_gap_plot": PLOTS / f"{LEGACY_MILESTONE5_TOKEN}_antirep_gap_delta.png",
}

artifact_status = pd.DataFrame(
    [
        {"artifact": name, "path": rel(path), "exists": path is not None and path.exists()}
        for name, path in artifact_paths.items()
    ]
)
display(artifact_status)

# %% [markdown]
# ## Final Summary
#
# This cell resumes the final milestone status and the main conclusion text from the saved summary.

# %%
final_summary = read_json(artifact_paths["final_summary"])
if final_summary:
    milestone_labels = {
        "w1_complete": "Milestone 1",
        "w2_complete": "Milestone 2",
        "w3_complete": "Milestone 3",
        "w4_complete": "Milestone 4",
        "w5_antirepetition_complete": "Milestone 5 anti-repetition",
        "w5_masking_complete": "Milestone 5 masking",
    }
    completion_flags = [
        {"milestone": label, "complete": final_summary[key]}
        for key, label in milestone_labels.items()
        if key in final_summary
    ]
    display(pd.DataFrame(completion_flags))
    display(Markdown(milestone_text(final_summary.get("final_conclusion_text", ""))))
else:
    display(Markdown("No final summary found. Run the milestone metric and ablation cells."))

# %% [markdown]
# ## Milestone 3 Regard Distributions
#
# The table below reports aggregate regard label proportions by demographic and decoding
# configuration.

# %%
milestone3_regard = read_csv(artifact_paths["milestone3_regard"])
if not milestone3_regard.empty:
    milestone3_regard = with_decoding_label(milestone3_regard)
    display(
        milestone3_regard[
            ["config_label", "group", "negative", "neutral", "positive", "other", "total"]
        ].head(12)
    )
    negative_by_config = milestone3_regard.pivot_table(
        index="config_label",
        columns="group",
        values="negative",
        aggfunc="mean",
        sort=False,
    )
    display(negative_by_config.round(4))
else:
    display(Markdown("No Milestone 3 regard distribution table found."))

# %% [markdown]
# ## Milestone 3 Negative-Regard Gaps
#
# Bias gaps are computed within prompt type as `P(negative | group A) - P(negative | group B)`,
# with bootstrap confidence intervals.

# %%
milestone3_gaps = read_csv(artifact_paths["milestone3_gaps"])
if not milestone3_gaps.empty:
    milestone3_gaps = with_decoding_label(milestone3_gaps)
    selected_gap = milestone3_gaps[
        (milestone3_gaps["prompt_type"] == "description")
        & (milestone3_gaps["group_a"] == "Black man")
        & (milestone3_gaps["group_b"] == "White woman")
    ].copy()
    selected_gap = selected_gap[
        [
            "decoding_strategy",
            "config_label",
            "temperature",
            "top_k",
            "top_p",
            "gap_neg",
            "ci_lower",
            "ci_upper",
            "n_samples_a",
            "n_samples_b",
        ]
    ]
    display(selected_gap.round(4))
else:
    display(Markdown("No Milestone 3 gap table found."))

# %% [markdown]
# ## Milestone 3 Quality Controls
#
# Quality controls track diversity and degeneration so that bias shifts are not interpreted without
# considering decoding-induced length or repetition effects.

# %%
milestone3_quality = read_csv(artifact_paths["milestone3_quality"])
if not milestone3_quality.empty:
    milestone3_quality = with_decoding_label(milestone3_quality)
    quality_columns = [
        "config_label",
        "n_generations",
        "distinct_1",
        "distinct_2",
        "repeated_3gram_rate",
        "longest_repetition_span",
    ]
    display(milestone3_quality[quality_columns].round(4))
else:
    display(Markdown("No Milestone 3 quality table found."))

# %% [markdown]
# ## Milestone 5 Masking Sensitivity
#
# The masking ablation compares released regard-classifier scores with and without replacing the
# demographic mention by `XYZ`.

# %%
masking_summary = read_json(artifact_paths["masking_summary"])
masking_gaps = read_csv(artifact_paths["masking_gaps"])

if masking_summary:
    masking_key_trace = masking_summary.get("key_trace", {})
    display(
        pd.DataFrame(
            {
                "main_milestone3_conclusion_changed": [
                    masking_summary.get("main_" + LEGACY_MILESTONE3_TOKEN + "_conclusion_changed")
                ],
                "key_trace_has_sign_flip": [masking_key_trace.get("has_sign_flip")],
            }
        )
    )

if not masking_gaps.empty:
    masking_selected = masking_gaps[
        (masking_gaps["prompt_type"] == "description")
        & (masking_gaps["group_a"] == "Black man")
        & (masking_gaps["group_b"] == "White woman")
    ][["config_label", "masked_gap_neg", "unmasked_gap_neg", "delta_gap_neg", "sign_changed"]]
    display(masking_selected.round(4))
else:
    display(Markdown("No Milestone 5 masking comparison table found."))

# %% [markdown]
# ## Milestone 5 Anti-Repetition Ablation
#
# The anti-repetition ablation reruns the same decoding grid with `no_repeat_ngram_size=3`.

# %%
antirep_summary = read_json(artifact_paths["antirep_summary"])
antirep_quality = read_csv(artifact_paths["antirep_quality"])
antirep_gaps = read_csv(artifact_paths["antirep_gaps"])

if antirep_summary:
    antirep_key_trace = antirep_summary.get("key_trace", {})
    display(
        pd.DataFrame(
            {
                "main_milestone3_conclusion_changed": [
                    antirep_summary.get("main_" + LEGACY_MILESTONE3_TOKEN + "_conclusion_changed")
                ],
                "distinct2_improved_config_count": [
                    antirep_summary.get("distinct2_improved_config_count")
                ],
                "key_trace_has_sign_flip": [antirep_key_trace.get("has_sign_flip")],
            }
        )
    )

if not antirep_quality.empty:
    display(
        antirep_quality[
            [
                "config_label",
                "baseline_distinct_2",
                "antirep_distinct_2",
                "delta_distinct_2",
                "baseline_repeated_3gram_rate",
                "antirep_repeated_3gram_rate",
                "delta_repeated_3gram_rate",
            ]
        ].round(4)
    )

if not antirep_gaps.empty:
    antirep_selected = antirep_gaps[
        (antirep_gaps["prompt_type"] == "description")
        & (antirep_gaps["group_a"] == "Black man")
        & (antirep_gaps["group_b"] == "White woman")
    ][["config_label", "baseline_gap_neg", "antirep_gap_neg", "delta_gap_neg", "sign_changed"]]
    display(antirep_selected.round(4))

if antirep_quality.empty and antirep_gaps.empty:
    display(Markdown("No Milestone 5 anti-repetition comparison tables found."))

# %% [markdown]
# ## Saved Plots
#
# These plots are loaded from the generated artifacts. They are optional for execution, but useful
# for quickly checking the Milestone 5 anti-repetition effects.

# %%
for plot_name in ("antirep_quality_plot", "antirep_gap_plot"):
    plot_path = artifact_paths[plot_name]
    if plot_path is None or not plot_path.exists():
        display(Markdown(f"`{plot_name}` is missing."))
        continue

    display(Markdown(f"### {plot_name.replace('_', ' ').title()}"))
    display(Image(filename=str(plot_path)))

# %% [markdown]
# ## Submission Notes
#
# This notebook should be submitted with the repository source. The large raw `outputs/` directory is
# intentionally excluded from version control, but the notebook can resume those artifacts when they
# exist locally. To reproduce the run from scratch, execute the reproduction cells after enabling the
# relevant flags.
