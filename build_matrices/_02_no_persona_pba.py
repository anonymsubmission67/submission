"""
Pipeline step **2/5** — no-persona political questionnaire (compass + coordinates).

Unified no-persona political questionnaire matrices (compass + coordinates).

Loads server JSON and API JSONL, normalises Likert scores (1–4), flips by ``Agree``,
computes item-level means per run batch, folds ``claim_id`` variants onto numeric bases
(for compass: ``42``, ``42_opp``, ``42_ref`` → column ``42``), then writes:

  ``data/claim_matrices/likert/<suffix>_mean.csv``
  ``data/claim_matrices/likert/<suffix>_variance.csv``

All ``simple`` and ``chain_of_thought`` server/API runs are **pooled** into one aggregation per
``(model_id, claim_id)`` variant before compass-style base folding.

Cell variance columns use ddof=1 variance across variant-level pooled means for each numeric base.

Run from repo root or from ``build_matrices/`` (paths are anchored to repo root).

Public entry-point: ``process_questionnaire``.
Legacy aliases: ``process_compass``, ``process_compass_api`` (same implementation).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from utils.claim_matrix_roots import CLAIM_MATRICES
from freetext_model_map import (
    has_server_json,
    reindex_matrices_to_target_llms,
    require_target_llm_coverage,
    target_llms,
    API_MODEL_MAPPING
)

REPO_ROOT = Path(__file__).resolve().parents[1]

LIKERT_MAP: dict[str, int] = {
    "strongly disagree": 1,
    "disagree": 2,
    "agree": 3,
    "strongly agree": 4,
}

ALLOWED_SCORES = {1.0, 2.0, 3.0, 4.0}
ALLOWED_FREE_TEXT_SCORES = {1.0, 2.0, 3.0}

_QUESTIONNAIRE_PATHS: dict[str, dict[str, Any]] = {
    "compass": {
        "statements_csv": REPO_ROOT / "data/political_compass_statements.csv",
        "server_subdir": "compass",
        "output_suffix": "compass",
        "api_prompting_segment": "compass",
    },
    "coordinates": {
        "statements_csv": REPO_ROOT / "data/political_coordinates_statements.csv",
        "server_subdir": "coordinates",
        "output_suffix": "coordinates",
        "api_prompting_segment": "coordinates",
    },
}


API_MODEL_IDS = list(API_MODEL_MAPPING.keys())



COVERAGE_OUTPUT = REPO_ROOT / "output/tables/questionnaire_run_coverage.csv"


def numeric_base(cid: Any) -> str:
    """Strip compass variant suffix → output column label (still string)."""
    s = str(cid).strip()
    if s.endswith("_opp"):
        return s[:-4]
    if s.endswith("_ref"):
        return s[:-4]
    return s


def _sort_base_keys(bases: Iterable[str]) -> list[str]:
    def key(b: str) -> tuple[int, str]:
        try:
            return (0, f"{int(b):010d}_{b}")
        except ValueError:
            return (1, b)

    return sorted(set(bases), key=key)


def flip_score(score: float, agree_dir: str) -> float:
    """Flip Likert scores (1–4) for statements where ``Agree == Left``."""
    if pd.isna(score) or score not in ALLOWED_SCORES:
        return float("nan")
    if agree_dir != "Left":
        return float(score)
    m = {1.0: 4.0, 2.0: 3.0, 3.0: 2.0, 4.0: 1.0}
    return float(m[float(score)])


def flip_score_free_text(score: float, agree_dir: str) -> float:
    """Flip free-text scores (1–3) for statements where ``Agree == Left``."""
    if pd.isna(score) or score not in ALLOWED_FREE_TEXT_SCORES:
        return float("nan")
    if agree_dir != "Left":
        return float(score)
    m = {1.0: 3.0, 2.0: 2.0, 3.0: 1.0}
    return float(m[float(score)])


def load_statement_agree(csv_path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    df = pd.read_csv(csv_path)
    df["claim_id"] = df["claim_id"].astype(str)
    agree = {row["claim_id"]: row["Agree"] for _, row in df.iterrows()}
    return df, agree


def agreement_from_run_text(run_txt: Any) -> str | None:
    """Parse free-text JSON / patterns for questionnaire ``agreement`` (server runs)."""
    text = str(run_txt)
    json_pattern = r"\{[^}]*\"agreement\"[^}]*\}"
    for json_match in re.findall(json_pattern, text):
        try:
            blob = json.loads(json_match)
            raw_ag = blob.get("agreement", "")
            ag = raw_ag.strip().lower() if isinstance(raw_ag, str) else ""
            if ag in LIKERT_MAP:
                return ag
        except json.JSONDecodeError:
            continue
    patterns = [
        r'"agreement"\s*:\s*"([^"]+)"',
        r"agreement\s*:\s*\"([^\"]+)\"",
        r"agreement\s*:\s*([a-z\s]+)",
    ]
    for pat in patterns:
        for m in re.findall(pat, text, re.IGNORECASE):
            ag = str(m).strip().lower()
            if ag in LIKERT_MAP:
                return ag
    return None


def _records_from_server_model(data_dir: Path, model_name: str) -> list[dict]:
    combined: list[dict] = []
    for name in (f"{model_name}.json", f"{model_name}_additional.json"):
        p = data_dir / name
        if p.exists():
            with p.open(encoding="utf-8") as f:
                combined.extend(json.load(f))
    return combined


def _model_has_api_questionnaire(
    model_id: str,
    cfg: dict[str, Any],
    prompt_styles: list[str],
) -> bool:
    segment = cfg["api_prompting_segment"]
    for src_model_id in API_MODEL_IDS:
        if API_MODEL_MAPPING[src_model_id] != model_id:
            continue
        for prompt_style in prompt_styles:
            model_dir = REPO_ROOT / "data/api_outputs" / prompt_style / segment / src_model_id
            if model_dir.is_dir() and any(model_dir.glob("baseline*.jsonl")):
                return True
    return False


def long_rows_server(
    prompt_style: str,
    cfg: dict[str, Any],
    agree_map: dict[str, str],
) -> list[dict]:
    rows: list[dict] = []
    data_dir = REPO_ROOT / "data/server_outputs" / prompt_style / cfg["server_subdir"]
    if not data_dir.exists():
        return rows

    for model in target_llms:
        if not has_server_json(data_dir, model):
            continue
        for record in _records_from_server_model(data_dir, model):
            sid_raw = record.get("id")
            if sid_raw is None:
                continue
            claim_id = str(sid_raw).strip()
            if claim_id not in agree_map:
                continue
            for run_txt in record.get("runs") or []:
                ag = agreement_from_run_text(run_txt)
                if ag is None:
                    continue
                rows.append(
                    {
                        "source": "server",
                        "model_id": model,
                        "claim_id": claim_id,
                        "score_raw_label": LIKERT_MAP[ag],
                    }
                )
    return rows


def long_rows_api(
    prompt_style: str,
    cfg: dict[str, Any],
    agree_map: dict[str, str],
) -> list[dict]:
    rows: list[dict] = []
    segment = cfg["api_prompting_segment"]
    root = REPO_ROOT / "data/api_outputs" / prompt_style

    if not root.exists():
        return rows

    for src_model_id in API_MODEL_IDS:
        out_id = API_MODEL_MAPPING[src_model_id]
        if out_id not in target_llms:
            continue
        model_dir = root / segment / src_model_id
        if not model_dir.exists():
            continue
        baseline_files = list(model_dir.glob("baseline*.jsonl"))
        for fp in baseline_files:
            with fp.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    cid = rec.get("Claim ID", rec.get("claim_id"))
                    ag_raw = rec.get("agreement")
                    if cid is None or ag_raw is None:
                        continue
                    claim_id = str(cid).strip()
                    agreement = ag_raw.strip().lower()
                    if agreement not in LIKERT_MAP or claim_id not in agree_map:
                        continue
                    rows.append(
                        {
                            "source": "api",
                            "model_id": out_id,
                            "claim_id": claim_id,
                            "score_raw_label": LIKERT_MAP[agreement],
                        }
                    )
    return rows


def apply_flip_and_filter(long_rows: list[dict], agree_map: dict[str, str]) -> pd.DataFrame:
    if not long_rows:
        return pd.DataFrame(
            columns=["source", "model_id", "claim_id", "score"]
        )
    df = pd.DataFrame(long_rows)
    scores: list[float] = []
    for _, r in df.iterrows():
        agdir = agree_map.get(str(r["claim_id"]), "Right")
        scores.append(flip_score(float(r["score_raw_label"]), agdir))
    df["score"] = scores
    df = df[np.isfinite(df["score"])].drop(columns=["score_raw_label"])
    return df


def pooled_item_mean_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """One pooled mean score per ``(model_id, claim_variant_id)`` over all loaded runs/sources."""
    if long_df.empty:
        return pd.DataFrame()
    g = (
        long_df.groupby(["model_id", "claim_id"], as_index=False)["score"]
        .mean()
        .rename(columns={"score": "item_mean"})
    )
    piv = g.pivot(index="model_id", columns="claim_id", values="item_mean")
    piv = piv.sort_index(axis=0)
    piv = piv.sort_index(axis=1, key=lambda ix: ix.map(lambda c: numeric_base(str(c))))
    return piv


def members_by_numeric_base(stmt_df: pd.DataFrame) -> dict[str, list[str]]:
    cid_list = stmt_df["claim_id"].astype(str).tolist()
    groups: dict[str, list[str]] = defaultdict(list)
    for cid in cid_list:
        groups[numeric_base(cid)].append(str(cid))

    def _memb_order(x: str) -> tuple[int, str]:
        if "_" not in x:
            return (0, x)
        if x.endswith("_opp"):
            return (1, x)
        if x.endswith("_ref"):
            return (2, x)
        return (3, x)

    out: dict[str, list[str]] = {}
    for base, memb in groups.items():
        out[base] = sorted(set(memb), key=_memb_order)
    return out


def collapse_to_bases(mean_wide: pd.DataFrame, members: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mean_wide.empty:
        return pd.DataFrame(), pd.DataFrame()
    bases = _sort_base_keys(members)
    cols_m: dict[str, pd.Series] = {}
    cols_v: dict[str, pd.Series] = {}
    for base in bases:
        memb = [c for c in members.get(base, []) if c in mean_wide.columns]
        if not memb:
            continue
        sub = mean_wide.loc[:, memb].apply(pd.to_numeric, errors="coerce").astype(float)
        cols_m[base] = sub.mean(axis=1, skipna=True)
        v = sub.var(axis=1, ddof=1)
        cols_v[base] = v.fillna(0.0)
    out_mean = pd.DataFrame(cols_m, index=mean_wide.index).sort_index(axis=1)
    out_var = pd.DataFrame(cols_v, index=mean_wide.index).sort_index(axis=1)
    return out_mean, out_var


def append_coverage(counter_rows: list[dict], questionnaire: str, long_df: pd.DataFrame) -> None:
    if long_df.empty:
        return
    g = long_df.groupby(["model_id", "claim_id"]).size().reset_index(name="n_valid_runs")
    for _, r in g.iterrows():
        counter_rows.append(
            {
                "questionnaire": questionnaire,
                "model_id": str(r["model_id"]),
                "claim_id": str(r["claim_id"]),
                "n_valid_runs": int(r["n_valid_runs"]),
            }
        )


def _write_matrix_pair(path_mean: Path, path_var: Path, mean_df: pd.DataFrame, var_df: pd.DataFrame) -> None:
    path_mean.parent.mkdir(parents=True, exist_ok=True)
    mean_df.sort_index(inplace=True)
    var_df.sort_index(inplace=True)
    mean_df.sort_index(axis=1, inplace=True)
    var_df.sort_index(axis=1, inplace=True)
    mean_df.to_csv(path_mean)
    var_df.to_csv(path_var)


def build_one_questionnaire(
    questionnaire: str,
    prompt_styles: list[str],
    coverage_accum: list[dict],
    *,
    fail_if_empty: bool = False,
) -> None:
    cfg = _QUESTIONNAIRE_PATHS[questionnaire]
    stmt_df, agree_map = load_statement_agree(cfg["statements_csv"])
    memb = members_by_numeric_base(stmt_df)
    suf = cfg["output_suffix"]
    context = f"PBA {questionnaire}"

    require_target_llm_coverage(
        context=context,
        repo_root=REPO_ROOT,
        prompt_styles=prompt_styles,
        server_subdir=cfg["server_subdir"],
        api_available=lambda model: _model_has_api_questionnaire(model, cfg, prompt_styles),
    )

    merged_records: list[dict] = []
    for ps in prompt_styles:
        merged_records.extend(long_rows_server(ps, cfg, agree_map))
        merged_records.extend(long_rows_api(ps, cfg, agree_map))

    long_df = apply_flip_and_filter(merged_records, agree_map)
    if long_df.empty:
        raise RuntimeError(f"{context}: no valid questionnaire rows after pooling")
    append_coverage(coverage_accum, questionnaire, long_df)

    piv = pooled_item_mean_wide(long_df)
    collapsed_m, collapsed_v = collapse_to_bases(piv, memb)
    collapsed_m, collapsed_v = reindex_matrices_to_target_llms(collapsed_m, collapsed_v)

    out_m = REPO_ROOT / f"data/claim_matrices/likert/{suf}_mean.csv"
    out_v = REPO_ROOT / f"data/claim_matrices/likert/{suf}_variance.csv"
    _write_matrix_pair(out_m, out_v, collapsed_m, collapsed_v)
    print(f"  [{questionnaire}] pooled → {out_m} ({collapsed_m.shape})")


def process_no_persona_questionnaire(
    prompt_styles: list[str] | None = None,
    questionnaire: str | None = None,
) -> None:
    """
    Build unified matrices.

    questionnaire: ``'compass'`` | ``'coordinates'`` — if omitted, both.
    prompt_styles: list e.g. ``['simple','chain_of_thought']``; defaults to dirs under ``data/api_outputs``.
    """

    api_root = REPO_ROOT / "data/api_outputs"
    if prompt_styles is None:
        prompt_styles = []
        if (api_root / "simple").exists():
            prompt_styles.append("simple")
        if (api_root / "chain_of_thought").exists():
            prompt_styles.append("chain_of_thought")
        if not prompt_styles:
            prompt_styles = ["simple", "chain_of_thought"]

    runs = ["compass", "coordinates"] if questionnaire is None else [questionnaire]

    coverage_accum: list[dict] = []
    for q in runs:
        print(f"\n{'=' * 72}\nquestionnaire={q!r}\n{'=' * 72}")
        build_one_questionnaire(q, prompt_styles, coverage_accum)

    cov_df = pd.DataFrame(coverage_accum)
    if not cov_df.empty:
        cov_df.sort_values(["questionnaire", "model_id", "claim_id"], inplace=True)
        COVERAGE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        cov_df.to_csv(COVERAGE_OUTPUT, index=False)
        print(f"\nCoverage table → {COVERAGE_OUTPUT}")
    print("\nDone ( questionnaire).")


# --------------------------------------------------------------------------- Aliases / CLI
process_no_persona_compass = process_no_persona_questionnaire
process_no_persona_compass_api = process_no_persona_questionnaire

if __name__ == "__main__":
    process_no_persona_questionnaire()
