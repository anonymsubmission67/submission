"""Helpers for no_persona PBA: load questionnaire/free-text matrices and derive pol_bias / 2D compass coords."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

def resolve_matrix_row(df: pd.DataFrame, llm: str, free_text: bool) -> Optional[str]:
    """Resolve row label if ``llm`` is indexed; short ids only (normalization lives in build_matrices)."""
    del free_text  # API kept for callers
    if llm in df.index:
        return llm
    return None


_LIKERT_MATRIX_DIR = Path("data/claim_matrices/likert")
_FREE_TEXT_MATRIX_DIR = Path("data/claim_matrices/free_text")


def load_questionnaire_concat(suffix: str) -> Optional[pd.DataFrame]:
    """suffix: 'compass' | 'coordinates' — likert matrix under ``data/claim_matrices/likert/``."""
    p = _LIKERT_MATRIX_DIR / f"{suffix}_mean.csv"
    if p.is_file():
        return pd.read_csv(p, index_col=0)
    return None


def load_free_text_matrix(subdir: str) -> Optional[pd.DataFrame]:
    """subdir: 'compass' | 'coordinates'."""
    p = _FREE_TEXT_MATRIX_DIR / f"{subdir}_mean.csv"
    if p.is_file():
        return pd.read_csv(p, index_col=0)
    return None


def load_free_text_variance_matrix(subdir: str) -> Optional[pd.DataFrame]:
    """Companion variance matrix (same model rows as mean) for coverage checks."""
    p = _FREE_TEXT_MATRIX_DIR / f"{subdir}_variance.csv"
    if p.is_file():
        return pd.read_csv(p, index_col=0)
    return None


def warn_free_text_model_coverage_three_matrices(
    canonical_llms: list[str],
    *,
    expected_row_count: int | None = None,
) -> None:
    """
    Print WARNING when any of three free-text tables does not match the expected model rows.

    Checks: compass mean, coordinates mean, compass variance.
    ``expected_row_count`` defaults to ``len(canonical_llms)``.
    """
    bundles: tuple[tuple[str, Optional[pd.DataFrame]], ...] = (
        ("free_text/compass_mean.csv", load_free_text_matrix("compass")),
        ("free_text/coordinates_mean.csv", load_free_text_matrix("coordinates")),
        ("free_text/compass_variance.csv", load_free_text_variance_matrix("compass")),
    )
    expected = expected_row_count if expected_row_count is not None else len(canonical_llms)

    def _inspect(df: pd.DataFrame) -> tuple[int, list[str]]:
        missing: list[str] = []
        for llm in canonical_llms:
            if resolve_matrix_row(df, llm, free_text=True) is None:
                missing.append(llm)
        return len(df.index), missing

    for path_label, df in bundles:
        if df is None or df.empty:
            print(
                f"WARNING: [{path_label}] missing or empty matrix "
                f"(expected_row_count={expected})."
            )
            continue
        n_ix, missing_llms = _inspect(df)
        if n_ix != expected:
            print(
                f"WARNING: [{path_label}] row count={n_ix} ≠ expected={expected}; "
                f"index_labels={list(map(str, df.index.tolist()))}"
            )
        if missing_llms:
            print(
                f"WARNING: [{path_label}] missing matrix row for target_llms IDs {missing_llms}."
            )


def _claim_domain_map(statements_csv: str) -> dict[str, str]:
    df = pd.read_csv(statements_csv)
    return {str(r["claim_id"]): r["Domain"] for _, r in df.iterrows()}


def _scale_value(raw, *, free_text: bool) -> float | None:
    """Map one raw score to [-1, 1]; None if missing/non-finite."""
    val = pd.to_numeric(raw, errors="coerce")
    if not np.isfinite(val):
        return None
    scaled = _scale_free_text_to_unit(float(val)) if free_text else _scale_likert_to_unit(float(val))
    return float(np.clip(scaled, -1.0, 1.0))


def pol_bias_1d(df: Optional[pd.DataFrame], llm: str, *, free_text: bool) -> float:
    """Scalar pol_bias on [-1, 1] scale (all statements); nan if missing."""
    if df is None or df.empty:
        return np.nan
    row = resolve_matrix_row(df, llm, free_text)
    if row is None:
        return np.nan
    scaled = [
        s
        for col in _statement_columns(df)
        if (s := _scale_value(df.loc[row, col], free_text=free_text)) is not None
    ]
    return float(np.mean(scaled)) if scaled else np.nan


def pol_bias_axes(
    df: Optional[pd.DataFrame],
    llm: str,
    statements_csv: str,
    *,
    free_text: bool,
) -> dict[str, float]:
    """PB-A on economic, social, and both axes ([-1, 1]); nan if unavailable."""
    nan_axes = {"economic": np.nan, "social": np.nan, "both": np.nan}
    if df is None or df.empty:
        return nan_axes
    row = resolve_matrix_row(df, llm, free_text)
    if row is None:
        return nan_axes

    domain_by_claim = _claim_domain_map(statements_csv)
    eco_vals: list[float] = []
    soc_vals: list[float] = []
    all_vals: list[float] = []

    for col in _statement_columns(df):
        scaled = _scale_value(df.loc[row, col], free_text=free_text)
        if scaled is None:
            continue
        all_vals.append(scaled)
        dom = domain_by_claim.get(str(col).strip())
        if dom == "Economic":
            eco_vals.append(scaled)
        elif dom == "Social":
            soc_vals.append(scaled)

    return {
        "economic": float(np.mean(eco_vals)) if eco_vals else np.nan,
        "social": float(np.mean(soc_vals)) if soc_vals else np.nan,
        "both": float(np.mean(all_vals)) if all_vals else np.nan,
    }


def _statement_columns(df: pd.DataFrame) -> list[str]:
    """Column labels excluding ``model_id``."""
    return [
        c
        for c in df.columns
        if c != "model_id" and not (isinstance(c, str) and str(c).lower() == "model_id")
    ]


def merge_compass_coordinates(
    df_compass: Optional[pd.DataFrame],
    df_coordinates: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """Concatenate compass + coordinates with prefixed statement keys (``compass:1``, ``coordinates:1``)."""
    frames: list[pd.DataFrame] = []
    for df, prefix in ((df_compass, "compass"), (df_coordinates, "coordinates")):
        if df is None or df.empty:
            continue
        col_map = _non_model_columns(df)
        sub = df.loc[:, [col_map[k] for k in sorted(col_map)]].copy()
        sub.columns = [f"{prefix}:{k}" for k in sorted(col_map)]
        frames.append(sub)
    if not frames:
        return None
    return pd.concat(frames, axis=1)


def _scale_likert_to_unit(value: float) -> float:
    """Map Likert scores on [1, 4] to [-1, 1]."""
    return float((value - 2.5) / 1.5)


def _scale_free_text_to_unit(value: float) -> float:
    """Map free-text scores on [1, 3] to [-1, 1]."""
    return float(value - 2.0)


def _scaled_or_neutral(raw, *, free_text: bool) -> float:
    """Missing/non-finite values become neutral (0 on [-1, 1])."""
    val = pd.to_numeric(raw, errors="coerce")
    if not np.isfinite(val):
        return 0.0
    scaled = _scale_free_text_to_unit(float(val)) if free_text else _scale_likert_to_unit(float(val))
    return float(np.clip(scaled, -1.0, 1.0))


def _merged_domain_map() -> dict[str, str]:
    """Prefixed statement keys (``compass:1``, ``coordinates:1``) -> Economic / Social."""
    compass = _claim_domain_map("data/political_compass_statements.csv")
    coordinates = _claim_domain_map("data/political_coordinates_statements.csv")
    out: dict[str, str] = {}
    for key, domain in compass.items():
        out[f"compass:{key}"] = domain
    for key, domain in coordinates.items():
        out[f"coordinates:{key}"] = domain
    return out


def _combined_statement_values(
    llm: str,
    df_likert_merged: Optional[pd.DataFrame],
    df_free_text_merged: Optional[pd.DataFrame],
) -> dict[str, float] | None:
    """Per-statement combined PB-A (Likert + free text averaged, missing -> neutral 0)."""
    if (df_likert_merged is None or df_likert_merged.empty) and (
        df_free_text_merged is None or df_free_text_merged.empty
    ):
        return None

    row_l = resolve_matrix_row(df_likert_merged, llm, free_text=False) if df_likert_merged is not None else None
    row_f = resolve_matrix_row(df_free_text_merged, llm, free_text=True) if df_free_text_merged is not None else None
    if row_l is None and row_f is None:
        return None

    all_cols: set[str] = set()
    if df_likert_merged is not None and not df_likert_merged.empty:
        all_cols.update(_statement_columns(df_likert_merged))
    if df_free_text_merged is not None and not df_free_text_merged.empty:
        all_cols.update(_statement_columns(df_free_text_merged))
    if not all_cols:
        return None

    out: dict[str, float] = {}
    for col in sorted(all_cols):
        raw_l = df_likert_merged.loc[row_l, col] if row_l is not None and col in df_likert_merged.columns else np.nan
        raw_f = (
            df_free_text_merged.loc[row_f, col] if row_f is not None and col in df_free_text_merged.columns else np.nan
        )
        scaled_l = _scaled_or_neutral(raw_l, free_text=False)
        scaled_f = _scaled_or_neutral(raw_f, free_text=True)
        out[col] = (scaled_l + scaled_f) / 2.0
    return out


def pol_bias_combined(
    llm: str,
    df_likert_merged: Optional[pd.DataFrame],
    df_free_text_merged: Optional[pd.DataFrame],
) -> float:
    """
    PB-A on merged compass+coordinates, averaging scaled Likert and free-text per statement.

    Each modality is mapped to [-1, 1] (Likert 1–4, free text 1–3); missing values are neutral (0).
    """
    per_statement = _combined_statement_values(llm, df_likert_merged, df_free_text_merged)
    if not per_statement:
        return np.nan
    return float(np.mean(list(per_statement.values())))


def pol_bias_combined_axes(
    llm: str,
    df_likert_merged: Optional[pd.DataFrame],
    df_free_text_merged: Optional[pd.DataFrame],
) -> dict[str, float]:
    """Combined PB-A on economic, social, and both axes across compass + coordinates."""
    nan_axes = {"economic": np.nan, "social": np.nan, "both": np.nan}
    per_statement = _combined_statement_values(llm, df_likert_merged, df_free_text_merged)
    if not per_statement:
        return nan_axes

    domain_map = _merged_domain_map()
    eco_vals: list[float] = []
    soc_vals: list[float] = []
    all_vals = list(per_statement.values())

    for col, val in per_statement.items():
        dom = domain_map.get(col)
        if dom == "Economic":
            eco_vals.append(val)
        elif dom == "Social":
            soc_vals.append(val)

    return {
        "economic": float(np.mean(eco_vals)) if eco_vals else np.nan,
        "social": float(np.mean(soc_vals)) if soc_vals else np.nan,
        "both": float(np.mean(all_vals)) if all_vals else np.nan,
    }


def economic_social_merged_mapped(
    df: Optional[pd.DataFrame],
    llm: str,
    *,
    free_text: bool,
) -> tuple[float, float]:
    """Economic (x) and Social (y) on merged compass+coordinates; missing statements dropped."""
    if df is None or df.empty:
        return np.nan, np.nan
    row = resolve_matrix_row(df, llm, free_text)
    if row is None:
        return np.nan, np.nan
    domain_map = _merged_domain_map()
    eco_scaled: list[float] = []
    soc_scaled: list[float] = []
    for col in _statement_columns(df):
        scaled = _scale_value(df.loc[row, col], free_text=free_text)
        if scaled is None:
            continue
        dom = domain_map.get(str(col).strip())
        if dom == "Economic":
            eco_scaled.append(scaled)
        elif dom == "Social":
            soc_scaled.append(scaled)
    eco_m = float(np.mean(eco_scaled)) if eco_scaled else np.nan
    soc_m = float(np.mean(soc_scaled)) if soc_scaled else np.nan
    return eco_m, soc_m


def economic_social_combined_mapped(
    llm: str,
    df_likert_merged: Optional[pd.DataFrame],
    df_free_text_merged: Optional[pd.DataFrame],
) -> tuple[float, float]:
    """Economic (x) and Social (y) for combined PB-A compass plot."""
    axes = pol_bias_combined_axes(llm, df_likert_merged, df_free_text_merged)
    return axes["economic"], axes["social"]


def _non_model_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map stripped claim key -> actual column label in ``df`` (first occurrence)."""
    out: dict[str, str] = {}
    for c in df.columns:
        if c == "model_id" or (isinstance(c, str) and str(c).lower() == "model_id"):
            continue
        sk = str(c).strip()
        if sk not in out:
            out[sk] = c
    return out


def pol_bias_1d_overlap_pair(
    df_questionnaire: Optional[pd.DataFrame],
    df_free_text: Optional[pd.DataFrame],
    llm: str,
) -> tuple[float, float]:
    """(pol_bias_questionnaire, pol_bias_free_text) on [-1, 1]; only claims finite in BOTH matrices."""
    if df_questionnaire is None or df_questionnaire.empty or df_free_text is None or df_free_text.empty:
        return np.nan, np.nan

    rq = resolve_matrix_row(df_questionnaire, llm, free_text=False)
    rf = resolve_matrix_row(df_free_text, llm, free_text=True)
    if rq is None or rf is None:
        return np.nan, np.nan

    map_q = _non_model_columns(df_questionnaire)
    map_f = _non_model_columns(df_free_text)
    common_keys = sorted(set(map_q) & set(map_f))

    vals_q_list: list[float] = []
    vals_f_list: list[float] = []
    for ck in common_keys:
        cq, cf = map_q[ck], map_f[ck]
        try:
            a = pd.to_numeric(df_questionnaire.loc[rq, cq], errors="coerce")
            b = pd.to_numeric(df_free_text.loc[rf, cf], errors="coerce")
        except (KeyError, TypeError):
            continue
        aq = float(a)
        bf = float(b)
        if np.isfinite(aq) and np.isfinite(bf):
            vals_q_list.append(aq)
            vals_f_list.append(bf)

    if not vals_q_list:
        return np.nan, np.nan

    sq = float(np.mean([_scale_likert_to_unit(v) for v in vals_q_list]))
    sf = float(np.mean([_scale_free_text_to_unit(v) for v in vals_f_list]))
    return sq, sf


def economic_social_mapped(
    df: Optional[pd.DataFrame],
    llm: str,
    statements_csv: str,
    *,
    free_text: bool,
) -> tuple[float, float]:
    """Economic (x) and Social (y) on mapped [-1, 1] scale; nan if unavailable."""
    if df is None or df.empty:
        return np.nan, np.nan
    row = resolve_matrix_row(df, llm, free_text)
    if row is None:
        return np.nan, np.nan
    domain_by_claim = _claim_domain_map(statements_csv)
    eco_cols = []
    soc_cols = []
    for c in df.columns:
        if c == "model_id" or (isinstance(c, str) and c.lower() == "model_id"):
            continue
        key = str(c).strip()
        dom = domain_by_claim.get(key)
        if dom == "Economic":
            eco_cols.append(c)
        elif dom == "Social":
            soc_cols.append(c)
    eco_scaled = [
        s
        for c in eco_cols
        if (s := _scale_value(df.loc[row, c], free_text=free_text)) is not None
    ]
    soc_scaled = [
        s
        for c in soc_cols
        if (s := _scale_value(df.loc[row, c], free_text=free_text)) is not None
    ]
    eco_m = float(np.mean(eco_scaled)) if eco_scaled else np.nan
    soc_m = float(np.mean(soc_scaled)) if soc_scaled else np.nan
    return eco_m, soc_m


def aggregated_point(eco_vals: list[float], soc_vals: list[float]) -> tuple[float, float, float, float]:
    """Returns x_mean, y_mean, x_err (sqrt nanvar ddof=1), y_err; err 0 if single or undefined."""
    ea = np.asarray(eco_vals, dtype=float)
    sa = np.asarray(soc_vals, dtype=float)
    x = float(np.nanmean(ea))
    y = float(np.nanmean(sa))
    n_e = np.sum(~np.isnan(ea))
    n_s = np.sum(~np.isnan(sa))
    xe = float(np.sqrt(np.nanvar(ea, ddof=1))) if n_e > 1 else 0.0
    ye = float(np.sqrt(np.nanvar(sa, ddof=1))) if n_s > 1 else 0.0
    if np.isnan(xe):
        xe = 0.0
    if np.isnan(ye):
        ye = 0.0
    return x, y, xe, ye


MODEL_METRICS_PATH = Path("data/interim_results/model_metrics.csv")

PBA_BASE_KEYS = (
    "pba_l_compass",
    "pba_l_coordinates",
    "pba_ft_compass",
    "pba_ft_coordinates",
    "pba_combined",
)
AXIS_SUFFIXES = ("economic", "social", "both")

PBA_METRIC_COLUMNS = tuple(
    f"{base}_{axis}" for base in PBA_BASE_KEYS for axis in AXIS_SUFFIXES
)

PBT_F1_METRIC_COLUMNS = (
    "f1_1",
    "f1_2",
    "f1_3",
    *(
        f"pbt_{prompt}_{axis}"
        for prompt in ("1", "2", "3")
        for axis in AXIS_SUFFIXES
    ),
)

REGRESSION_METRIC_COLUMNS = ("speaker_R",)

OVERVIEW_LATEX_LABELS: dict[str, str] = {
    "pbt_1_both": r"PB-T$_1$",
    "pbt_2_both": r"PB-T$_2$",
    "pbt_3_both": r"PB-T$_3$",
    "pba_l_compass_both": r"PB-A$_\mathrm{L,C}$",
    "pba_l_coordinates_both": r"PB-A$_\mathrm{L,Co}$",
    "pba_ft_compass_both": r"PB-A$_\mathrm{FT,C}$",
    "pba_ft_coordinates_both": r"PB-A$_\mathrm{FT,Co}$",
    "pba_combined_both": r"PB-A$_\mathrm{comb.}$",
    "speaker_R": "claim maker rel.",
}


def overview_latex_columns() -> tuple[tuple[str, str], ...]:
    """LaTeX overview_table: combined axis (_both) only; CSV keeps all axes and F1."""
    keys: list[str] = []
    keys.extend(f"pbt_{prompt}_both" for prompt in ("1", "2", "3"))
    keys.extend(f"{base}_both" for base in PBA_BASE_KEYS)
    keys.append("speaker_R")
    return tuple((key, OVERVIEW_LATEX_LABELS[key]) for key in keys)

PBA_CORR_COLUMNS = tuple(f"{base}_both" for base in PBA_BASE_KEYS[:4])

PBA_CORR_LABELS = {
    "pba_l_compass_both": "Likert Compass",
    "pba_l_coordinates_both": "Likert Coordinates",
    "pba_ft_compass_both": "Free Text Compass",
    "pba_ft_coordinates_both": "Free Text Coordinates",
}

CORR_MATRIX_COLUMNS = (
    "pbt_1_both",
    "pbt_2_both",
    *PBA_CORR_COLUMNS,
)

CORR_MATRIX_LABELS = {
    "pbt_1_both": "agnostic",
    "pbt_2_both": "aware",
    **PBA_CORR_LABELS,
}

# (outer group label, [(column key, inner axis label), ...])
CORR_MATRIX_AXIS_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("PB-T", [("pbt_1_both", "agnostic"), ("pbt_2_both", "aware")]),
    ("Likert", [("pba_l_compass_both", "COM"), ("pba_l_coordinates_both", "COO")]),
    ("Free Text", [("pba_ft_compass_both", "COM"), ("pba_ft_coordinates_both", "COO")]),
]


def load_model_metrics() -> pd.DataFrame:
    if not MODEL_METRICS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {MODEL_METRICS_PATH}; run rq2_pba_pbt/_01_run_data_preperation.py first."
        )
    return pd.read_csv(MODEL_METRICS_PATH)


def pba_l_mean(row: pd.Series) -> float:
    """Average PB-A from Likert compass and coordinates ([-1, 1] scale, both axes)."""
    values = pd.to_numeric(
        row[["pba_l_compass_both", "pba_l_coordinates_both"]], errors="coerce"
    )
    return float(values.mean(skipna=True))


def pba_ft_mean(row: pd.Series) -> float:
    """Average PB-A from free-text compass and coordinates ([-1, 1] scale, both axes)."""
    values = pd.to_numeric(
        row[["pba_ft_compass_both", "pba_ft_coordinates_both"]], errors="coerce"
    )
    return float(values.mean(skipna=True))
