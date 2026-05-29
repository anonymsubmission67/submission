"""Build no_persona model metrics (PBT, F1, PB-A) and export CSV and LaTeX table."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

script_dir = Path(__file__).resolve().parent
evaluation_dir = script_dir.parent
if str(evaluation_dir) not in sys.path:
    sys.path.insert(0, str(evaluation_dir))

from rq1_no_persona._04_build_claim_maker_regression import load_regression_coefficients
from utils import llm_info, pbt_mean_path, target_llms
from .utils import (
    AXIS_SUFFIXES,
    MODEL_METRICS_PATH,
    PBA_METRIC_COLUMNS,
    PBT_F1_METRIC_COLUMNS,
    REGRESSION_METRIC_COLUMNS,
    merge_compass_coordinates,
    load_free_text_matrix,
    load_questionnaire_concat,
    overview_latex_columns,
    pol_bias_axes,
    pol_bias_combined_axes,
)

LABEL_TO_NUM = {
    "pants-fire": 0,
    "false": 1,
    "mostly-false": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5,
}

PROMPTS = ("1", "2", "3")


def _round_metric(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), 3)


def _me_diff_by_axis(df_long_with_party: pd.DataFrame, llm: str, axis: str) -> float | None:
    if axis == "both":
        subset = df_long_with_party[df_long_with_party["axis"].isin(["economic", "social"])]
    else:
        subset = df_long_with_party[df_long_with_party["axis"] == axis]
    if subset.empty:
        return None
    bias = subset[llm] - subset["label"]
    bias_by_party = bias.groupby(subset["party"]).mean()
    me_diff = bias_by_party.get("Republican", np.nan) - bias_by_party.get("Democrat", np.nan)
    return _round_metric(me_diff)


def _compute_pbt_f1(
    llm: str,
    prompt: str,
    *,
    claims: pd.DataFrame,
) -> tuple[dict[str, float | None], float | None]:
    pooled_path = pbt_mean_path(prompt)
    if not pooled_path.is_file():
        print(f"    Warning: Missing {pooled_path}")
        return {axis: None for axis in AXIS_SUFFIXES}, None

    df_long = pd.read_csv(pooled_path, index_col=0)
    if llm not in df_long.index:
        print(f"    Warning: {llm} not found in prompt {prompt} data")
        return {axis: None for axis in AXIS_SUFFIXES}, None

    llm_predictions = df_long.loc[llm].dropna()
    df_long_with_labels = df_long.T.merge(
        claims[["label"]], left_index=True, right_index=True, how="left"
    )
    df_long_with_labels["label"] = df_long_with_labels["label"].map(LABEL_TO_NUM)
    true_labels = df_long_with_labels.loc[llm_predictions.index, "label"].dropna()

    common_indices = llm_predictions.index.intersection(true_labels.index)
    if len(common_indices) == 0:
        print(f"    Warning: No common indices for {llm} in prompt {prompt}")
        return {axis: None for axis in AXIS_SUFFIXES}, None

    aligned_predictions = llm_predictions.loc[common_indices]
    aligned_true_labels = true_labels.loc[common_indices]

    df_long_with_party = df_long.T.merge(
        claims[["party", "label", "axis"]], left_index=True, right_index=True, how="left"
    )
    df_long_with_party["label"] = df_long_with_party["label"].map(LABEL_TO_NUM)
    df_long_with_party[llm] = pd.to_numeric(df_long_with_party[llm], errors="coerce")

    pbt_by_axis = {axis: _me_diff_by_axis(df_long_with_party, llm, axis) for axis in AXIS_SUFFIXES}

    rounded_predictions = np.clip(np.round(aligned_predictions).astype(int), 0, 5)
    f1 = f1_score(aligned_true_labels, rounded_predictions, average="macro")

    return pbt_by_axis, _round_metric(f1)


def _store_pba_axes(row: dict, base_key: str, axes: dict[str, float]) -> None:
    for axis in AXIS_SUFFIXES:
        row[f"{base_key}_{axis}"] = _round_metric(axes.get(axis))


def _build_model_rows(llms: list[str]) -> list[dict]:
    claims = pd.read_csv("data/claims_metadata.csv").set_index("claim_id")
    df_l_compass = load_questionnaire_concat("compass")
    df_l_coordinates = load_questionnaire_concat("coordinates")
    df_ft_compass = load_free_text_matrix("compass")
    df_ft_coordinates = load_free_text_matrix("coordinates")

    df_l_merged = merge_compass_coordinates(df_l_compass, df_l_coordinates)
    df_ft_merged = merge_compass_coordinates(df_ft_compass, df_ft_coordinates)

    rows: list[dict] = []
    for llm in llms:
        print(f"\nProcessing {llm}...")
        row: dict = {"llm": llm}

        for prompt in PROMPTS:
            print(f"  Processing prompt {prompt}...")
            pbt_by_axis, f1 = _compute_pbt_f1(llm, prompt, claims=claims)
            for axis in AXIS_SUFFIXES:
                row[f"pbt_{prompt}_{axis}"] = pbt_by_axis[axis]
            row[f"f1_{prompt}"] = f1
            pbt_both = pbt_by_axis.get("both")
            if pbt_both is not None and f1 is not None:
                print(f"    PB-T (both): {pbt_both:.3f}, F1: {f1:.3f}")

        _store_pba_axes(
            row,
            "pba_l_compass",
            pol_bias_axes(
                df_l_compass,
                llm,
                "data/political_compass_statements.csv",
                free_text=False,
            ),
        )
        _store_pba_axes(
            row,
            "pba_l_coordinates",
            pol_bias_axes(
                df_l_coordinates,
                llm,
                "data/political_coordinates_statements.csv",
                free_text=False,
            ),
        )
        _store_pba_axes(
            row,
            "pba_ft_compass",
            pol_bias_axes(
                df_ft_compass,
                llm,
                "data/political_compass_statements.csv",
                free_text=True,
            ),
        )
        _store_pba_axes(
            row,
            "pba_ft_coordinates",
            pol_bias_axes(
                df_ft_coordinates,
                llm,
                "data/political_coordinates_statements.csv",
                free_text=True,
            ),
        )
        _store_pba_axes(row, "pba_combined", pol_bias_combined_axes(llm, df_l_merged, df_ft_merged))
        rows.append(row)

    return rows


def _format_latex_cell(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "---"
    return f"{float(value):.3f}"


def build_model_overview_table(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    latex_columns = overview_latex_columns()
    for key, _ in latex_columns:
        if key.endswith("_economic") or key.endswith("_social"):
            raise ValueError(f"LaTeX overview must use combined axis only, got {key!r}.")

    available_llms = list(df["llm"].unique())
    llms = [llm for llm in target_llms if llm in available_llms]
    llms.extend(sorted(llm for llm in available_llms if llm not in target_llms))

    header = " & ".join(label for _, label in latex_columns)
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{No-persona model metrics (PB-T, PB-A; combined economic and social axes)}",
        r"\label{tab:model_overview}",
        r"{\small",
        r"\begin{tabular}{@{}l" + "c" * len(latex_columns) + "@{}}",
        r"\toprule",
        r"\textbf{LLM} & " + header + r" \\",
        r"\midrule",
    ]

    metric_keys = [key for key, _ in latex_columns]
    for llm in llms:
        llm_row = df[df["llm"] == llm]
        if llm_row.empty:
            continue
        data_row = llm_row.iloc[0]
        model_name = llm_info.get(llm, {}).get("name", llm.capitalize())
        cells = [_format_latex_cell(data_row.get(key)) for key in metric_keys]
        latex_lines.append(f"\\textbf{{{model_name}}} & " + " & ".join(cells) + r" \\")

    latex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    out_path.write_text("\n".join(latex_lines), encoding="utf-8")
    print(f"LaTeX table saved to: {out_path}")


def _attach_speaker_relevance(df: pd.DataFrame) -> pd.DataFrame:
    reg = load_regression_coefficients()[["llm", "speaker_R"]].copy()
    reg["speaker_R"] = reg["speaker_R"].apply(
        lambda v: _round_metric(float(v)) if pd.notna(v) and np.isfinite(float(v)) else None
    )
    return df.merge(reg, on="llm", how="left")


def run_data_preperation() -> pd.DataFrame:
    llms = target_llms
    rows = _build_model_rows(llms)
    df = pd.DataFrame(rows)
    df = df.reindex(columns=["llm", *PBT_F1_METRIC_COLUMNS, *PBA_METRIC_COLUMNS])
    df = _attach_speaker_relevance(df)
    df = df.reindex(columns=["llm", *PBT_F1_METRIC_COLUMNS, *PBA_METRIC_COLUMNS, *REGRESSION_METRIC_COLUMNS])

    interim_dir = MODEL_METRICS_PATH.parent
    interim_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(MODEL_METRICS_PATH, index=False)

    build_model_overview_table(df, Path("output/tables/overview_table.tex"))

    print(f"\nMetrics saved to: {MODEL_METRICS_PATH}")
    print(f"Processed {len(df)} models")
    return df


if __name__ == "__main__":
    run_data_preperation()
