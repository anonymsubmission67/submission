import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add evaluation directory to path to allow importing utils
script_dir = Path(__file__).resolve().parent
evaluation_dir = script_dir.parent
if str(evaluation_dir) not in sys.path:
    sys.path.insert(0, str(evaluation_dir))

from utils import apply_fdr_bh, latex_cellcolor_prefix, llm_info, target_llms

POOL_TAG = "pooled"
PROMPT_SPECS = (
    ("1", "party-agnostic"),
    ("2", "party-aware"),
)

# Feature toggle for background colors in small table
USE_BACKGROUND_COLORS = True


def _get_prompt_row(llm_data: pd.DataFrame, prompt_num: str, llm: str) -> pd.Series:
    rows = llm_data[llm_data["Prompt"].str.contains(prompt_num, na=False)]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one prompt_{prompt_num} row for LLM {llm!r}, found {len(rows)}."
        )
    return rows.iloc[0]


def _require_field(row: pd.Series, field: str, llm: str, prompt_label: str):
    value = row.get(field)
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        raise ValueError(f"Missing {field} for LLM {llm!r} ({prompt_label}).")
    return value


def _directional_stars(p_adj: float, dem_mean: float, rep_mean: float) -> str:
    if p_adj < 0.01:
        stars = r"\ast\ast"
    elif p_adj < 0.05:
        stars = r"\ast"
    else:
        return "$-$"
    if dem_mean > rep_mean:
        return rf"\textcolor{{blue}}{{${stars}$}}"
    return rf"\textcolor{{red}}{{${stars}$}}"


def _test_label(test_type: str) -> str:
    if test_type == "t":
        return "t"
    if test_type == "U":
        return "U"
    raise ValueError(f"Unexpected Test_type {test_type!r}; expected 't' or 'U'.")


def _collect_llm_prompt_entries(both_df: pd.DataFrame, unique_llms: list[str]) -> list[dict]:
    entries = []
    for llm in unique_llms:
        llm_data = both_df[both_df["LLM"] == llm]
        for prompt_num, prompt_label in PROMPT_SPECS:
            row = _get_prompt_row(llm_data, prompt_num, llm)
            raw_p = float(_require_field(row, "Test_p_value", llm, prompt_label))
            test_type = _require_field(row, "Test_type", llm, prompt_label)
            rep_mean = float(_require_field(row, "Republican_mean", llm, prompt_label))
            dem_mean = float(_require_field(row, "Democrat_mean", llm, prompt_label))
            entries.append(
                {
                    "llm": llm,
                    "prompt_num": prompt_num,
                    "prompt_label": prompt_label,
                    "raw_p": raw_p,
                    "test_type": test_type,
                    "rep_mean": rep_mean,
                    "dem_mean": dem_mean,
                    "pb_t": rep_mean - dem_mean,
                }
            )
    return entries


def build_pbt_small():
    """
    Create a small table with significance test results using data from shapiro_wilk_summaries.
    The table shows test type, significance symbols, and truth bias for both prompts.
    """
    out_dir = Path("output/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(f"data/interim_results/shapiro_wilk_summary_{POOL_TAG}.csv")
    both_df = summary_df[(summary_df["Metric"] == "ME") & (summary_df["Axis"] == "both")].copy()

    def get_latex_color(llm):
        if not USE_BACKGROUND_COLORS:
            return ""
        return latex_cellcolor_prefix(llm, opacity=10)

    available_llms = both_df["LLM"].unique()
    unique_llms = [llm for llm in target_llms if llm in available_llms]
    remaining_llms = [llm for llm in available_llms if llm not in target_llms]
    unique_llms.extend(sorted(remaining_llms))

    entries = _collect_llm_prompt_entries(both_df, unique_llms)
    expected_n = len(unique_llms) * len(PROMPT_SPECS)
    if len(entries) != expected_n:
        raise ValueError(f"Expected {expected_n} test p-values, got {len(entries)}.")

    raw_ps = [entry["raw_p"] for entry in entries]
    adj_ps = apply_fdr_bh(raw_ps)
    for entry, p_adj in zip(entries, adj_ps):
        entry["adj_p"] = float(p_adj)

    adj_by_key = {(e["llm"], e["prompt_num"]): e for e in entries}

    latex_code_small = """
    \\begin{table}[h]
        \\centering
        {\\small
        \\setlength{\\tabcolsep}{3.5pt}
        \\begin{tabular}{@{}l|ccc|ccc@{}}
            \\toprule
            \\multirow{2}{*}{\\textbf{}} & \\multicolumn{3}{c}{\\textbf{party-agnostic}} & \\multicolumn{3}{c}{\\textbf{party-aware}} \\\\
            \\cmidrule(lr){2-4} \\cmidrule(lr){5-7}
            & \\textbf{test} & \\textbf{sig} & \\textbf{PB-T} & \\textbf{test} & \\textbf{sig} & \\textbf{PB-T} \\\\
            \\midrule
    """

    for llm in unique_llms:
        p1 = adj_by_key[(llm, "1")]
        p2 = adj_by_key[(llm, "2")]

        prompt1_parametric = _test_label(p1["test_type"])
        prompt1_symbol = _directional_stars(p1["adj_p"], p1["dem_mean"], p1["rep_mean"])
        prompt1_tb = f"{p1['pb_t']:.3f}"

        prompt2_parametric = _test_label(p2["test_type"])
        prompt2_symbol = _directional_stars(p2["adj_p"], p2["dem_mean"], p2["rep_mean"])
        prompt2_tb = f"{p2['pb_t']:.3f}"

        llm_name = llm_info.get(llm, {}).get("name", llm.capitalize())
        background_color = get_latex_color(llm)
        latex_code_small += (
            f"        {background_color}{llm_name} & {background_color}{prompt1_parametric} "
            f"& {background_color}{prompt1_symbol} & {background_color}{prompt1_tb} "
            f"& {background_color}{prompt2_parametric} & {background_color}{prompt2_symbol} "
            f"& {background_color}{prompt2_tb} \\\\\n"
        )

    latex_code_small += """
            \\bottomrule
        \\end{tabular}
        \\caption{Adaptive significance test results: Republican vs Democrat responses (both axes combined). Significance symbols use Benjamini--Hochberg FDR-adjusted $p$-values across all models and both prompt conditions (\\texttt{*}, \\texttt{**} at $p<0.05$, $p<0.01$). Blue: Democrat $>$ Republican; red: Republican $>$ Democrat.}
        \\label{tab:pbt_small}
        }
    \\end{table}
    """

    with open("output/tables/pbt_small.tex", "w") as f:
        f.write(latex_code_small)

    print(f"Small table saved to: output/tables/pbt_small.tex")
    print(f"Processed {len(unique_llms)} LLMs ({len(entries)} FDR-adjusted tests)")


if __name__ == "__main__":
    build_pbt_small()
