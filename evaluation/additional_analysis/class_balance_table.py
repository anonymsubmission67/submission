"""
Build a LaTeX table of class balance (party × axis × truthfulness label).

Reads labels exactly as in ``data/claims_metadata.csv`` (no relabelling or row filtering).

Input : data/claims_metadata.csv
Output: output/tables/class_balance.tex
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = EVAL_DIR.parent

CLAIMS_CSV = PROJECT_ROOT / "data" / "claims_metadata.csv"
OUT_TEX = PROJECT_ROOT / "output" / "tables" / "class_balance.tex"


def _safe_label_tex(label) -> str:
    s = str(label) if pd.notna(label) else "—"
    return s.replace("_", r"\_").replace("%", r"\%")


def _cell(tab: pd.DataFrame, party: str, axis: str, label) -> int:
    try:
        return int(tab.loc[(party, axis), label])
    except KeyError:
        return 0


def build_latex(distribution_table: pd.DataFrame, labels: list, n_rows: int) -> str:
    lines = [
        r"% Requires: \usepackage{booktabs,multirow}",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Distribution of claims by truthfulness label, speaker party, and topic axis (\texttt{claims\_metadata.csv}).}",
        r"\label{tab:class_balance}",
        r"{\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Label}} & \multicolumn{2}{c}{\textbf{Republican}} & \multicolumn{2}{c}{\textbf{Democrat}} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r" & \textbf{Economic} & \textbf{Social} & \textbf{Economic} & \textbf{Social} \\",
        r"\midrule",
    ]

    for label in labels:
        rep_e = _cell(distribution_table, "Republican", "economic", label)
        rep_s = _cell(distribution_table, "Republican", "social", label)
        dem_e = _cell(distribution_table, "Democrat", "economic", label)
        dem_s = _cell(distribution_table, "Democrat", "social", label)
        lines.append(
            f"{_safe_label_tex(label)} & {rep_e} & {rep_s} & {dem_e} & {dem_s} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            f"% Total rows in {CLAIMS_CSV.name}: {n_rows}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if not CLAIMS_CSV.is_file():
        raise FileNotFoundError(f"Missing input: {CLAIMS_CSV}")

    df = pd.read_csv(CLAIMS_CSV)
    if df.empty:
        OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
        OUT_TEX.write_text("% class_balance: empty input CSV.\n", encoding="utf-8")
        print(f"Wrote placeholder: {OUT_TEX}")
        return

    # Label order: first appearance in the file (pandas unique preserves order)
    labels = list(pd.unique(df["label"].dropna()))

    distribution_table = (
        df.groupby(["party", "axis", "label"]).size().unstack(fill_value=0)
    )

    latex = build_latex(distribution_table, labels, len(df))

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(latex, encoding="utf-8")
    print(f"Wrote {OUT_TEX} ({len(df)} rows, {len(labels)} label levels)")


if __name__ == "__main__":
    main()
