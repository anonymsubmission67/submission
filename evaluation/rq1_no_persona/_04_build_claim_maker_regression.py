"""
OLS regression: model score ~ intercept + party + speaker + label (Republican dummies).

Exports ``output/tables/claim_maker_significance.tex`` and
``data/interim_results/party_speaker_regression.csv`` (signed coefficients + p-values).

Coefficients are displayed as absolute values; the largest in each model row is bold in the LaTeX output.

``speaker_R`` p-values are Benjamini–Hochberg FDR-adjusted across models; stars (*, **) only for that term.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import sys

script_dir = Path(__file__).resolve().parent
evaluation_dir = script_dir.parent
if str(evaluation_dir) not in sys.path:
    sys.path.insert(0, str(evaluation_dir))

from utils import latex_cellcolor_wrap, llm_info, pbt_mean_path, target_llms

REGRESSION_COEF_PATH = Path("data/interim_results/claim_maker_significance.csv")

# Fitted with intercept; table shows slopes only (no intercept column).
HEADER_TERMS = ["Intercept", "party_R", "speaker_R", "label"]
TABLE_TERMS = ["label", "party_R", "speaker_R"]
DISPLAY = {
    "party_R": r"\makecell{\textbf{claim} \\ \textbf{party}}",
    "speaker_R": r"\makecell{\textbf{claim maker} \\ \textbf{party}}",
    "label": r"\makecell{\textbf{golden} \\ \textbf{truth}}",
}


def flip_party(party):
    if party == "Democrat":
        return "Republican"
    if party == "Republican":
        return "Democrat"
    return party


def fmt_num(x, digits=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    if abs(x) < 1e-6 and abs(x) > 0:
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}"


def p_to_stars(p) -> str:
    """* if p < 0.05, ** if p < 0.01; empty otherwise."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def cell_coef_abs(coef, *, bold=False, digits=3):
    """Absolute coefficient in the table; inference (stars) still uses signed slopes."""
    if coef is not None and np.isfinite(float(coef)):
        txt = fmt_num(abs(coef), digits=digits)
    else:
        txt = fmt_num(coef, digits=digits)
    if bold and txt != "---":
        return rf"\textbf{{{txt}}}"
    return txt


def cell_stars(p):
    stars = p_to_stars(p)
    if not stars:
        return ""
    if stars == "**":
        return r"\textcolor{blue}{$\ast\ast$}"
    return r"\textcolor{blue}{$\ast$}"


def _apply_speaker_fdr(pval_df: pd.DataFrame) -> pd.DataFrame:
    """Benjamini–Hochberg FDR on ``speaker_R`` p-values across models (``fdr_bh``, alpha=0.05)."""
    out = pval_df.copy()
    speaker_p = pd.to_numeric(out["speaker_R"], errors="coerce")
    valid = speaker_p.notna()
    if valid.sum() == 0:
        return out
    _, p_adj, _, _ = multipletests(speaker_p[valid].values, alpha=0.05, method="fdr_bh")
    out.loc[valid, "speaker_R"] = p_adj
    return out


def load_regression_coefficients() -> pd.DataFrame:
    if not REGRESSION_COEF_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {REGRESSION_COEF_PATH}; run rq1 _05_build_party_speaker_regression.py first."
        )
    return pd.read_csv(REGRESSION_COEF_PATH)


def _save_regression_csv(
    coef_df: pd.DataFrame,
    pval_df: pd.DataFrame,
    *,
    pval_raw_df: pd.DataFrame | None = None,
) -> Path:
    REGRESSION_COEF_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"llm": coef_df.index})
    for term in HEADER_TERMS:
        key = term.lower() if term == "Intercept" else term
        out[key] = coef_df[term].values
        p_key = f"{key}_p" if key != "intercept" else "intercept_p"
        out[p_key] = pval_df[term].values
        if term == "speaker_R" and pval_raw_df is not None:
            out["speaker_R_p_raw"] = pval_raw_df["speaker_R"].values
    out.to_csv(REGRESSION_COEF_PATH, index=False)
    return REGRESSION_COEF_PATH


def run_regression_analysis():
    """OLS table → ``output/tables/regression_table.tex``; returns ``(tex_path, coef_df, combined)``."""
    claims = pd.read_csv("data/claims_metadata.csv").set_index("claim_id")
    label_to_num = {
        "pants-fire": 0,
        "false": 1,
        "mostly-false": 2,
        "half-true": 3,
        "mostly-true": 4,
        "true": 5,
    }

    list_long = []
    for prompt in ["2", "3"]:
        pth = pbt_mean_path(prompt)
        if not pth.is_file():
            continue
        df = pd.read_csv(pth, index_col=0).T.merge(
            claims[["party", "label", "axis"]], left_index=True, right_index=True
        )
        df["label"] = df["label"].map(label_to_num)
        df["prompt_number"] = prompt
        list_long.append(df)

    df_long = pd.concat(list_long, axis=0)
    df_long["speaker"] = np.where(
        df_long["prompt_number"].astype(str).eq("3"),
        df_long["party"].map(flip_party),
        df_long["party"],
    )

    T = df_long.assign(
        pb=(df_long.party == "Republican").astype(float),
        sb=(df_long.speaker == "Republican").astype(float),
    )

    coef_rows = []
    pval_rows = []

    for m in target_llms:
        if m not in T.columns:
            continue
        z = T[[m, "pb", "sb", "label"]].dropna()
        if len(z) < 12:
            continue
        fit = sm.OLS(z[m], sm.add_constant(z[["pb", "sb", "label"]])).fit()
        c = fit.params.rename({"const": "Intercept", "pb": "party_R", "sb": "speaker_R", "label": "label"})
        pvals = fit.pvalues.rename({"const": "Intercept", "pb": "party_R", "sb": "speaker_R", "label": "label"})
        coef_rows.append(pd.Series({t: c[t] for t in HEADER_TERMS}, name=m))
        pval_rows.append(pd.Series({t: pvals[t] for t in HEADER_TERMS}, name=m))

    coef_df = pd.DataFrame(coef_rows)
    pval_raw_df = pd.DataFrame(pval_rows)
    pval_df = _apply_speaker_fdr(pval_raw_df)

    if not coef_df.empty:
        csv_path = _save_regression_csv(coef_df, pval_df, pval_raw_df=pval_raw_df)
        print(f"Regression coefficients saved to: {csv_path}")

    combined = pd.DataFrame(index=coef_df.index)
    for term in TABLE_TERMS:
        combined[(term, "coef")] = coef_df[term].abs()
        if term == "speaker_R":
            combined[(term, "sig")] = pval_df[term].map(p_to_stars)
    combined.columns = pd.MultiIndex.from_tuples(combined.columns)

    out_dir = Path("output/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = out_dir / "claim_maker_significance.tex"

    # --- LaTeX: model + coef columns; stars only for FDR-adjusted speaker_R ---
    if coef_df.empty:
        tex_path.write_text("% No regressions (no model columns / too few observations).\n", encoding="utf-8")
    else:
        col_spec = "@{}lccc@{\\hspace{1pt}}c@{}"

        caption = (
            r"\caption{OLS regressions ($y$: model score). Republican indicators for actual and pretended party; "
            r"golden-truth ordinal score. Intercept omitted from display (still fitted in OLS). "
            r"\textbf{Coefficient columns} report $\left|\hat{\beta}\right|$; "
            r"within each model row the largest magnitude is set in bold. "
            r"Significance for claim maker party only: Benjamini--Hochberg FDR-adjusted $p$-values across models "
            r"(\texttt{*}, \texttt{**} at $p<0.05$, $p<0.01$).}"
        )
        lines = [
            r"% Requires: \usepackage{xcolor,colortbl,booktabs}",
            r"\begin{table}[ht]",
            r"\centering",
            r"{\small",
            r"\setlength{\tabcolsep}{3.5pt}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
        ]

        lines.append(
            " & "
            + DISPLAY["label"]
            + " & "
            + DISPLAY["party_R"]
            + " & "
            + r"\multicolumn{2}{c}{"
            + DISPLAY["speaker_R"]
            + r"} \\"
        )
        lines.append(r"\midrule")

        def tint(body: str, llm: str) -> str:
            return latex_cellcolor_wrap(body, llm, opacity=12)

        for mid in coef_df.index:
            name = llm_info.get(mid, {}).get("name", mid).replace("_", "\\_")
            abs_coefs = [abs(float(coef_df.loc[mid, t])) for t in TABLE_TERMS]
            max_abs = max(abs_coefs)
            parts = [tint(name, mid)]
            for term, a in zip(TABLE_TERMS, abs_coefs):
                cc = coef_df.loc[mid, term]
                pv = pval_df.loc[mid, term]
                is_max = np.isfinite(cc) and np.isclose(a, max_abs, rtol=0.0, atol=1e-12)
                parts.append(tint(cell_coef_abs(cc, bold=is_max), mid))
                if term == "speaker_R":
                    parts.append(tint(cell_stars(pv), mid))
            lines.append(" & ".join(parts) + r" \\")

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            caption,
            r"\label{tab:regression_results}",
            r"}",
            r"\end{table}",
            "",
        ]
        tex_path.write_text("\n".join(lines), encoding="utf-8")

    return tex_path, coef_df, combined


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    tex_path, coef_df, combined = run_regression_analysis()
    if not coef_df.empty:
        print(combined.round(3))
    print(f"\nSaved: {tex_path}")
