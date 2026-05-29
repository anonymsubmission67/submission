import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

# Matrices under ``data/claim_matrices/`` (PBT: ``{1,2,3}_mean.csv``; PBA likert/free_text subdirs).
CLAIM_MATRIX_ROOT = Path("data/claim_matrices")
LIKERT_MATRIX_DIR = CLAIM_MATRIX_ROOT / "likert"
FREE_TEXT_MATRIX_DIR = CLAIM_MATRIX_ROOT / "free_text"


def pbt_mean_path(prompt: str) -> Path:
    """Pooled PolitiFact matrix for prompt ``1`` | ``2`` | ``3``."""
    return CLAIM_MATRIX_ROOT / f"{prompt}_mean.csv"


def likert_mean_path(questionnaire: str) -> Path:
    """Likert PB-A matrix (``compass`` | ``coordinates``)."""
    return LIKERT_MATRIX_DIR / f"{questionnaire}_mean.csv"


def likert_variance_path(questionnaire: str) -> Path:
    return LIKERT_MATRIX_DIR / f"{questionnaire}_variance.csv"


def free_text_mean_path(questionnaire: str) -> Path:
    return FREE_TEXT_MATRIX_DIR / f"{questionnaire}_mean.csv"


def free_text_variance_path(questionnaire: str) -> Path:
    return FREE_TEXT_MATRIX_DIR / f"{questionnaire}_variance.csv"

target_llms = ["phi","phi_small","phi_medium","liberal", "conservative", "ministral3", "ministral8", "ministral14", "mixtral", "qwen", "qwen_big","qwen_very_big", "llama3", "llama3_big", "llama4", "gpt-oss","gpt4", "deepseek_small","deepseek",  "grok"] #  "llama4"

llm_colors = {

    'ministral3': {"fillcolor": "white", "edgecolor": "orchid", "latexcolor": "violet"},
    'ministral8': {"fillcolor": "white", "edgecolor": "m", "latexcolor": "magenta"},
    'ministral14': {"fillcolor": "white", "edgecolor": "purple", "latexcolor": "purple"},
    'mixtral': {"fillcolor": "white", "edgecolor": "indigo", "latexcolor": "blue!50!black"},

    'llama3': {"fillcolor": "white", "edgecolor": "limegreen", "latexcolor": "green"},
    'llama3_big': {"fillcolor": "white", "edgecolor": "forestgreen", "latexcolor": "green!50!black"},
    'llama4': {"fillcolor": "white", "edgecolor": "darkgreen", "latexcolor": "green!70!black"},

    'qwen': {"fillcolor": "white", "edgecolor": "sandybrown", "latexcolor": "brown"},
    'qwen_big': {"fillcolor": "white", "edgecolor": "chocolate", "latexcolor": "brown!50!black"},
    'qwen_very_big': {"fillcolor": "white", "edgecolor": "saddlebrown", "latexcolor": "brown!70!black"},

    'deepseek_small': {"fillcolor": "white", "edgecolor": "mediumpurple", "latexcolor": "purple"},
    'deepseek': {"fillcolor": "white", "edgecolor": "rebeccapurple", "latexcolor": "purple!50!black"},

    'gpt-oss': {"fillcolor": "white", "edgecolor": "darkturquoise", "latexcolor": "cyan"},
    'gpt4': {"fillcolor": "white", "edgecolor": "teal", "latexcolor": "teal"},

    # "grok_small": {"fillcolor": "white", "edgecolor": "violet", "latexcolor": "violet"},
    'grok': {"fillcolor": "white", "edgecolor": "gold", "latexcolor": "orange"},

    'phi': {"fillcolor": "white", "edgecolor": "darkgray", "latexcolor": "gray"},
    'phi_small': {"fillcolor": "white", "edgecolor": "dimgray", "latexcolor": "gray!70!black"},
    'phi_medium': {"fillcolor": "white", "edgecolor": "black", "latexcolor": "gray!40!black"},

    'conservative': {"fillcolor": "white", "edgecolor": "red", "latexcolor": "red"},
    'liberal': {"fillcolor": "white", "edgecolor": "blue", "latexcolor": "blue"},
    "american": {"fillcolor": "white", "edgecolor": "black", "latexcolor": "black"},

    # 'men': {"fillcolor": "orange", "edgecolor": "orange", "latexcolor": "orange"},
    # 'women': {"fillcolor": "darkred", "edgecolor": "darkred", "latexcolor": "red!70!black"},
    # 'teenagers': {"fillcolor": "purple", "edgecolor": "purple", "latexcolor": "purple"},
    # 'over_30': {"fillcolor": "pink", "edgecolor": "pink", "latexcolor": "pink"},
    # 'old_people': {"fillcolor": "gray", "edgecolor": "gray", "latexcolor": "gray"},
    # 'qwen_right': {"fillcolor": "white", "edgecolor": "gray", "latexcolor": "gray"},
}

_DEFAULT_LLM_COLORS = {"fillcolor": "white", "edgecolor": "gray", "latexcolor": "gray"}


def llm_color(llm: str) -> dict:
    """Matplotlib + LaTeX color spec for one model key."""
    return llm_colors.get(llm, _DEFAULT_LLM_COLORS)


def latex_cellcolor_prefix(llm: str, *, opacity: int = 10) -> str:
    """LaTeX ``\\cellcolor`` prefix for table cells (content follows unwrapped)."""
    latex = llm_color(llm)["latexcolor"]
    return f"\\cellcolor{{{latex}!{opacity}}}"


def latex_cellcolor_wrap(body: str, llm: str, *, opacity: int = 12) -> str:
    """Wrap cell body in ``\\cellcolor{...}``."""
    latex = llm_color(llm)["latexcolor"]
    return f"\\cellcolor{{{latex}!{opacity}}}{{{body}}}"


def p_to_stars(p) -> str:
    """* if p < 0.05, ** if p < 0.01; empty otherwise."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def apply_fdr_bh(p_values, *, alpha: float = 0.05) -> np.ndarray:
    """Benjamini–Hochberg FDR-adjusted p-values (``fdr_bh``); NaNs preserved."""
    p = pd.Series(p_values, dtype=float)
    valid = p.notna()
    if valid.sum() == 0:
        return p.to_numpy()
    _, p_adj, _, _ = multipletests(p[valid].to_numpy(), alpha=alpha, method="fdr_bh")
    out = p.copy()
    out.loc[valid] = p_adj
    return out.to_numpy()
llm_info = {
"mistral": {"source": "mistralai/Mistral-7B-Instruct-v0.1", "size_B": 7, "name": "Mistral 7B", "license": "Apache-2.0"},
"mixtral": {"source": "mistralai/Mixtral-8x7B-Instruct-v0.1", "size_B": 56, "name": "Mixtral 8x7B", "license": "Apache-2.0"},

"ministral3": {"source": "mistralai/Ministral-3-3B-Instruct-2512", "size_B": 3, "name": "Ministral 3B", "license": "Apache-2.0"},
"ministral8": {"source": "mistralai/Ministral-3-8B-Instruct-2512", "size_B": 8, "name": "Ministral 8B", "license": "Apache-2.0"},
"ministral14": {"source": "mistralai/Ministral-3-14B-Instruct-2512", "size_B": 14, "name": "Ministral 14B", "license": "Apache-2.0"},


"qwen": {"source": "Qwen/Qwen3-4B-Instruct-2507", "size_B": 4, "name": "Qwen3 4B", "license": "Apache-2.0"},
"qwen_big": {"source": "Qwen/Qwen3-30B-A3B-Instruct-2507", "size_B": 72, "name": "Qwen3 30B", "license": "Apache-2.0"},
"qwen_very_big": {"source": "Qwen/Qwen3-Next-80B-A3B-Instruct", "size_B": 80, "name": "Qwen3 80B", "license": "Apache-2.0"},

"llama3": {"source": "meta-llama/Meta-Llama-3-8B-Instruct", "size_B": 8, "name": "Llama3 8B", "license": "llama3"},
"llama3_big": {"source": "meta-llama/Meta-Llama-3-70B-Instruct", "size_B": 80, "name": "Llama3 80B", "license": "llama3"},
"llama4": {"source": "meta-llama/Llama-4-Scout-17B-16E-Instruct", "size_B": 109, "name": "Llama 4 Scout", "license": "llama4"}, # active: 17B

"gpt-oss": {"source": "openai/gpt-oss-20b", "size_B": 20, "name": "GPT-OSS 20B", "license": "Apache-2.0"},
"gpt4": {"source": "https://openrouter.ai/openai/gpt-4.1", "size_B": 200, "name": "GPT 4.1", "license": "paid service"}, # nur eine Schätzung für GPT-4o https://aiexpjourney.substack.com/p/the-number-of-parameters-of-gpt-4o

"deepseek_small": {"source": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "size_B": 32, "name": "DeepSeek R1 Dist.", "license": "mit"},
"deepseek": {"source": "https://openrouter.ai/deepseek/deepseek-chat-v3.1:free", "size_B": 671, "name": "DeepSeek 3.1", "license": "paid service"},

"grok": {"source": "https://openrouter.ai/x-ai/grok-4.1", "size_B": 1700, "name": "Grok 4.1", "license": "paid service"}, # mehrere quellen sagen 1.7 trillion e.g. https://www.datastudios.org/post/grok-4-vs-previous-models-1-1-5-2-3-3-5-full-comparison-of-architecture-capabilities-and-r#:~:text=Grok%204%20(July%202025)%20represents,a%20huge%20jump%20in%20capacity.


"phi": {"source": "unsloth/Phi-3-mini-4k-instruct", "size_B": 3.8, "name": "Phi 3 mini", "license": "access upon request"},
"phi_small": {"source": "microsoft/Phi-3-small-8k-instruct", "size_B": 7, "name": "Phi 3 small", "license": "access upon request"},
"phi_medium": {"source": "microsoft/Phi-3-medium-128k-instruct", "size_B": 14, "name": "Phi 3 medium", "license": "access upon request"},

"conservative": {"source": "Opinion-GPT/opiniongpt-phi3-conservative", "size_B": 3.8, "name": "Phi 3 Conservative", "license": "access upon request"},
"liberal": {"source": "Opinion-GPT/opiniongpt-phi3-liberal", "size_B": 3.8, "name": "Phi 3 Liberal", "license": "access upon request"},

}


def make_tertile(s):
    s = pd.to_numeric(s, errors="coerce")
    cats, bins = pd.qcut(s, q=3, labels=None, retbins=True, duplicates="drop")
    n = len(bins) - 1  # actual number of bins after dropping duplicates
    labels = ["low", "middle", "high"][:n]
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True)




claims = pd.read_csv("data/claims_metadata.csv").set_index("claim_id")


# Map dataset labels to numeric 0..5
label_to_num = {
    "pants-fire": 0, "false": 1, "mostly-false": 2,
    "half-true": 3, "mostly-true": 4, "true": 5,
}

def calc_me_diff(df):
        df_long = df.T.merge(claims[["party", "label"]], left_index=True, right_index=True, how="left")
        df_long["label"] = df_long["label"].map(label_to_num)

        # Subtract dataset_label from all columns except "party" and "dataset_label"
        for col in df_long.columns:
            if col not in ["party", "label"]:
                df_long[col] = df_long[col] - df_long["label"]

        df_long = df_long.drop(columns=["label"])

        bias_by_party = (
            df_long.groupby(["party"])
            .mean()
        ).T

        return bias_by_party["Republican"] - bias_by_party["Democrat"]



def draw_eco_social_axis_labels(ax: plt.Axes, *, fontsize: int = 11) -> None:
    """Dimension labels on economic (x) and social (y) axes; shared by PB-T and PB-A panels."""
    _, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.text(x_max * 0.45, y_min * 0.03, "economic", ha="center", va="center", fontsize=fontsize)
    y_social = 0.5 * (y_min + y_max)
    ax.text(x_max * 0.03, y_social, "social", ha="center", va="center", fontsize=fontsize, rotation=90)


def draw_y_party_labels(ax: plt.Axes, *, fontsize: int = 11) -> None:
    """Democrat at the lower y limit; Republican at y=0 (upper axis end)."""
    x_min, x_max = ax.get_xlim()
    y_min, _ = ax.get_ylim()
    x_left = x_min + 0.04 * (x_max - x_min)
    ax.text(x_left, y_min, "Democrat", ha="right", va="bottom", fontsize=fontsize)
    ax.text(x_left, 0, "Republican", ha="right", va="center", fontsize=fontsize)

