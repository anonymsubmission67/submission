"""PB-T economic/social scatter (prompts 1–3, marker styles)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

script_dir = Path(__file__).resolve().parent
evaluation_dir = script_dir.parent
if str(evaluation_dir) not in sys.path:
    sys.path.insert(0, str(evaluation_dir))

from utils import CLAIM_MATRIX_ROOT, draw_eco_social_axis_labels, llm_color, llm_info, pbt_mean_path, target_llms

MARKER_SIZE_CIRCLE = 78
MARKER_SIZE_RECT = 80

PBT_XLIM = (-1.0, 0.5)
PBT_YLIM = (-1.0, 0.5)

LABEL_TO_NUM = {
    "pants-fire": 0,
    "false": 1,
    "mostly-false": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5,
}


def _me_diff_cols_for_prompt(prompt: str) -> tuple[str, str]:
    return f"me_diff_economic_prompt_{prompt}", f"me_diff_social_prompt_{prompt}"


def load_me_diff() -> pd.DataFrame:
    claims = pd.read_csv("data/claims_metadata.csv").set_index("claim_id")
    me_diff_results: dict[str, pd.DataFrame] = {}
    base = CLAIM_MATRIX_ROOT

    for prompt in ["1", "2", "3"]:
        csv_path = pbt_mean_path(prompt)
        if not csv_path.is_file():
            continue

        df_long = pd.read_csv(csv_path, index_col=0)
        df_long = df_long.T.merge(
            claims[["party", "label", "axis"]],
            left_index=True,
            right_index=True,
            how="left",
        )
        df_long["label"] = df_long["label"].map(LABEL_TO_NUM)

        for col in df_long.columns:
            if col not in ["party", "label", "axis"]:
                df_long[col] = df_long[col] - df_long["label"]

        df_long = df_long.drop(columns=["label"])
        bias_by_party = df_long.groupby(["party", "axis"]).mean().T
        needed_cols = [
            ("Republican", "economic"),
            ("Democrat", "economic"),
            ("Republican", "social"),
            ("Democrat", "social"),
        ]
        if not all(c in bias_by_party.columns for c in needed_cols):
            continue

        me_diff_economic = bias_by_party[("Republican", "economic")] - bias_by_party[("Democrat", "economic")]
        me_diff_social = bias_by_party[("Republican", "social")] - bias_by_party[("Democrat", "social")]
        me_diff_results[f"prompt_{prompt}"] = pd.DataFrame(
            {"me_diff_economic": me_diff_economic, "me_diff_social": me_diff_social}
        )

    if not me_diff_results:
        raise FileNotFoundError(
            f"No mean claim matrices found under {base} (expected {{1,2,3}}_mean.csv)."
        )

    me_diff = pd.concat(me_diff_results, axis=1)
    me_diff.columns = [f"{col[1]}_{col[0]}" for col in me_diff.columns]
    return me_diff


def _llm_plotted(me_diff: pd.DataFrame, llm: str) -> bool:
    if llm not in me_diff.index:
        return False
    row = me_diff.loc[llm]
    for prompt_type in ["1", "2", "3"]:
        cx, cy = _me_diff_cols_for_prompt(prompt_type)
        if cx not in row.index or cy not in row.index:
            continue
        if pd.notna(row[cx]) and pd.notna(row[cy]):
            return True
    return False


def draw_pbt_scatter_on_ax(
    ax: plt.Axes,
    me_diff: pd.DataFrame,
    *,
    marker_size_circle: float = MARKER_SIZE_CIRCLE,
    marker_size_rect: float = MARKER_SIZE_RECT,
) -> None:
    x_min, x_max = PBT_XLIM
    y_min, y_max = PBT_YLIM

    for prompt_type in ["1", "2", "3"]:
        col_x, col_y = _me_diff_cols_for_prompt(prompt_type)
        if col_x not in me_diff.columns or col_y not in me_diff.columns:
            continue

        for llm in target_llms:
            if llm not in me_diff.index:
                continue
            x_val = me_diff.loc[llm, col_x]
            y_val = me_diff.loc[llm, col_y]
            if pd.isna(x_val) or pd.isna(y_val):
                continue

            color = llm_color(llm)
            if prompt_type == "1":
                ax.scatter(
                    x_val,
                    y_val,
                    facecolor="none",
                    edgecolor=color["edgecolor"],
                    linewidth=2,
                    marker="o",
                    s=marker_size_circle,
                    zorder=5,
                )
            elif prompt_type == "2":
                ax.scatter(
                    x_val,
                    y_val,
                    facecolor=color["edgecolor"],
                    edgecolor=color["edgecolor"],
                    linewidth=2,
                    marker="o",
                    s=marker_size_circle,
                    zorder=5,
                )
            else:
                ax.scatter(
                    x_val,
                    y_val,
                    facecolors=color["edgecolor"],
                    edgecolors=color["edgecolor"],
                    linewidths=1.75,
                    marker="s",
                    s=marker_size_rect,
                    zorder=5,
                )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.annotate("", xy=(x_min * 0.9, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.annotate("", xy=(x_max * 0.9, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.annotate("", xy=(0, y_min * 0.9), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.annotate("", xy=(0, y_max * 0.9), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    ax.text(x_max * 0.85, y_min * 0.05, "Republican", ha="center", va="center", fontsize=11)
    ax.text(x_min * 0.85, y_min * 0.05, "Democrat", ha="center", va="center", fontsize=11)
    ax.text(x_max * 0.25, y_max * 0.85, "Republican", ha="center", va="center", fontsize=11)
    ax.text(x_max * 0.20, y_min * 0.85, "Democrat", ha="center", va="center", fontsize=11)
    draw_eco_social_axis_labels(ax)


def pbt_scatter_model_legend_handles(me_diff: pd.DataFrame) -> list[mlines.Line2D]:
    handles: list[mlines.Line2D] = []
    for llm in target_llms:
        if not _llm_plotted(me_diff, llm):
            continue
        llm_name = llm_info.get(llm, {}).get("name", llm.capitalize())
        ec = llm_color(llm)["edgecolor"]
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=ec,
                marker="o",
                linestyle="None",
                markersize=9,
                markerfacecolor=ec,
                markeredgecolor=ec,
                markeredgewidth=0,
                label=llm_name,
            )
        )
    return handles


def pbt_scatter_prompt_legend_handles(me_diff: pd.DataFrame) -> list[mlines.Line2D]:
    prompt_specs = [
        (
            "1",
            mlines.Line2D(
                [],
                [],
                color="gray",
                marker="o",
                linestyle="None",
                markersize=9,
                markerfacecolor="white",
                markeredgecolor="gray",
                markeredgewidth=1.5,
                label="party-agnostic",
            ),
        ),
        (
            "2",
            mlines.Line2D(
                [],
                [],
                color="gray",
                marker="o",
                linestyle="None",
                markersize=9,
                markerfacecolor="gray",
                markeredgecolor="gray",
                markeredgewidth=0,
                label="party-aware",
            ),
        ),
        (
            "3",
            mlines.Line2D(
                [],
                [],
                color="gray",
                marker="s",
                linestyle="None",
                markersize=10,
                markerfacecolor="gray",
                markeredgecolor="dimgray",
                markeredgewidth=1.25,
                label="party-flipped",
            ),
        ),
    ]
    return [h for p, h in prompt_specs if _me_diff_cols_for_prompt(p)[0] in me_diff.columns]


def build_pbt_scatter() -> None:
    me_diff = load_me_diff()
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_pbt_scatter_on_ax(ax, me_diff)
    ax.legend(
        handles=pbt_scatter_prompt_legend_handles(me_diff),
        loc="upper left",
        bbox_to_anchor=(0, 0.40),
        title="prompt conditions",
        framealpha=1,
        facecolor="white",
        columnspacing=1.0,
        handletextpad=0.5,
    )
    fig.tight_layout()
    out = Path("output/images/pbt_scatter.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="png", dpi=300)
    plt.close(fig)
    print(f"PB-T scatter saved to: {out}")


if __name__ == "__main__":
    build_pbt_scatter()
