"""
RQ3 — LLM size scaling plot (Phi, Ministral, Qwen, Llama families).

Five stacked subplots (one metric each) share five x tiers (3–4B … 80B). At each tier,
size-matched variants from each family share the same nominal x with a small horizontal offset.
Points use per-model ``llm_colors``; connecting lines use the middle model color per family.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_evaluation_root = Path(__file__).resolve().parents[1]
if str(_evaluation_root) not in sys.path:
    sys.path.insert(0, str(_evaluation_root))

from rq1_no_persona._04_build_claim_maker_regression import load_regression_coefficients
from rq2_pba_pbt.utils import load_model_metrics, pba_ft_mean, pba_l_mean
from utils import llm_color, llm_info

TIER_LABELS = ["3-4B", "7-8B", "14B", "30B", "80B"]
N_TIERS = len(TIER_LABELS)

PHI_TIER = [("phi", 0), ("phi_small", 1), ("phi_medium", 2)]
MINISTRAL_TIER = [("ministral3", 0), ("ministral8", 1), ("ministral14", 2)]
QWEN_TIER = [("qwen", 0), ("qwen_big", 3), ("qwen_very_big", 4)]
LLAMA_TIER = [("llama3", 1), ("llama3_big", 4)]

FAMILIES: dict[str, list[tuple[str, int]]] = {
    "phi": PHI_TIER,
    "ministral": MINISTRAL_TIER,
    "qwen": QWEN_TIER,
    "llama": LLAMA_TIER,
}
FAMILY_ORDER = ["phi", "ministral", "qwen", "llama"]
FAMILY_DISPLAY = {"phi": "Phi3", "ministral": "Ministral", "qwen": "Qwen3", "llama": "Llama3"}
LINE_COLOR_KEY = {
    "phi": "phi_small",
    "ministral": "ministral8",
    "qwen": "qwen_big",
    "llama": "llama3",
}
FAMILY_X_OFFSET = {family: 0.0 for family in FAMILY_ORDER}

FAMILY_LABEL_X_PAD = 0.10

# (metric_key, inner y-label per panel)
METRIC_PANELS: list[tuple[str, str]] = [
    ("pbt_1", "party-agnostic"),
    ("pbt_2", "party-aware"),
    ("pba_l", "Likert"),
    ("pba_ft", "free text"),
    ("speaker_R", "claim maker rel."),
]

# Outer y-label groups: (label, panel row indices) spanning the left margin column.
Y_LABEL_GROUPS: list[tuple[str, list[int]]] = [
    ("PB-T", [0, 1]),
    ("PB-A", [2, 3]),
]

GROUP_LABEL_FONTSIZE = 11
INNER_YLABEL_FONTSIZE = 10
LEFT_LABEL_WIDTH = 0.13


def _metrics_lookup(metrics: pd.DataFrame) -> dict[str, pd.Series]:
    return {str(row["llm"]): row for _, row in metrics.iterrows()}


def _regression_lookup(reg: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for _, row in reg.iterrows():
        llm = str(row["llm"])
        val = pd.to_numeric(row["speaker_R"], errors="coerce")
        if pd.notna(val):
            out[llm] = float(val)
    return out


def _model_at_tier(family: str, tier: int) -> str | None:
    for model_id, model_tier in FAMILIES[family]:
        if model_tier == tier:
            return model_id
    return None


def _first_finite_metric_y(
    family: str,
    metric_key: str,
    metrics_by_llm: dict[str, pd.Series],
    speaker_by_llm: dict[str, float],
) -> float | None:
    for model_id, _ in FAMILIES[family]:
        y = _metric_value(model_id, metric_key, metrics_by_llm, speaker_by_llm)
        if y is not None and np.isfinite(y):
            return y
    return None


def _tier_x(family: str, tier: int) -> float:
    return float(tier) + FAMILY_X_OFFSET[family]


def _point_color(model_id: str) -> str:
    return llm_color(model_id)["edgecolor"]


def _line_color(family: str) -> str:
    key = LINE_COLOR_KEY[family]
    return llm_color(key)["edgecolor"]


def _empty_legend_handle() -> mlines.Line2D:
    return mlines.Line2D(
        [],
        [],
        color="none",
        marker="None",
        linestyle="None",
        label="",
    )


def _legend_handles() -> list[mlines.Line2D]:
    """4 rows (families) × 4 columns (family name + models), row-major for ``ncol=4``."""
    handles: list[mlines.Line2D] = []
    for family in FAMILY_ORDER:
        handles.append(
            mlines.Line2D(
                [],
                [],
                color="none",
                marker="None",
                linestyle="None",
                label=FAMILY_DISPLAY[family],
            )
        )
        tier_models = FAMILIES[family]
        for i in range(3):
            if i < len(tier_models):
                model_id = tier_models[i][0]
                handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        color=_point_color(model_id),
                        marker="o",
                        linestyle="None",
                        markersize=7,
                        label=llm_info.get(model_id, {}).get("name", model_id),
                    )
                )
            else:
                handles.append(_empty_legend_handle())
    return handles


def _annotate_family_labels(
    ax,
    metric_key: str,
    metrics_by_llm: dict[str, pd.Series],
    speaker_by_llm: dict[str, float],
) -> None:
    for family in FAMILY_ORDER:
        y = _first_finite_metric_y(family, metric_key, metrics_by_llm, speaker_by_llm)
        if y is None or not np.isfinite(y):
            continue
        x = _tier_x(family, 0)
        ax.text(
            x - FAMILY_LABEL_X_PAD,
            y,
            FAMILY_DISPLAY[family],
            ha="right",
            va="center",
            fontsize=9,
            color=_line_color(family),
            fontweight="bold",
            clip_on=False,
        )


def _plot_series(
    ax,
    tier_models: list[tuple[str, int]],
    family: str,
    y_values: list[float | None],
    *,
    linewidth: float = 1.8,
) -> None:
    xs: list[float] = []
    ys: list[float] = []
    for (model_id, tier), y in zip(tier_models, y_values):
        if y is None or not np.isfinite(y):
            continue
        x = _tier_x(family, tier)
        color = _point_color(model_id)
        ax.scatter(
            x,
            y,
            c=color,
            s=55,
            edgecolors="black",
            linewidth=0.5,
            zorder=3,
        )
        xs.append(x)
        ys.append(float(y))

    if len(xs) >= 2:
        order = np.argsort(xs)
        ax.plot(
            [xs[i] for i in order],
            [ys[i] for i in order],
            color=_line_color(family),
            linestyle="-",
            linewidth=linewidth,
            zorder=2,
        )


def _metric_value(
    model_id: str,
    metric_key: str,
    metrics_by_llm: dict[str, pd.Series],
    speaker_by_llm: dict[str, float],
) -> float | None:
    if metric_key == "speaker_R":
        val = speaker_by_llm.get(model_id)
        if val is None or not np.isfinite(val):
            return None
        return abs(float(val))
    if model_id not in metrics_by_llm:
        return None
    row = metrics_by_llm[model_id]
    if metric_key == "pbt_1":
        val = pd.to_numeric(row.get("pbt_1_both"), errors="coerce")
    elif metric_key == "pbt_2":
        val = pd.to_numeric(row.get("pbt_2_both"), errors="coerce")
    elif metric_key == "pba_l":
        val = pba_l_mean(row)
    elif metric_key == "pba_ft":
        val = pba_ft_mean(row)
    else:
        raise ValueError(f"Unknown metric_key: {metric_key!r}")
    return float(val) if pd.notna(val) and np.isfinite(val) else None


def _apply_pbt_pba_yaxis(ax, metric_key: str) -> None:
    """PB-T / PB-A panels: y upper bound at 0; ticks labeled left (bottom) and center (y=0)."""
    if metric_key not in ("pbt_1", "pbt_2", "pba_l", "pba_ft"):
        return
    ymin, _ = ax.get_ylim()
    ax.set_ylim(ymin, 0)
    ax.set_yticks([ymin, 0])
    ax.set_yticklabels(["left", "center"])


def _plot_metric_panel(
    ax,
    metric_key: str,
    metrics_by_llm: dict[str, pd.Series],
    speaker_by_llm: dict[str, float],
) -> None:
    for family, tier_models in FAMILIES.items():
        y_values = [
            _metric_value(model_id, metric_key, metrics_by_llm, speaker_by_llm)
            for model_id, _ in tier_models
        ]
        _plot_series(ax, tier_models, family, y_values)
    _annotate_family_labels(ax, metric_key, metrics_by_llm, speaker_by_llm)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    _apply_pbt_pba_yaxis(ax, metric_key)


def _add_group_ylabels(fig, outer) -> None:
    for group_label, rows in Y_LABEL_GROUPS:
        ax_g = fig.add_subplot(outer[rows[0] : rows[-1] + 1, 0])
        ax_g.axis("off")
        ax_g.text(
            0.55,
            0.5,
            group_label,
            rotation=90,
            ha="center",
            va="center",
            transform=ax_g.transAxes,
            fontsize=GROUP_LABEL_FONTSIZE,
            fontweight="bold",
        )


def build_llm_size_correlation_plot() -> Path:
    metrics = load_model_metrics()
    reg = load_regression_coefficients()
    metrics_by_llm = _metrics_lookup(metrics)
    speaker_by_llm = _regression_lookup(reg)

    n_panels = len(METRIC_PANELS)
    fig = plt.figure(figsize=(8, 9))
    outer = fig.add_gridspec(
        nrows=n_panels,
        ncols=2,
        width_ratios=[LEFT_LABEL_WIDTH, 1.0],
        hspace=0.22,
    )
    _add_group_ylabels(fig, outer)

    axes: list[plt.Axes] = []
    for i, (metric_key, ylabel) in enumerate(METRIC_PANELS):
        ax = fig.add_subplot(outer[i, 1], sharex=axes[0] if axes else None)
        _plot_metric_panel(ax, metric_key, metrics_by_llm, speaker_by_llm)
        ax.set_ylabel(ylabel, fontsize=INNER_YLABEL_FONTSIZE)
        axes.append(ax)

    fig.subplots_adjust(top=0.86, bottom=0.06)

    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False, bottom=False)

    axes[-1].set_xticks(list(range(N_TIERS)))
    axes[-1].set_xticklabels(TIER_LABELS)
    axes[-1].set_xlim(-0.55, N_TIERS - 1 + 0.35)

    fig.legend(
        handles=_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=4,
        framealpha=1,
        fontsize=8,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    out_dir = Path("output/images")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "llm_size_correlation.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"RQ3 size correlation plot saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    build_llm_size_correlation_plot()
