"""
Build a PB-T plot for no_persona and for economic vs social axes.

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
import sys

# Add evaluation directory to path to allow importing utils
script_dir = Path(__file__).resolve().parent
evaluation_dir = script_dir.parent
if str(evaluation_dir) not in sys.path:
    sys.path.insert(0, str(evaluation_dir))

from utils import target_llms, llm_colors, opinion_llms, llm_info, PROMPT_TYPE

# Constants
LABEL_TO_NUM = {
    "pants-fire": 0, "false": 1, "mostly-false": 2,
    "half-true": 3, "mostly-true": 4, "true": 5,
}
PROMPTS = ["1", "2"]
PLOT_LIMITS = {
    'x_min': -1.0, 'x_max': 0.5,
    'y_min': -1.0, 'y_max': 0.5
}


def _calculate_me_differences(prompt: str) -> pd.DataFrame:
    """
    Calculate ME differences (Republican - Democrat) for economic and social axes.
    """
    # Load data
    hpc_path = f"data/claim_matrices/{PROMPT_TYPE}/no_persona_{prompt}_mean.csv"
    api_path = f"data/claim_matrices/{PROMPT_TYPE}/no_persona_api_{prompt}_mean.csv"
    
    df_hpc = pd.read_csv(hpc_path, index_col=0)
    df_api = pd.read_csv(api_path, index_col=0)
    df_long = pd.concat([df_hpc, df_api], axis=0)

    # Merge with claims metadata
    claims = pd.read_csv("data/claims_metadata.csv").set_index("claim_id")
    df_long = df_long.T.merge(
        claims[["party", "label", "axis"]],
        left_index=True,
        right_index=True,
        how="left"
    )
    df_long["label"] = df_long["label"].map(LABEL_TO_NUM)

    # Calculate Mean Error (subtract ground truth label)
    for col in df_long.columns:
        if col not in ["party", "label", "axis"]:
            df_long[col] = df_long[col] - df_long["label"]

    df_long = df_long.drop(columns=["label"])

    # Calculate mean bias by party and axis
    bias_by_party = df_long.groupby(["party", "axis"]).mean().T

    # Calculate ME differences (Republican - Democrat)
    me_diff_economic = bias_by_party[("Republican", "economic")] - bias_by_party[("Democrat", "economic")]
    me_diff_social = bias_by_party[("Republican", "social")] - bias_by_party[("Democrat", "social")]

    return pd.DataFrame({
        "me_diff_economic": me_diff_economic,
        "me_diff_social": me_diff_social
    })


def build_no_persona_pbt_plot():
    """
    Create plot showing ME differences for economic vs social axes.

    """
    out_dir = Path("output/images")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Calculate ME differences for both prompts
    me_diff_results = {}
    for prompt in PROMPTS:
        me_diff_results[f"prompt_{prompt}"] = _calculate_me_differences(prompt)

    # Combine into single DataFrame
    me_diff = pd.concat(me_diff_results, axis=1, keys=['prompt_1', 'prompt_2'])
    me_diff.columns = [f"{col[1]}_{col[0]}" for col in me_diff.columns]

    # Create figure
    plt.style.use('default')
    plt.figure(figsize=(8, 8))

    # Plot data for both prompt types
    for prompt_type in PROMPTS:
        for llm in target_llms + opinion_llms:
            # Skip if LLM not in data
            if llm not in me_diff.index:
                continue
                
            x_val = me_diff.loc[llm, f"me_diff_economic_prompt_{prompt_type}"]
            y_val = me_diff.loc[llm, f"me_diff_social_prompt_{prompt_type}"]
            
            if pd.isna(x_val) or pd.isna(y_val):
                continue
                
            color = llm_colors.get(llm, {'fillcolor': 'gray', 'edgecolor': 'gray'})
            
            # Use different facecolor for different prompts
            # Prompt 1 (party-agnostic): use fillcolor
            # Prompt 2 (party-aware): use edgecolor
            facecolor = color["fillcolor"] if prompt_type == "1" else color["edgecolor"]

            plt.scatter(
                x_val, y_val,
                facecolor=facecolor,
                edgecolor=color["edgecolor"],
                linewidth=2,
                marker='o',
                s=70
            )

    # Configure axes
    ax = plt.gca()
    plt.grid(False)
    plt.xlim(PLOT_LIMITS['x_min'], PLOT_LIMITS['x_max'])
    plt.ylim(PLOT_LIMITS['y_min'], PLOT_LIMITS['y_max'])

    # Remove tick marks
    plt.tick_params(axis='both', which='both', length=0)
    plt.xticks([])
    plt.yticks([])

    # Remove all standard spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Draw arrows for axes
    x_min, x_max = PLOT_LIMITS['x_min'], PLOT_LIMITS['x_max']
    y_min, y_max = PLOT_LIMITS['y_min'], PLOT_LIMITS['y_max']
    
    plt.annotate('', xy=(x_min*0.9, 0), xytext=(0, 0), 
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    plt.annotate('', xy=(x_max*0.9, 0), xytext=(0, 0), 
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    plt.annotate('', xy=(0, y_min*0.9), xytext=(0, 0), 
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    plt.annotate('', xy=(0, y_max*0.9), xytext=(0, 0), 
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Add axis labels
    plt.text(x_max*0.85, y_min*0.05, 'right', ha='center', va='center', fontsize=12)
    plt.text(x_min*0.85, y_min*0.05, 'left', ha='center', va='center', fontsize=12)
    plt.text(x_max*0.25, y_max*0.85, 'authoritarian', ha='center', va='center', fontsize=12)
    plt.text(x_max*0.20, y_min*0.85, 'libertarian', ha='center', va='center', fontsize=12)
    plt.text(x_max*0.45, y_min*0.03, 'economic', ha='center', va='center', fontsize=12)
    plt.text(x_max*0.03, y_min*0.5, 'social', ha='center', va='center', fontsize=12, rotation=90)

    # Create legends
    # First legend: Models
    models_legend_entries = []
    for llm in target_llms + opinion_llms:
        llm_name = llm_info.get(llm, {}).get("name", llm.capitalize())
        models_legend_entries.append(
            mlines.Line2D([], [], color=llm_colors[llm]["edgecolor"], marker='o', 
                         linestyle='None', markersize=8, label=llm_name)
        )

    # Second legend: Prompt Types
    prompt_legend_entries = [
        mlines.Line2D([], [], color='gray', marker='o', linestyle='None', 
                     markersize=8, label='party-aware'),
        mlines.Line2D([], [], color='gray', marker='o', linestyle='None', 
                     markersize=8, markerfacecolor='white', markeredgecolor='gray',
                     label='party-agnostic')
    ]

    # Add legends
    legend1 = plt.legend(handles=models_legend_entries, loc='upper left', 
                        title="Models", framealpha=1, facecolor="white",
                        columnspacing=1.0, handletextpad=0.5)
    plt.gca().add_artist(legend1)
    legend2 = plt.legend(handles=prompt_legend_entries, loc='upper left', 
                        bbox_to_anchor=(0, 0.55), title="Prompt Types", 
                        framealpha=1, facecolor="white", 
                        columnspacing=1.0, handletextpad=0.5)

    # Save figure
    output_path = out_dir / "political_compass_plot.png"
    plt.tight_layout()
    plt.savefig(output_path, format='png', dpi=300)
    print(f"Political compass plot saved to: {output_path}")

if __name__ == "__main__":
    build_no_persona_pbt_plot()