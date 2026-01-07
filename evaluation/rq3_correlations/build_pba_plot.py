"""
Build political compass plot showing LLM positions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.lines as mlines
import sys

# Add evaluation directory to path to allow importing utils
script_dir = Path(__file__).resolve().parent
evaluation_dir = script_dir.parent
if str(evaluation_dir) not in sys.path:
    sys.path.insert(0, str(evaluation_dir))

from utils import target_llms, llm_colors, opinion_llms, llm_info, PROMPT_TYPE

# Constants
COMPASS_SCALE_OFFSET = 2.5
COMPASS_SCALE_DIVISOR = 1.5
PLOT_LIMITS = (-1.5, 1.5)
QUADRANT_COLORS = {
    'authoritarian_left': '#EE7D79',
    'authoritarian_right': '#5697DF',
    'libertarian_left': '#ADEB9F',
    'libertarian_right': '#BA9CE7'
}


def _load_compass_data() -> tuple:
    """
    Load compass matrix and statement metadata.
    
    Returns:
        Tuple of (compass DataFrame, statement metadata DataFrame)
    """
    df_hpc = pd.read_csv(f"data/claim_matrices/{PROMPT_TYPE}/no_persona_compass.csv", index_col=0)
    df_api = pd.read_csv(f"data/claim_matrices/{PROMPT_TYPE}/no_persona_api_compass.csv", index_col=0)
    df_matrix = pd.concat([df_hpc, df_api], axis=0)
    
    df_statements = pd.read_csv("data/political_compass_statements.csv")
    
    return df_matrix, df_statements


def _calculate_model_scores(df_matrix: pd.DataFrame, df_statements: pd.DataFrame) -> dict:
    """
    Calculate economic and social scores for each model.
    
    Args:
        df_matrix: Compass matrix DataFrame
        df_statements: Statement metadata DataFrame
        
    Returns:
        Dictionary mapping model names to their scores
    """
    # Create mapping from statement index to domain
    statement_id_to_domain = {idx: row["Domain"] for idx, row in df_statements.iterrows()}
    
    # Separate statements by domain
    economic_statements = [sid for sid, domain in statement_id_to_domain.items() if domain == "Economic"]
    social_statements = [sid for sid, domain in statement_id_to_domain.items() if domain == "Social"]
    
    # Check which statements are actually in the matrix
    economic_statements_in_matrix = [
        sid for sid in economic_statements if str(sid) in df_matrix.columns
    ]
    social_statements_in_matrix = [
        sid for sid in social_statements if str(sid) in df_matrix.columns
    ]
    
    # Calculate scores for each model
    model_scores = {}
    for model in df_matrix.index:
        economic_scores = df_matrix.loc[
            model, [str(sid) for sid in economic_statements_in_matrix]
        ].dropna()
        social_scores = df_matrix.loc[
            model, [str(sid) for sid in social_statements_in_matrix]
        ].dropna()
        
        economic_mean = economic_scores.mean() if len(economic_scores) > 0 else np.nan
        social_mean = social_scores.mean() if len(social_scores) > 0 else np.nan
        
        # Map from 0-4 scale to -1 to +1 scale
        economic_mapped = (
            (economic_mean - COMPASS_SCALE_OFFSET) / COMPASS_SCALE_DIVISOR
            if not np.isnan(economic_mean) else np.nan
        )
        social_mapped = (
            (social_mean - COMPASS_SCALE_OFFSET) / COMPASS_SCALE_DIVISOR
            if not np.isnan(social_mean) else np.nan
        )
        
        model_scores[model] = {
            'economic': economic_mapped,
            'social': social_mapped
        }
    
    return model_scores


def build_pba_plot():
    """
    Create political compass plot showing LLM positions.
    
    The plot visualizes where different LLMs position themselves in political
    space based on their responses to political compass statements.
    """
    # Load data
    df_matrix, df_statements = _load_compass_data()
    model_scores = _calculate_model_scores(df_matrix, df_statements)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(*PLOT_LIMITS)
    ax.set_ylim(*PLOT_LIMITS)
    
    # Color the quadrants
    ax.fill_between([-1.2, 0], [0, 0], [1.2, 1.2], color=QUADRANT_COLORS['authoritarian_left'], alpha=0.3)
    ax.fill_between([0, 1.2], [0, 0], [1.2, 1.2], color=QUADRANT_COLORS['authoritarian_right'], alpha=0.3)
    ax.fill_between([-1.2, 0], [-1.2, -1.2], [0, 0], color=QUADRANT_COLORS['libertarian_left'], alpha=0.3)
    ax.fill_between([0, 1.2], [-1.2, -1.2], [0, 0], color=QUADRANT_COLORS['libertarian_right'], alpha=0.3)

    # Add axis labels outside the graph
    ax.text(0, 1.3, 'Authoritarian', ha='center', va='center', color='black', fontsize=12, fontweight='bold')
    ax.text(0, -1.3, 'Libertarian', ha='center', va='center', color='black', fontsize=12, fontweight='bold')
    ax.text(-1.3, 0, 'Left', ha='center', va='center', color='black', fontsize=12, fontweight='bold')
    ax.text(1.3, 0, 'Right', ha='center', va='center', color='black', fontsize=12, fontweight='bold')

    # Plot model points
    for model in target_llms + opinion_llms:
        if model in model_scores:
            scores = model_scores[model]
            if not np.isnan(scores['economic']) and not np.isnan(scores['social']):
                color = llm_colors.get(model, {'fillcolor': 'gray', 'edgecolor': 'black'})
                llm_name = llm_info.get(model, {}).get("name", model.capitalize())
                ax.scatter(
                    scores['economic'], scores['social'],
                    facecolor=color["fillcolor"],
                    edgecolor=color["edgecolor"],
                    linewidth=2,
                    marker='o',
                    s=100,
                    label=llm_name,
                    zorder=5
                )

    # Remove grid, ticks, axis labels, and spines
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')

    # Remove all spines (black border)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Create legend
    models_legend_entries = []
    for model in target_llms + opinion_llms:
        if model in model_scores and model in llm_colors:
            llm_name = llm_info.get(model, {}).get("name", model.capitalize())
            models_legend_entries.append(
                mlines.Line2D([], [], color=llm_colors[model]["edgecolor"], marker='o',
                             linestyle='None', markersize=8, label=llm_name)
            )
    
    plt.legend(handles=models_legend_entries, loc='upper left',
              title="Models", framealpha=1, facecolor="white",
              columnspacing=1.0, handletextpad=0.5)
    
    ax.set_facecolor('white')
    
    # Save plot
    output_dir = Path("output/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "political_compass_actual.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Political Compass plot saved to: {output_path}")

if __name__ == "__main__":
    build_pba_plot()