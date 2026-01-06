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


def build_pba_plot():
    # Load the compass matrix data
    df_hpc = pd.read_csv(f"data/claim_matrices/{PROMPT_TYPE}/no_persona_compass.csv", index_col=0)
    df_api = pd.read_csv(f"data/claim_matrices/{PROMPT_TYPE}/no_persona_api_compass.csv", index_col=0)
    df_matrix = pd.concat([df_hpc, df_api], axis=0)

    # Load statement metadata
    df_statements = pd.read_csv("data/political_compass_statements.csv")

    # Create mapping from statement index to domain
    statement_id_to_domain = {}
    for idx, row in df_statements.iterrows():
        statement_id_to_domain[idx] = row["Domain"]

    # Separate statements by domain
    economic_statements = [sid for sid, domain in statement_id_to_domain.items() if domain == "Economic"]
    social_statements = [sid for sid, domain in statement_id_to_domain.items() if domain == "Social"]

    print(f"Available statement IDs in matrix: {list(df_matrix.columns)}")
    print(f"Economic statement IDs: {economic_statements}")
    print(f"Social statement IDs: {social_statements}")

    # Check which statements are actually in the matrix
    economic_statements_in_matrix = [sid for sid in economic_statements if str(sid) in df_matrix.columns]
    social_statements_in_matrix = [sid for sid in social_statements if str(sid) in df_matrix.columns]

    print(f"Economic statements in matrix: {economic_statements_in_matrix}")
    print(f"Social statements in matrix: {social_statements_in_matrix}")

    print(f"Economic statements: {len(economic_statements_in_matrix)}")
    print(f"Social statements: {len(social_statements_in_matrix)}")

    # Calculate mean scores for each domain per model
    model_scores = {}

    for model in df_matrix.index:
        # Get economic and social scores using the statements that are actually in the matrix
        economic_scores = df_matrix.loc[model, [str(sid) for sid in economic_statements_in_matrix]].dropna()
        social_scores = df_matrix.loc[model, [str(sid) for sid in social_statements_in_matrix]].dropna()
        
        # Calculate means
        economic_mean = economic_scores.mean() if len(economic_scores) > 0 else np.nan
        social_mean = social_scores.mean() if len(social_scores) > 0 else np.nan
        
        # Map from 0-4 scale to -1 to +1 scale
        # Original: 1=strongly disagree, 2=disagree, 3=agree, 4=strongly agree
        # Mapped: -1=strongly disagree, -0.33=disagree, +0.33=agree, +1=strongly agree
        economic_mapped = (economic_mean - 2.5) / 1.5 if not np.isnan(economic_mean) else np.nan
        social_mapped = (social_mean - 2.5) / 1.5 if not np.isnan(social_mean) else np.nan
        
        model_scores[model] = {
            'economic': economic_mapped,
            'social': social_mapped,
            'economic_raw': economic_mean,
            'social_raw': social_mean
        }
        
        print(f"{model}: Economic={economic_mapped:.3f}, Social={social_mapped:.3f}")

    # Create the political compass plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set up the compass background (extended to accommodate outside labels)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    # Draw the compass quadrants (no black lines)
    # ax.axhline(y=0, color='black', linewidth=0.9, alpha=0.3)
    # ax.axvline(x=0, color='black', linewidth=0.9, alpha=0.3)

    # Color the quadrants with hex colors
    # Top-left: Authoritarian Left (EE7D79)
    ax.fill_between([-1.2, 0], [0, 0], [1.2, 1.2], color='#EE7D79', alpha=0.3)
    # Top-right: Authoritarian Right (5697DF)  
    ax.fill_between([0, 1.2], [0, 0], [1.2, 1.2], color='#5697DF', alpha=0.3)
    # Bottom-left: Libertarian Left (ADEB9F)
    ax.fill_between([-1.2, 0], [-1.2, -1.2], [0, 0], color='#ADEB9F', alpha=0.3)
    # Bottom-right: Libertarian Right (BA9CE7)
    ax.fill_between([0, 1.2], [-1.2, -1.2], [0, 0], color='#BA9CE7', alpha=0.3)

    # Add axis labels outside the graph
    ax.text(0, 1.3, 'Authoritarian', ha='center', va='center', color='black', fontsize=12, fontweight='bold')
    ax.text(0, -1.3, 'Libertarian', ha='center', va='center', color='black', fontsize=12, fontweight='bold')
    ax.text(-1.3, 0, 'Left', ha='center', va='center', color='black', fontsize=12, fontweight='bold')
    ax.text(1.3, 0, 'Right', ha='center', va='center', color='black', fontsize=12, fontweight='bold')

    # Plot model points using consistent colors and markers
    # Only plot models that are in target_llms or opinion_llms
    for model in target_llms + opinion_llms:
        if model in model_scores:
            scores = model_scores[model]
            if not np.isnan(scores['economic']) and not np.isnan(scores['social']):
                # Use consistent colors from utils
                color = llm_colors.get(model, {'fillcolor': 'gray', 'edgecolor': 'black'})
                
                # Different sizes for opinion_llms vs target_llms
                size = 100 if model in opinion_llms else 100
                
                llm_name = llm_info.get(model, {}).get("name", model.capitalize())
                ax.scatter(scores['economic'], scores['social'], 
                        facecolor=color["fillcolor"],
                        edgecolor=color["edgecolor"],
                        linewidth=2,
                        marker='o', s=size,
                        label=llm_name, zorder=5)

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

    # Add legend similar to political compass plot
    models_legend_entries = []
    for model in target_llms + opinion_llms:
        if model in model_scores and model in llm_colors:
            llm_name = llm_info.get(model, {}).get("name", model.capitalize())
            models_legend_entries.append(mlines.Line2D([], [], color=llm_colors[model]["edgecolor"], marker='o', 
                                                    linestyle='None', markersize=8, 
                                                    label=llm_name))

    # Create the legend
    legend = plt.legend(handles=models_legend_entries, loc='upper left', 
                    title="Models", framealpha=1, facecolor="white",
                    columnspacing=1.0, handletextpad=0.5)

    ax.set_facecolor('white')

    # Save the plot
    output_dir = Path("output/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "political_compass_actual.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    print("\nPolitical Compass plot saved to: output/images/political_compass_actual.png")

if __name__ == "__main__":
    build_pba_plot()