import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
import matplotlib.lines as mlines
import sys

# Add evaluation directory to path to allow importing utils
script_dir = Path(__file__).resolve().parent
evaluation_dir = script_dir.parent
if str(evaluation_dir) not in sys.path:
    sys.path.insert(0, str(evaluation_dir))

from utils import llm_info, llm_colors, target_llms, opinion_llms


def build_scatter_plots():
        
    # Load the overview data
    with open("data/interim_results/overview_data.json", "r") as f:
        data = json.load(f)

    # Convert to DataFrame for easier manipulation
    df_data = []
    for item in data:
        # Get LLM size from utils
        llm_size = llm_info.get(item["llm"], {}).get("size_B", None)
        
        row = {
            "llm": item["llm"],
            "persona": item["persona"],
            "pol_bias": item["pol_bias"],
            "llm_size": llm_size,
            "me_diff_1": item["prompt_1"].get("me_diff", None),
            "f1_1": item["prompt_1"].get("f1", None),
            "me_diff_2": item["prompt_2"].get("me_diff", None),
            "f1_2": item["prompt_2"].get("f1", None)
        }
        df_data.append(row)

    df = pd.DataFrame(df_data)

    # Prepare data for no_persona (only LLMs with complete data)
    no_persona_data = df[df["persona"] == "no_persona"].copy()
    no_persona_metrics = no_persona_data[["pol_bias", "llm_size", "me_diff_1", "f1_1", "me_diff_2", "f1_2"]].dropna()
    no_persona_scatter = no_persona_data[["llm", "pol_bias", "me_diff_2"]].dropna()

    # Prepare data for personas (concatenate all persona types, only LLMs with complete data)
    persona_data = df[df["persona"].isin(["democrat", "republican", "no_specific_political_view"])].copy()
    persona_metrics = persona_data[["pol_bias", "llm_size", "me_diff_1", "f1_1", "me_diff_2", "f1_2"]].dropna()
    persona_scatter = persona_data[["llm", "pol_bias", "me_diff_2"]].dropna()

    print(f"No Persona data shape: {no_persona_metrics.shape}")
    print(f"Persona data shape: {persona_metrics.shape}")

    # Calculate common axis scales for both plots
    if len(no_persona_scatter) > 0 and len(persona_scatter) > 0:
        # Get min/max values for both datasets
        no_persona_x_min, no_persona_x_max = no_persona_scatter['pol_bias'].min(), no_persona_scatter['pol_bias'].max()
        no_persona_y_min, no_persona_y_max = no_persona_scatter['me_diff_2'].min(), no_persona_scatter['me_diff_2'].max()
        
        persona_x_min, persona_x_max = persona_scatter['pol_bias'].min(), persona_scatter['pol_bias'].max()
        persona_y_min, persona_y_max = persona_scatter['me_diff_2'].min(), persona_scatter['me_diff_2'].max()
        
        # Use the wider range for both plots
        common_x_min = min(no_persona_x_min, persona_x_min)
        common_x_max = max(no_persona_x_max, persona_x_max)
        common_y_min = min(no_persona_y_min, persona_y_min)
        common_y_max = max(no_persona_y_max, persona_y_max)
        
        # Add some padding
        x_padding = (common_x_max - common_x_min) * 0.1
        y_padding = (common_y_max - common_y_min) * 0.1
        
        common_x_min -= x_padding
        common_x_max += x_padding
        common_y_min -= y_padding
        common_y_max += y_padding
        
        print(f"Common axis ranges: x=[{common_x_min:.3f}, {common_x_max:.3f}], y=[{common_y_min:.3f}, {common_y_max:.3f}]")
    else:
        common_x_min = common_x_max = common_y_min = common_y_max = None

    # Create output directory
    output_dir = Path("output/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    # # Plot 1a: No Persona - Correlation heatmap only
    # if len(no_persona_metrics) > 1:
    #     fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        
    #     corr_no_persona = no_persona_metrics.corr(method='pearson')
        
    #     # Create triangular mask
    #     mask = np.triu(np.ones_like(corr_no_persona, dtype=bool))
    #     corr_no_persona[mask] = np.nan
    #     corr_subset = corr_no_persona.drop('f1_2', axis=1).drop("pol_bias", axis=0)

    #     map_labels = {
    #         "pol_bias": "Political Bias",
    #         "llm_size": "LLM Size",
    #         "me_diff_1": "TB without",
    #         "f1_1": "F1 (without)",
    #         "me_diff_2": "TB with",
    #         "f1_2": "F1 (with)",
    #     }

    #     # Use imshow for better control over aspect ratio - use grayscale colormap
    #     im = ax.imshow(corr_subset, cmap='gray',vmin=-1, vmax=1) #,  aspect = "auto"

    #     # Set ticks and labels (adjusted for subset)
    #     ax.set_xticks(range(len(corr_subset.columns)))
    #     ax.set_yticks(range(len(corr_subset.index)))
    #     ax.set_xticklabels(corr_subset.columns.map(map_labels), rotation=0, fontsize=14)
    #     ax.set_yticklabels(corr_subset.index.map(map_labels), rotation=0, fontsize=14)
        
    #     # Add annotations with larger font (skip NaN values)
    #     for i in range(len(corr_subset.index)):
    #         for j in range(len(corr_subset.columns)):
    #             if not pd.isna(corr_subset.iloc[i, j]):
    #                 text = ax.text(j, i, f'{corr_subset.iloc[i, j]:.3f}',
    #                             ha="center", va="center", color="black", fontsize=14)
        
    #     # Add colorbar
    #     plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8, aspect=40, pad=0.1)
        
    #     # Remove black frame
    #     for spine in ax.spines.values():
    #         spine.set_visible(False)

    #     plt.tight_layout()
    #     plt.savefig(output_dir / "correlation_no_persona_metrics.png", dpi=300, bbox_inches='tight')
    #     # plt.show()
    #     print("No Persona metrics correlation plot saved to: output/images/correlation_no_persona_metrics.png")

    # Plot 1b: No Persona - Two scatter plots (without source and with source)
    if len(no_persona_scatter) > 0:
        # Fixed total width, adjust plot area to leave space for legend
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5))

        fig.subplots_adjust(right=0.65, top=0.98, bottom=0.1, hspace=0.2)
        
        # Prepare data for both prompts
        no_persona_scatter_1 = no_persona_data[["llm", "pol_bias", "me_diff_1"]].dropna()
        no_persona_scatter_2 = no_persona_data[["llm", "pol_bias", "me_diff_2"]].dropna()
        
        # Calculate common axis scales for both plots
        all_pol_bias = pd.concat([no_persona_scatter_1['pol_bias'], no_persona_scatter_2['pol_bias']])
        all_me_diff_1 = no_persona_scatter_1['me_diff_1']
        all_me_diff_2 = no_persona_scatter_2['me_diff_2']
        
        # Get min/max values for both datasets
        x_min, x_max = all_pol_bias.min(), all_pol_bias.max()
        y1_min, y1_max = all_me_diff_1.min(), all_me_diff_1.max()
        y2_min, y2_max = all_me_diff_2.min(), all_me_diff_2.max()
        
        # Use the wider range for both plots
        y_min = min(y1_min, y2_min)
        y_max = max(y1_max, y2_max)
        
        # Add some padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        
        # Plot 1: Without source (top)
        unique_llms_1 = no_persona_scatter_1['llm'].unique()
        for llm in unique_llms_1:
            llm_data = no_persona_scatter_1[no_persona_scatter_1['llm'] == llm]
            
            # Get color and size from utils
            color_info = llm_colors.get(llm, {"edgecolor": "gray"})
            color = color_info["edgecolor"]
            
            # Get LLM size for point size (scale appropriately)
            llm_size = llm_info.get(llm, {}).get("size_B", 8)  # Default to 8B if not found
            point_size = max(20, min(200, llm_size * 2))  # Scale between 20-200, with 2x multiplier
            
            ax1.scatter(llm_data['pol_bias'], llm_data['me_diff_1'], 
                    c=color, s=point_size, edgecolors='black', linewidth=0.5)
        
        #ax1.set_xlabel('Political Bias', fontsize=12)
        ax1.set_ylabel('PB-T party-agnostic', fontsize=12)
        ax1.grid(True, alpha=0.3)
        # ax1.set_title('without source', fontsize=14, pad=10)
        
        # Set common axis limits
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        
        # Add correlation coefficient for prompt 1
        corr_coef_1 = no_persona_scatter_1['pol_bias'].corr(no_persona_scatter_1['me_diff_1'])
        ax1.text(0.025, 0.90, f'r = {corr_coef_1:.3f}', transform=ax1.transAxes, 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Plot 2: With source (bottom)
        unique_llms_2 = no_persona_scatter_2['llm'].unique()
        for llm in unique_llms_2:
            llm_data = no_persona_scatter_2[no_persona_scatter_2['llm'] == llm]
            
            # Get color and size from utils
            color_info = llm_colors.get(llm, {"edgecolor": "gray"})
            color = color_info["edgecolor"]
            
            # Get LLM size for point size (scale appropriately)
            llm_size = llm_info.get(llm, {}).get("size_B", 8)  # Default to 8B if not found
            point_size = max(20, min(200, llm_size * 2))  # Scale between 20-200, with 2x multiplier
            
            ax2.scatter(llm_data['pol_bias'], llm_data['me_diff_2'], 
                    c=color, s=point_size, edgecolors='black', linewidth=0.5)
        
        ax2.set_xlabel('PB-A', fontsize=12)
        ax2.set_ylabel('PB-T party-aware', fontsize=12)
        ax2.grid(True, alpha=0.3)
        # ax2.set_title('with source', fontsize=14, pad=10)
        
        # Set common axis limits
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        
        # Add correlation coefficient for prompt 2
        corr_coef_2 = no_persona_scatter_2['pol_bias'].corr(no_persona_scatter_2['me_diff_2'])
        ax2.text(0.025, 0.90, f'r = {corr_coef_2:.3f}', transform=ax2.transAxes, 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Create two legends for the bottom plot only
        # First legend: Model colors
        models_legend_entries_no_persona = []
        for llm in target_llms + opinion_llms:
            if llm in no_persona_scatter_2['llm'].values:  # Only include models that have data
                llm_name = llm_info.get(llm, {}).get("name", llm.capitalize())
                models_legend_entries_no_persona.append(mlines.Line2D([], [], color=llm_colors[llm]["edgecolor"], marker='o', 
                                                            linestyle='None', markersize=8, 
                                                            label=llm_name))

        # Second legend: Model sizes
        size_legend_entries = []
        size_examples = [4, 8, 20, 100]  # Example model sizes in billions
        for size_b in size_examples:
            point_size = max(20, min(200, size_b * 2))
            size_legend_entries.append(mlines.Line2D([], [], color='gray', marker='o', 
                                                linestyle='None', markersize=np.sqrt(point_size)/2, 
                                                label=f'{size_b}B                     '))

        # Create the first legend (Models) - positioned right outside the upper plot
        legend1_no_persona = ax1.legend(handles=models_legend_entries_no_persona, loc='center right', 
                                      bbox_to_anchor=(1.65, 0.15), title="Models", framealpha=1, facecolor="white", # 
                                    columnspacing=1.0, handletextpad=0.5)
        ax1.add_artist(legend1_no_persona)

        # Create the second legend (Model Sizes) - positioned right outside the lower plot
        legend2_no_persona = ax2.legend(handles=size_legend_entries, loc='lower right', 
                                     bbox_to_anchor=(1.647, -0.10), title="Model Size", framealpha=0.9, facecolor="white",
                                    columnspacing=1.0, handletextpad=0.5) # ,

        plt.savefig(output_dir / "correlation_no_persona.png", dpi=300) #, bbox_inches='tight'
        # plt.show()
        print("No Persona scatter plots saved to: output/images/correlation_no_persona.png")
    else:
        print("Insufficient data for No Persona plots")

    # Calculate correlation matrix for persona (keep calculation but don't plot)
    if len(persona_metrics) > 1:
        corr_persona = persona_metrics.corr(method='pearson')
        print("Persona correlation matrix calculated")

    # Plot 2: Persona - Only scatter plot with custom symbols
    if len(persona_scatter) > 0:
        # Fixed total width, adjust plot area to leave space for legend
        fig, ax = plt.subplots(1, 1, figsize=(7, 3.7))
        # Adjust subplot parameters to leave space for legend on the right
        fig.subplots_adjust(right=0.50)
        
        political_symbols = {
            'democrat': 'D',
            'republican': 'R',
            'no_specific_political_view': 'N'
        }
        
        unique_llms = persona_scatter['llm'].unique()
        
        for llm in unique_llms:
            llm_data = persona_scatter[persona_scatter['llm'] == llm]
            
            # Get LLM color from utils
            color_info = llm_colors.get(llm, {"edgecolor": "gray"})
            llm_color = color_info["edgecolor"]
            
            # Plot white circles with colored borders
            ax.scatter(llm_data['pol_bias'], llm_data['me_diff_2'], 
                    c='white', s=100, edgecolors=llm_color, linewidth=1)
            
            # Add text symbols in the center
            for _, row in llm_data.iterrows():
                # Determine political view from persona data
                persona_data = df[(df['llm'] == llm) & (df['pol_bias'] == row['pol_bias']) & (df['me_diff_2'] == row['me_diff_2'])]
                if not persona_data.empty:
                    persona_type = persona_data.iloc[0]['persona']
                    symbol = political_symbols.get(persona_type, '?')
                    
                    ax.text(row['pol_bias'], row['me_diff_2'], symbol, 
                        ha='center', va='center', fontsize=7, fontweight='bold',
                        color=llm_color)
        
        ax.set_xlabel('PB-A', fontsize=12)
        ax.set_ylabel('PB-T party-aware', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Set common axis limits
        if common_x_min is not None:
            ax.set_xlim(common_x_min, common_x_max)
            ax.set_ylim(common_y_min, common_y_max)
        
        # # Add correlation coefficient
        # corr_coef = persona_scatter['pol_bias'].corr(persona_scatter['me_diff_2'])
        # ax.text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=ax.transAxes, 
        #         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Create legends similar to political compass plot
        # First legend: Model colors
        models_legend_entries = []
        for llm in target_llms + opinion_llms:
            if llm in persona_scatter['llm'].values:  # Only include models that have data
                llm_name = llm_info.get(llm, {}).get("name", llm.capitalize())
                models_legend_entries.append(mlines.Line2D([], [], color=llm_colors[llm]["edgecolor"], marker='o', 
                                                        linestyle='None', markersize=5, 
                                                        label=llm_name))

        # Second legend: Political symbols
        political_legend_entries = [
            mlines.Line2D([], [], color='black', marker='o', linestyle='None', 
                        markersize=5, label='R - Republican'),
            mlines.Line2D([], [], color='black', marker='o', linestyle='None', 
                        markersize=5, label='D - Democrat'),
            mlines.Line2D([], [], color='black', marker='o', linestyle='None', 
                        markersize=5, label='N - No Viewpoint')
        ]

        # Create the first legend (Models) - positioned right outside the plot
        legend1 = plt.legend(handles=models_legend_entries, loc='center left', 
                            bbox_to_anchor=(1.02, 0.59), title="Models", framealpha=0.8, facecolor="white",
                            columnspacing=1.0, handletextpad=0.5, fontsize=7, title_fontsize=8)
        plt.gca().add_artist(legend1)

        # Create the second legend (Political Symbols) - positioned right outside below the first
        legend2 = plt.legend(handles=political_legend_entries, loc='center left', 
                            bbox_to_anchor=(1.02, 0.06), title="Political Viewpoints", 
                            framealpha=0.8, facecolor="white", 
                            columnspacing=1.0, handletextpad=0.4, fontsize=7, title_fontsize=8)

        plt.savefig(output_dir / "correlation_persona.png", dpi=300, bbox_inches='tight')
        # plt.show()
        print("Persona scatter plot saved to: output/images/correlation_persona.png")
    else:
        print("Insufficient data for Persona plots")

    # Print summary statistics
    print(f"\nSummary:")
    print(f"No Persona - Correlation between pol_bias and me_diff_2: {no_persona_scatter['pol_bias'].corr(no_persona_scatter['me_diff_2']):.3f}")
    print(f"Persona - Correlation between pol_bias and me_diff_2: {persona_scatter['pol_bias'].corr(persona_scatter['me_diff_2']):.3f}")

    print(f"\nNo Persona data points: {len(no_persona_scatter)}")
    print(f"Persona data points: {len(persona_scatter)}")

if __name__ == "__main__":
    build_scatter_plots()