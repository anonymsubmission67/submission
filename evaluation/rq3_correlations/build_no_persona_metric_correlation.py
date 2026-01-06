import pandas as pd
import numpy as np
import json
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add evaluation directory to path to allow importing utils
script_dir = Path(__file__).resolve().parent
evaluation_dir = script_dir.parent
if str(evaluation_dir) not in sys.path:
    sys.path.insert(0, str(evaluation_dir))

from utils import llm_info, llm_colors, target_llms, opinion_llms


def build_no_persona_metric_correlation():
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

    print("=== CORRELATION HEATMAPS WITH SIGNIFICANCE STARS (NO PERSONA) ===")
    print(f"Total data points: {len(df)}")
    print(f"Data shape: {df.shape}")
    print()

    # Prepare data for no_persona (only LLMs with complete data)
    no_persona_data = df[df["persona"] == "no_persona"].copy()

    # Create output directory
    output_dir = Path("output/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    def create_combined_correlation_heatmaps(prompt1_data, prompt2_data):
        """Create combined correlation heatmaps with two subplots side by side"""
        
        if len(prompt1_data) < 3 or len(prompt2_data) < 3:
            print("Insufficient data for correlation heatmaps")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
        
        # Remove space between subplots
        plt.subplots_adjust(wspace=0.05)
        
        # Process both prompts
        datasets = [
            (prompt1_data, ax1, "party-agnostic"),
            (prompt2_data, ax2, "party-aware")
        ]
        
        for data_subset, ax, prompt_name in datasets:
            # Calculate correlation matrix
            corr_matrix = data_subset.corr(method='pearson')
            
            # Create triangular mask (lower triangle only)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            corr_matrix[mask] = np.nan
            
            # Drop specific rows/columns based on prompt type
            if prompt_name == "party-agnostic":
                corr_subset = corr_matrix.drop('f1_1', axis=0).drop("llm_size", axis=1)
            else:  # Prompt 2
                corr_subset = corr_matrix.drop('f1_2', axis=0).drop("llm_size", axis=1)
            
            # Labels mapping
            map_labels = {
                "f1_1": "F1",
                "f1_2": "F1", 
                "pol_bias": "PB-A",
                "me_diff_1": "PB-T",
                "me_diff_2": "PB-T",
                "llm_size": "LLM Size"
            }
            
            cmap = plt.cm.gray.reversed()
            cmap = plt.cm.colors.ListedColormap(cmap(np.linspace(0.0, 0.5, 256)))

            # Use imshow for better control over aspect ratio - use lighter grayscale colormap
            im = ax.imshow(corr_subset, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
            
            # Set ticks and labels
            ax.set_xticks(range(len(corr_subset.columns)))
            ax.set_yticks(range(len(corr_subset.index)))
            ax.set_xticklabels([map_labels.get(col, col) for col in corr_subset.columns], 
                               rotation=0, fontsize=15)
            
            # Only show y-tick labels for Prompt 1 (left plot)
            if prompt_name == "party-agnostic":
                ax.set_yticklabels([map_labels.get(idx, idx) for idx in corr_subset.index], 
                                   rotation=0, fontsize=15)
            else:  # Prompt 2 - remove y-tick labels for more space
                ax.set_yticklabels([])
            
            # Calculate p-values for each correlation pair
            def calculate_p_value(x, y):
                """Calculate p-value for correlation coefficient"""
                if len(x) < 3:
                    return np.nan
                mask = ~(np.isnan(x) | np.isnan(y))
                x_clean = x[mask]
                y_clean = y[mask]
                if len(x_clean) < 3:
                    return np.nan
                _, _, _, p_value, _ = stats.linregress(x_clean, y_clean)
                return p_value
            
            # Add annotations with correlation coefficients and significance stars
            for i in range(len(corr_subset.index)):
                for j in range(len(corr_subset.columns)):
                    if not pd.isna(corr_subset.iloc[i, j]):
                        # Get the actual data for p-value calculation
                        row_name = corr_subset.index[i]
                        col_name = corr_subset.columns[j]
                        
                        x_data = data_subset[col_name]
                        y_data = data_subset[row_name]
                        
                        p_value = calculate_p_value(x_data, y_data)
                        
                        # Format the annotation text
                        corr_val = corr_subset.iloc[i, j]
                        
                        # Add significance symbol above the coefficient
                        if not pd.isna(p_value) and p_value < 0.00833333:
                            annotation_text = f'*\n{corr_val:.3f}'
                        else:
                            annotation_text = f'-\n{corr_val:.3f}'
                        
                        text = ax.text(j, i, annotation_text,
                                     ha="center", va="center", color="black", fontsize=15)
            
            # Remove black frame
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Add title
            if prompt_name == "party-agnostic":
                ax.set_title(f'party-agnostic', fontsize=16, pad=15)
            else:
                ax.set_title(f'party-aware', fontsize=16, pad=15)

        # # Add single colorbar for both subplots (positioned lower with more space)
        # cbar = fig.colorbar(im, ax=[ax1, ax2], orientation='horizontal', shrink=0.8, aspect=40, pad=0.2)
        # cbar.ax.tick_params(labelsize=14)  # Increase colorbar label font size

        # Adjust layout manually to prevent colorbar from being pulled into plot
        plt.subplots_adjust(bottom=0.15)  # Leave space for colorbar
        plt.savefig(output_dir / "correlation_no_persona_metrics.png", dpi=300, bbox_inches='tight')
        print("Combined correlation plots saved to: output/images/correlation_combined_significance.png")

    # Prompt 1: F1, Political Bias, MED, LLM Size
    prompt1_data = no_persona_data[["f1_1", "pol_bias", "me_diff_1", "llm_size"]].dropna()
    print(f"Prompt 1 data points: {len(prompt1_data)}")

    # Prompt 2: F1, Political Bias, MED, LLM Size  
    prompt2_data = no_persona_data[["f1_2", "pol_bias", "me_diff_2", "llm_size"]].dropna()
    print(f"Prompt 2 data points: {len(prompt2_data)}")

    # Create combined heatmap with two subplots
    create_combined_correlation_heatmaps(prompt1_data, prompt2_data)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Created combined correlation heatmap with two subplots:")
    print("1. Left: party-agnostic: F1, Political Bias, MED, LLM Size")
    print("2. Right: party-aware: F1, Political Bias, MED, LLM Size")
    print()
    print("Significance stars (*) indicate p < 0.0555556")
    print("H0: β1 = 0 (no linear relationship)")
    print("H1: β1 ≠ 0 (linear relationship exists)")

if __name__ == "__main__":
    build_no_persona_metric_correlation()
