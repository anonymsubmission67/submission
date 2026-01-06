import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
import sys

# Add evaluation directory to path to allow importing utils
script_dir = Path(__file__).resolve().parent
evaluation_dir = script_dir.parent
if str(evaluation_dir) not in sys.path:
    sys.path.insert(0, str(evaluation_dir))

from utils import llm_info


def build_correlation_plot():

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

    # Prepare data for personas (concatenate all persona types, only LLMs with complete data)
    persona_data = df[df["persona"].isin(["democrat", "republican", "no_specific_political_view"])].copy()
    persona_metrics = persona_data[["pol_bias", "llm_size", "me_diff_1", "f1_1", "me_diff_2", "f1_2"]].dropna()

    # Filter to only LLMs that have complete data in both conditions
    llms_no_persona = set(no_persona_data[no_persona_data[["pol_bias", "llm_size", "me_diff_1", "f1_1", "me_diff_2", "f1_2"]].notna().all(axis=1)]["llm"])
    llms_persona = set(persona_data[persona_data[["pol_bias", "llm_size", "me_diff_1", "f1_1", "me_diff_2", "f1_2"]].notna().all(axis=1)]["llm"])
    common_llms = llms_no_persona.intersection(llms_persona)

    print(f"LLMs with complete data in both conditions: {sorted(common_llms)}")

    # Filter to only common LLMs
    no_persona_metrics = no_persona_data[no_persona_data["llm"].isin(common_llms)][["pol_bias", "llm_size", "me_diff_1", "f1_1", "me_diff_2", "f1_2"]]
    persona_metrics = persona_data[persona_data["llm"].isin(common_llms)][["pol_bias", "llm_size", "me_diff_1", "f1_1", "me_diff_2", "f1_2"]]

    print(f"No Persona data shape: {no_persona_metrics.shape}")
    print(f"Persona data shape: {persona_metrics.shape}")

    # Create separate correlation plots with triangular heatmaps
    output_dir = Path("output/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: No Persona correlations
    if len(no_persona_metrics) > 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))  # Wider, shorter figure
        corr_no_persona = no_persona_metrics.corr(method='pearson')
        

        # Create triangular mask
        mask = np.triu(np.ones_like(corr_no_persona, dtype=bool))
        
        corr_no_persona[mask] = np.nan
        corr_subset = corr_no_persona.drop('f1_2', axis=1).drop("pol_bias", axis=0)

        map_labels = {
            "pol_bias": "Political Bias",
            "llm_size": "LLM Size",
            "me_diff_1": "MED (without)",
            "f1_1": "F1 (without)",
            "me_diff_2": "MED (with)",
            "f1_2": "F1 (with)",
        }

        # Use imshow for better control over aspect ratio
        im = ax.imshow(corr_subset, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

        # Set ticks and labels (adjusted for subset)
        ax.set_xticks(range(len(corr_subset.columns)))
        ax.set_yticks(range(len(corr_subset.index)))
        ax.set_xticklabels(corr_subset.columns.map(map_labels), rotation=0, fontsize=14)
        ax.set_yticklabels(corr_subset.index.map(map_labels), rotation=0, fontsize=14)
        
        # Add annotations with larger font (skip NaN values)
        for i in range(len(corr_subset.index)):
            for j in range(len(corr_subset.columns)):
                if not pd.isna(corr_subset.iloc[i, j]):
                    text = ax.text(j, i, f'{corr_subset.iloc[i, j]:.3f}',
                                ha="center", va="center", color="black", fontsize=12)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8, aspect=40, pad=0.1)
        
        # Remove black frame
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # ax.set_title('No Persona Correlations', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / "correlation_no_persona.png", dpi=300, bbox_inches='tight')
        # plt.show()
        print("No Persona correlation plot saved to: output/images/correlation_no_persona.png")
    else:
        print("Insufficient data for No Persona correlations")

    # Plot 2: Persona correlations
    if len(persona_metrics) > 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))  # Wider, shorter figure
        corr_persona = persona_metrics.corr(method='pearson')
        
        map_labels = {
            "pol_bias": "Political Bias",
            "llm_size": "LLM Size",
            "me_diff_1": "MED (without)",
            "f1_1": "F1 (without)",
            "me_diff_2": "MED (with)",
            "f1_2": "F1 (with)",
        }
        
        # Create triangular mask
        mask = np.triu(np.ones_like(corr_persona, dtype=bool))
        
        corr_persona[mask] = np.nan
        corr_subset = corr_persona.drop('f1_2', axis=1).drop("pol_bias", axis=0)
        
        # Use imshow for better control over aspect ratio
        im = ax.imshow(corr_subset, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

        # Set ticks and labels (adjusted for subset)
        ax.set_xticks(range(len(corr_subset.columns)))
        ax.set_yticks(range(len(corr_subset.index)))
        ax.set_xticklabels(corr_subset.columns.map(map_labels), rotation=0, fontsize=14)
        ax.set_yticklabels(corr_subset.index.map(map_labels), rotation=0, fontsize=14)
        
        # Add annotations with larger font (skip NaN values)
        for i in range(len(corr_subset.index)):
            for j in range(len(corr_subset.columns)):
                if not pd.isna(corr_subset.iloc[i, j]):
                    text = ax.text(j, i, f'{corr_subset.iloc[i, j]:.3f}',
                                ha="center", va="center", color="black", fontsize=12)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8, aspect=40, pad=0.1)
        
        # Remove black frame
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # ax.set_title('Persona Correlations (All Types)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / "correlation_persona.png", dpi=300, bbox_inches='tight')
        # plt.show()
        print("Persona correlation plot saved to: output/images/correlation_persona.png")
    else:
        print("Insufficient data for Persona correlations")

    print(f"No Persona: {len(no_persona_metrics)} samples")
    print(f"Persona: {len(persona_metrics)} samples")

if __name__ == "__main__":
    build_correlation_plot()