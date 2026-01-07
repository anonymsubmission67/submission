"""
Build box plots showing PB-T differences by political view for personas.

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add evaluation directory to path to allow importing utils
script_dir = Path(__file__).resolve().parent
evaluation_dir = script_dir.parent
if str(evaluation_dir) not in sys.path:
    sys.path.insert(0, str(evaluation_dir))

from utils import calc_me_diff, calc_me_diff_personas, target_llms, llm_colors, llm_info, PROMPT_TYPE, PERSONAS_PATH

# Constants
POLITICAL_VIEW_MAPPING = {
    'Republican': 'R',
    'Democrat': 'D',
    'No Specific Political View': 'N'
}
SIGNIFICANCE_THRESHOLD_STRONG = 0.005  # p < 0.005: **
SIGNIFICANCE_THRESHOLD_WEAK = 0.025    # p < 0.025: *


def _load_no_persona_data() -> tuple:
    """
    Load and calculate ME differences for no-persona data.
    """
    no_persona_1_hpc = pd.read_csv(f"data/claim_matrices/{PROMPT_TYPE}/no_persona_1_mean.csv", index_col=0)
    no_persona_1_api = pd.read_csv(f"data/claim_matrices/{PROMPT_TYPE}/no_persona_api_1_mean.csv", index_col=0)
    no_persona_1 = pd.concat([no_persona_1_hpc, no_persona_1_api], axis=0)

    no_persona_2_hpc = pd.read_csv(f"data/claim_matrices/{PROMPT_TYPE}/no_persona_2_mean.csv", index_col=0)
    no_persona_2_api = pd.read_csv(f"data/claim_matrices/{PROMPT_TYPE}/no_persona_api_2_mean.csv", index_col=0)
    no_persona_2 = pd.concat([no_persona_2_hpc, no_persona_2_api], axis=0)

    no_persona_me_diffs_1 = calc_me_diff(no_persona_1)
    no_persona_me_diffs_2 = calc_me_diff(no_persona_2)
    
    return no_persona_me_diffs_1, no_persona_me_diffs_2


def _create_plot_data(prompt: str, no_persona_me_diffs: pd.Series) -> pd.DataFrame:
    """
    Create plot data for a specific prompt.
    
    Args:
        prompt: Prompt number ('1' or '2')
        no_persona_me_diffs: ME differences for no-persona baseline
        
    Returns:
        DataFrame with plot data
    """
    # Load personas metadata
    all_data = pd.read_csv(PERSONAS_PATH, index_col="id")
    
    # Merge data for each LLM
    for llm in target_llms:
        personas_me_diff = calc_me_diff_personas(llm)
        # Filter to only include personas from PERSONAS_PATH
        personas_me_diff_filtered = personas_me_diff[personas_me_diff.index.isin(all_data.index)]
        all_data = all_data.merge(
            personas_me_diff_filtered[[f"prompt_{prompt}"]].rename(columns={f"prompt_{prompt}": llm}),
            left_index=True, right_index=True, how="inner"
        )
    
    # Create plot data
    plot_data = []
    for llm in target_llms:
        for _, row in all_data.iterrows():
            political_view_short = POLITICAL_VIEW_MAPPING.get(
                row['political_view'], row['political_view']
            )
            
            plot_data.append({
                'llm': llm,
                'political_view': political_view_short,
                'prompt_value': row[llm],
                'prompt': f'Prompt {prompt}'
            })
    
    return pd.DataFrame(plot_data)


def _add_significance_stars(ax, llm_index: int, llm: str, significance_df: pd.DataFrame, 
                            prompt: str, y_position_factor: float):
    """
    Add significance stars to plot based on ANOVA results.
    
    Args:
        ax: Matplotlib axis
        llm_index: Index of LLM in target_llms list
        llm: LLM identifier
        significance_df: DataFrame with ANOVA significance results
        prompt: Prompt number ('1' or '2')
        y_position_factor: Factor for y-position of stars
    """
    sig_data = significance_df[
        (significance_df['llm'] == llm) &
        ((significance_df['prompt'] == int(prompt)) | (significance_df['prompt'] == prompt)) &
        (significance_df['axis'] == 'both')
    ]
    
    if not sig_data.empty:
        p_value = sig_data.iloc[0]['political_view']
        if pd.notna(p_value):
            y_pos = ax.get_ylim()[1] * y_position_factor
            x_pos = llm_index * 3 + 1
            
            if p_value < SIGNIFICANCE_THRESHOLD_STRONG:
                ax.text(x_pos, y_pos, '**', ha='center', va='center',
                       fontsize=16, fontweight='bold', color='black')
            elif p_value < SIGNIFICANCE_THRESHOLD_WEAK:
                ax.text(x_pos, y_pos, '*', ha='center', va='center',
                       fontsize=16, fontweight='bold', color='black')


def build_box_plots():
    """
    Create box plots showing ME differences by political view for personas.
    
    The plot shows side-by-side comparisons of party-agnostic (Prompt 1) and
    party-aware (Prompt 2) prompts, with reference lines for no-persona baselines
    and significance stars based on ANOVA results.
    """
    out_dir = Path("output/images")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    no_persona_me_diffs_1, no_persona_me_diffs_2 = _load_no_persona_data()
    significance_df = pd.read_csv("data/personas_significance_both.csv")

    # Create data for both prompts
    df_1 = _create_plot_data("1", no_persona_me_diffs_1)
    df_2 = _create_plot_data("2", no_persona_me_diffs_2)
    df_combined = pd.concat([df_1, df_2], ignore_index=True)


    # Create the combined plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 6))

    # Plot for Prompt 1 - group by LLM first, then political view
    df_prompt1 = df_combined[df_combined['prompt'] == 'Prompt 1']
    df_prompt1['x_label'] = df_prompt1['llm'] + '_' + df_prompt1['political_view']
    # Create palette with edgecolor for seaborn
    palette_edge = {llm: llm_colors[llm]["edgecolor"] for llm in target_llms}
    sns.boxplot(data=df_prompt1, x='x_label', y='prompt_value', hue='llm', palette=palette_edge, ax=ax1, 
                order=[f'{llm}_{view}' for llm in target_llms for view in ['D', 'N', 'R']])

    # Configure Prompt 1 plot
    _configure_prompt_plot(ax1, df_prompt1, no_persona_me_diffs_1, "PB-T party-agnostic", show_xlabels=False)
    
    # Configure Prompt 2 plot
    _configure_prompt_plot(ax2, df_prompt2, no_persona_me_diffs_2, "PB-T party-aware", show_xlabels=True)

    # Remove top spines for better readability of LLM names
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Add LLM names above each group
    for i, llm in enumerate(target_llms):
        llm_name = llm_info.get(llm, {}).get("name", llm.capitalize())
        
        # Add line breaks for long names (max 12 characters per line)
        if len(llm_name) > 11:
            # Find a good break point (prefer spaces or hyphens)
            break_point = 11
            for j in range(11, 0, -1):
                if llm_name[j] in [' ', '-', '_']:
                    break_point = j
                    break
            llm_name = llm_name[:break_point] + '\n' + llm_name[break_point+1:]
        
        ax1.text(i * 3 + 1, ax1.get_ylim()[1] * 0.9, llm_name, 
                ha='center', va='center', fontsize=14, fontweight='bold')
        # ax2.text(i * 3 + 1, ax2.get_ylim()[1] * 0.9, llm_name, 
        #          ha='center', va='center', fontsize=12, fontweight='bold')

    # Add significance stars
    for i, llm in enumerate(target_llms):
        _add_significance_stars(ax1, i, llm, significance_df, "1", 0.40)
        _add_significance_stars(ax2, i, llm, significance_df, "2", 0.50)

    plt.subplots_adjust(hspace=0.1)
    plt.tight_layout()
    
    output_path = out_dir / 'box_plot.png'
    plt.savefig(output_path, format='png', dpi=300)
    plt.close()
    print(f"Box plot saved to: {output_path}")


def _configure_prompt_plot(ax, df_prompt: pd.DataFrame, no_persona_me_diffs: pd.Series,
                           ylabel: str, show_xlabels: bool):
    """
    Configure a single prompt plot with box plots, reference lines, and formatting.
    
    Args:
        ax: Matplotlib axis
        df_prompt: DataFrame with plot data for one prompt
        no_persona_me_diffs: ME differences for no-persona baseline
        ylabel: Y-axis label
        show_xlabels: Whether to show x-axis labels
    """
    # Create x labels
    df_prompt['x_label'] = df_prompt['llm'] + '_' + df_prompt['political_view']
    
    # Create palette with edgecolor for seaborn
    palette_edge = {llm: llm_colors[llm]["edgecolor"] for llm in target_llms}
    
    # Plot box plots
    sns.boxplot(
        data=df_prompt,
        x='x_label',
        y='prompt_value',
        hue='llm',
        palette=palette_edge,
        ax=ax,
        order=[f'{llm}_{view}' for llm in target_llms for view in ['D', 'N', 'R']]
    )
    
    # Add horizontal lines for no_persona values
    for i, llm in enumerate(target_llms):
        no_persona_value = no_persona_me_diffs.loc[llm]
        color = llm_colors[llm]["edgecolor"]
        ax.axhline(
            y=no_persona_value,
            xmin=i*3/len(target_llms)/3,
            xmax=(i+1)*3/len(target_llms)/3,
            color=color,
            linestyle='--',
            linewidth=2,
            alpha=0.8
        )
    
    # Add vertical lines between LLM groups
    for i in range(1, len(target_llms)):
        ax.axvline(x=i*3-0.5, color='lightgray', linestyle='-', linewidth=1, alpha=0.7)
    
    # Configure axes
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend().remove()
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_ylim(None, 0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    
    # Configure x-axis
    if show_xlabels:
        ax.set_xlabel('Political Views', fontsize=12)
        ax.set_xticklabels(['D', 'N', 'R'] * len(target_llms))
    else:
        ax.set_xticklabels([])
        ax.set_xlabel('')
    
    # Remove top spine
    ax.spines['top'].set_visible(False)
    
    # Add LLM names above each group
    for i, llm in enumerate(target_llms):
        llm_name = llm_info.get(llm, {}).get("name", llm.capitalize())
        
        # Add line breaks for long names (max 12 characters per line)
        if len(llm_name) > 11:
            break_point = 11
            for j in range(11, 0, -1):
                if llm_name[j] in [' ', '-', '_']:
                    break_point = j
                    break
            llm_name = llm_name[:break_point] + '\n' + llm_name[break_point+1:]
        
        ax.text(
            i * 3 + 1,
            ax.get_ylim()[1] * 0.9,
            llm_name,
            ha='center',
            va='center',
            fontsize=14,
            fontweight='bold'
        )


if __name__ == "__main__":
    build_box_plots()