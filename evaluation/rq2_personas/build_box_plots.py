import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add evaluation directory to path to allow importing utils
script_dir = Path(__file__).resolve().parent
evaluation_dir = script_dir.parent
if str(evaluation_dir) not in sys.path:
    sys.path.insert(0, str(evaluation_dir))

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from utils import calc_me_diff, calc_me_diff_personas, target_llms, llm_colors, opinion_llms, llm_info, PROMPT_TYPE, PERSONAS_PATH


def build_box_plots():

    out_dir = Path("output/images")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load no_persona data for both prompts

    no_persona_1_hpc = pd.read_csv(f"data/claim_matrices/{PROMPT_TYPE}/no_persona_1_mean.csv", index_col=0)
    no_persona_1_api = pd.read_csv(f"data/claim_matrices/{PROMPT_TYPE}/no_persona_api_1_mean.csv", index_col=0)
    no_persona_1 = pd.concat([no_persona_1_hpc, no_persona_1_api], axis=0)

    no_persona_2_hpc = pd.read_csv(f"data/claim_matrices/{PROMPT_TYPE}/no_persona_2_mean.csv", index_col=0)
    no_persona_2_api = pd.read_csv(f"data/claim_matrices/{PROMPT_TYPE}/no_persona_api_2_mean.csv", index_col=0)
    no_persona_2 = pd.concat([no_persona_2_hpc, no_persona_2_api], axis=0)

    no_persona_me_diffs_1 = calc_me_diff(no_persona_1)
    no_persona_me_diffs_2 = calc_me_diff(no_persona_2)

    significance_df = pd.read_csv("data/personas_significance_both.csv")

    # Create combined plot data directly
    def create_plot_data(prompt):
        # Load personas metadata
        all_data = pd.read_csv(PERSONAS_PATH, index_col="id")
        no_persona_me_diffs = no_persona_me_diffs_1 if prompt == "1" else no_persona_me_diffs_2
        
        # Merge data for each LLM
        for llm in target_llms:
            personas_me_diff = calc_me_diff_personas(llm)
            # Filter personas_me_diff to only include personas from PERSONAS_PATH
            personas_me_diff_filtered = personas_me_diff[personas_me_diff.index.isin(all_data.index)]
            all_data = all_data.merge(
                personas_me_diff_filtered[[f"prompt_{prompt}"]].rename(columns={f"prompt_{prompt}": llm}), 
                left_index=True, right_index=True, how="inner"
            )
        
        # Create plot data
        plot_data = []
        for llm in target_llms:
            # Add personas data only
            for _, row in all_data.iterrows():
                # Shorten political view labels and ensure correct order
                political_view_short = {
                    'Republican': 'R',
                    'Democrat': 'D', 
                    'No Specific Political View': 'N'
                }.get(row['political_view'], row['political_view'])
                
                plot_data.append({
                    'llm': llm,
                    'political_view': political_view_short,
                    'prompt_value': row[llm],
                    'prompt': f'Prompt {prompt}'
                })
        
        return pd.DataFrame(plot_data), no_persona_me_diffs

    # Create data for both prompts
    df_1, no_persona_1 = create_plot_data("1")
    df_2, no_persona_2 = create_plot_data("2")
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

    # Add horizontal lines for no_persona values for Prompt 1
    for i, llm in enumerate(target_llms):
        no_persona_value = no_persona_1.loc[llm]
        color = llm_colors[llm]["edgecolor"]
        ax1.axhline(y=no_persona_value, xmin=i*3/len(target_llms)/3, xmax=(i+1)*3/len(target_llms)/3, 
                color=color, linestyle='--', linewidth=2, alpha=0.8)

    # Add vertical lines between LLM groups for Prompt 1
    for i in range(1, len(target_llms)):
        ax1.axvline(x=i*3-0.5, color='lightgray', linestyle='-', linewidth=1, alpha=0.7)

    # ax1.set_title('Prompt 1', fontsize=14, fontweight='bold')
    ax1.set_ylabel('PB-T party-agnostic', fontsize=12)
    ax2.set_xlabel('')
    ax1.legend().remove()
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_ylim(None, 0.5)  # Cut off bottom at 0.1
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))

    # Remove x-axis labels and title for Prompt 1
    ax1.set_xticklabels([])
    ax1.set_xlabel('')

    # Plot for Prompt 2 - group by LLM first, then political view
    df_prompt2 = df_combined[df_combined['prompt'] == 'Prompt 2']
    df_prompt2['x_label'] = df_prompt2['llm'] + '_' + df_prompt2['political_view']
    sns.boxplot(data=df_prompt2, x='x_label', y='prompt_value', hue='llm', palette=palette_edge, ax=ax2,
                order=[f'{llm}_{view}' for llm in target_llms for view in ['D', 'N', 'R']])

    # Add horizontal lines for no_persona values for Prompt 2
    for i, llm in enumerate(target_llms):
        no_persona_value = no_persona_2.loc[llm]
        color = llm_colors[llm]["edgecolor"]
        ax2.axhline(y=no_persona_value, xmin=i*3/len(target_llms)/3, xmax=(i+1)*3/len(target_llms)/3, 
                color=color, linestyle='--', linewidth=2, alpha=0.8)

    # Add vertical lines between LLM groups for Prompt 2
    for i in range(1, len(target_llms)):
        ax2.axvline(x=i*3-0.5, color='lightgray', linestyle='-', linewidth=1, alpha=0.7)

    # ax2.set_title('Prompt 2', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Political Views', fontsize=12)
    ax2.set_ylabel('PB-T party-aware', fontsize=12)
    ax2.legend().remove()
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_ylim(None, 0.5)  # Cut off bottom at 0.7
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))

    # Set x-axis labels for Prompt 2 (only D, N, R)
    ax2.set_xticklabels(['D', 'N', 'R'] * len(target_llms))

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

    # Add significance stars for Prompt 1
    for i, llm in enumerate(target_llms):
        # Check significance for political_view for this LLM and prompt 1 (handle both string and int)
        sig_data = significance_df[
            (significance_df['llm'] == llm) & 
            ((significance_df['prompt'] == 1) | (significance_df['prompt'] == '1')) & 
            (significance_df['axis'] == 'both')
        ]
        
        if not sig_data.empty:
            p_value = sig_data.iloc[0]['political_view']
            if pd.notna(p_value):
                if p_value < 0.005:
                    # Add double star above the middle box (N = No Specific Political View)
                    ax1.text(i * 3 + 1, ax1.get_ylim()[1] * 0.40, '**', 
                            ha='center', va='center', fontsize=16, fontweight='bold', color='black')
                elif p_value < 0.025:
                    # Add single star above the middle box (N = No Specific Political View)
                    ax1.text(i * 3 + 1, ax1.get_ylim()[1] * 0.40, '*', 
                            ha='center', va='center', fontsize=16, fontweight='bold', color='black')

    # Add significance stars for Prompt 2
    for i, llm in enumerate(target_llms):
        # Check significance for political_view for this LLM and prompt 2 (handle both string and int)
        sig_data = significance_df[
            (significance_df['llm'] == llm) & 
            ((significance_df['prompt'] == 2) | (significance_df['prompt'] == '2')) & 
            (significance_df['axis'] == 'both')
        ]
        
        if not sig_data.empty:
            p_value = sig_data.iloc[0]['political_view']
            if pd.notna(p_value):
                if p_value < 0.005:
                    # Add double star above the middle box (N = No Specific Political View)
                    ax2.text(i * 3 + 1, ax2.get_ylim()[1] * 0.50, '**', 
                            ha='center', va='center', fontsize=16, fontweight='bold', color='black')
                elif p_value < 0.025:
                    # Add single star above the middle box (N = No Specific Political View)
                    ax2.text(i * 3 + 1, ax2.get_ylim()[1] * 0.50, '*', 
                            ha='center', va='center', fontsize=16, fontweight='bold', color='black')

    plt.subplots_adjust(hspace=0.1)  # Reduce space between subplots
    plt.tight_layout()
    plt.savefig(out_dir / 'box_plot.png', format='png', dpi=300)
    plt.close()


if __name__ == "__main__":
    build_box_plots()