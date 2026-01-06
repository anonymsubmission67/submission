import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add evaluation directory to path to allow importing utils
script_dir = Path(__file__).resolve().parent
evaluation_dir = script_dir.parent
if str(evaluation_dir) not in sys.path:
    sys.path.insert(0, str(evaluation_dir))

from utils import target_llms, opinion_llms, llm_info, PROMPT_TYPE

# Feature toggle for background colors in small table
USE_BACKGROUND_COLORS = True


def build_no_persona_small():
    """
    Create a small table with significance test results using data from shapiro_wilk_summaries.
    The table shows test type, significance symbols, and truth bias for both prompts.
    """
    
    out_dir = Path("output/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load test results from shapiro_wilk_summary
    summary_df = pd.read_csv(f"data/interim_results/shapiro_wilk_summary_{PROMPT_TYPE}.csv")
    
    # Filter for ME metric and "both" axis only
    both_df = summary_df[(summary_df['Metric'] == 'ME') & (summary_df['Axis'] == 'both')].copy()

    # Create small LaTeX table (only "both" axis with symbols)

    # Create color definitions for LaTeX
    def get_latex_color(llm):
        """Convert LLM edge color to LaTeX color with low opacity"""
        if not USE_BACKGROUND_COLORS:
            return ""
        
        # Map to LaTeX-compatible colors (using xcolor package colors)
        color_map = {
            'mistral': 'yellow',           # gold -> yellow
            'mixtral': 'orange',           # darkorange -> orange  
            'llama3': 'green',             # limegreen -> green
            'llama4': 'green!50!black',    # darkgreen -> darker green
            'qwen': 'brown',               # chocolate -> brown
            'qwen_big': 'brown!50!black',  # saddlebrown -> darker brown
            'deepseek_small': 'purple',    # mediumpurple -> purple
            'deepseek': 'purple!50!black', # rebeccapurple -> darker purple
            'gpt-oss': 'cyan',             # darkturquoise -> cyan
            'gpt4': 'teal',                # teal -> teal
            'grok_small':'magenta',        # violet -> violet
            'grok':  'violet',             # magenta -> magenta
            'phi': 'gray',                 # gray -> gray
            'conservative': 'red',         # red -> red
            'liberal': 'blue',             # blue -> blue
            'american': 'black'            # black -> black
        }
        
        latex_color = color_map.get(llm, 'gray')
        return f"\\cellcolor{{{latex_color}!10}}"  # 10% opacity

    latex_code_small = f"""
    \\begin{{table}}[h]
        \\centering
        \\caption{{Adaptive Significance Test Results: Republican vs Democrat Responses (Both Axes Combined)}}
        \\label{{tab:no_persona_adaptive_small}}
        {{\\small
        \\begin{{tabular}}{{@{{}}l|ccc|ccc@{{}}}}
            \\toprule
            \\multirow{{2}}{{*}}{{\\textbf{{}}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{party-agnostic}}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{party-aware}}}} \\\\
            \\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-7}}
            & \\textbf{{test}} & \\textbf{{sig}} & \\textbf{{PB-T}} & \\textbf{{test}} & \\textbf{{sig}} & \\textbf{{PB-T}} \\\\
            \\midrule
    """

    # Sort LLMs according to target_llms order
    available_llms = both_df['LLM'].unique()
    unique_llms = [llm for llm in target_llms if llm in available_llms]
    # Add any LLMs not in target_llms at the end
    remaining_llms = [llm for llm in available_llms if llm not in target_llms]
    unique_llms.extend(sorted(remaining_llms))
    
    # Group by LLM to create one row per LLM
    for llm in unique_llms:
        llm_data = both_df[both_df['LLM'] == llm]
        
        # Get data for prompt 1 and prompt 2
        # Note: Prompt column contains "prompt_1" or "prompt_2"
        prompt1_data = llm_data[llm_data['Prompt'].str.contains('1', na=False)]
        prompt2_data = llm_data[llm_data['Prompt'].str.contains('2', na=False)]
        
        # Process prompt 1
        prompt1_symbol = "$-$"
        prompt1_parametric = "U"
        prompt1_tb = "N/A"
        if not prompt1_data.empty:
            p1_row = prompt1_data.iloc[0]
            # Calculate Truth Bias (TB = ME Republican - ME Democrat)
            rep_mean = p1_row['Republican_mean'] if pd.notna(p1_row['Republican_mean']) else 0
            dem_mean = p1_row['Democrat_mean'] if pd.notna(p1_row['Democrat_mean']) else 0
            prompt1_tb = f"{rep_mean - dem_mean:.3f}"
            
            # Get test p-value and type from the summary
            test_p_val = p1_row['Test_p_value'] if pd.notna(p1_row.get('Test_p_value', None)) else None
            test_type = p1_row['Test_type'] if pd.notna(p1_row.get('Test_type', None)) else None
            
            if test_p_val is not None:
                # Determine significance symbols: ** for p < 0.005, * for p < 0.025
                if test_p_val < 0.005:
                    if dem_mean > rep_mean:
                        prompt1_symbol = "\\textcolor{blue}{$\\ast\\ast$}"  # Democrat > Republican
                    else:
                        prompt1_symbol = "\\textcolor{red}{$\\ast\\ast$}"   # Republican > Democrat
                elif test_p_val < 0.025:
                    if dem_mean > rep_mean:
                        prompt1_symbol = "\\textcolor{blue}{$\\ast$}"  # Democrat > Republican
                    else:
                        prompt1_symbol = "\\textcolor{red}{$\\ast$}"   # Republican > Democrat
            
            # Determine if parametric (t-test) or non-parametric (Mann-Whitney U)
            if test_type == 't':
                prompt1_parametric = "t"
            elif test_type == 'U':
                prompt1_parametric = "U"
            else:
                prompt1_parametric = "U"  # Default
        
        # Process prompt 2
        prompt2_symbol = "$-$"
        prompt2_parametric = "U"
        prompt2_tb = "N/A"
        if not prompt2_data.empty:
            p2_row = prompt2_data.iloc[0]
            # Calculate Truth Bias (TB = ME Republican - ME Democrat)
            rep_mean = p2_row['Republican_mean'] if pd.notna(p2_row['Republican_mean']) else 0
            dem_mean = p2_row['Democrat_mean'] if pd.notna(p2_row['Democrat_mean']) else 0
            prompt2_tb = f"{rep_mean - dem_mean:.3f}"
            
            # Get test p-value and type from the summary
            test_p_val = p2_row['Test_p_value'] if pd.notna(p2_row.get('Test_p_value', None)) else None
            test_type = p2_row['Test_type'] if pd.notna(p2_row.get('Test_type', None)) else None
            
            if test_p_val is not None:
                # Determine significance symbols: ** for p < 0.005, * for p < 0.025
                if test_p_val < 0.005:
                    if dem_mean > rep_mean:
                        prompt2_symbol = "\\textcolor{blue}{$\\ast\\ast$}"  # Democrat > Republican
                    else:
                        prompt2_symbol = "\\textcolor{red}{$\\ast\\ast$}"   # Republican > Democrat
                elif test_p_val < 0.025:
                    if dem_mean > rep_mean:
                        prompt2_symbol = "\\textcolor{blue}{$\\ast$}"  # Democrat > Republican
                    else:
                        prompt2_symbol = "\\textcolor{red}{$\\ast$}"   # Republican > Democrat
            
            # Determine if parametric (t-test) or non-parametric (Mann-Whitney U)
            if test_type == 't':
                prompt2_parametric = "t"
            elif test_type == 'U':
                prompt2_parametric = "U"
            else:
                prompt2_parametric = "U"  # Default
        
        llm_name = llm_info.get(llm, {}).get("name", llm.capitalize())
        background_color = get_latex_color(llm)
        latex_code_small += f"        {background_color}{llm_name} & {background_color}{prompt1_parametric} & {background_color}{prompt1_symbol} & {background_color}{prompt1_tb} & {background_color}{prompt2_parametric} & {background_color}{prompt2_symbol} & {background_color}{prompt2_tb} \\\\\n"

    latex_code_small += """
            \\bottomrule
        \\end{tabular}
        }
        \caption{Results of ANOVA significance tests on persona attributes are shown for prompts without source (left) and with source (right). All attributes except political view exhibit only a limited impact on MED. In contrast, the political viewpoint of a persona has a strong influence: 5 out of 9 models are significant without a source, and 8 out of 9 with a source. For further details, see \ref{fig:box_plot} and \ref{tab:personas_big}.}
        \label{tab:no_persona_small}
    \\end{table}
    """

    with open('output/tables/no_persona_small.tex', 'w') as f:
        f.write(latex_code_small)
    
    print(f"Small table saved to: output/tables/no_persona_small.tex")
    print(f"Processed {len(unique_llms)} LLMs")

if __name__ == "__main__":
    build_no_persona_small()