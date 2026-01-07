"""
Build a small summary table with significance test results.

"""

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

# Constants
USE_BACKGROUND_COLORS = True  # Feature toggle for background colors in small table
SIGNIFICANCE_THRESHOLD_STRONG = 0.005  # p < 0.005: **
SIGNIFICANCE_THRESHOLD_WEAK = 0.025    # p < 0.025: *


def _get_latex_color(llm: str) -> str:
    """
    Convert LLM edge color to LaTeX color with low opacity.
    """
    if not USE_BACKGROUND_COLORS:
        return ""
    
    # Map to LaTeX-compatible colors (using xcolor package colors)
    color_map = {
        'mistral': 'yellow',
        'mixtral': 'orange',
        'llama3': 'green',
        'llama4': 'green!50!black',
        'qwen': 'brown',
        'qwen_big': 'brown!50!black',
        'deepseek_small': 'purple',
        'deepseek': 'purple!50!black',
        'gpt-oss': 'cyan',
        'gpt4': 'teal',
        'grok_small': 'magenta',
        'grok': 'violet',
        'phi': 'gray',
        'conservative': 'red',
        'liberal': 'blue',
        'american': 'black'
    }
    
    latex_color = color_map.get(llm, 'gray')
    return f"\\cellcolor{{{latex_color}!10}}"  # 10% opacity


def _process_prompt_data(prompt_data: pd.DataFrame) -> tuple:
    """
    Process prompt data to extract test type, significance symbol, and truth bias.
    """
    if prompt_data.empty:
        return ("U", "$-$", "N/A")
    
    row = prompt_data.iloc[0]
    
    # Calculate Truth Bias (TB = ME Republican - ME Democrat)
    rep_mean = row['Republican_mean'] if pd.notna(row['Republican_mean']) else 0
    dem_mean = row['Democrat_mean'] if pd.notna(row['Democrat_mean']) else 0
    truth_bias = f"{rep_mean - dem_mean:.3f}"
    
    # Get test p-value and type
    test_p_val = row['Test_p_value'] if pd.notna(row.get('Test_p_value', None)) else None
    test_type = row['Test_type'] if pd.notna(row.get('Test_type', None)) else None
    
    # Determine test type
    if test_type == 't':
        test_type_str = "t"
    elif test_type == 'U':
        test_type_str = "U"
    else:
        test_type_str = "U"  # Default
    
    # Determine significance symbol
    significance_symbol = "$-$"
    if test_p_val is not None:
        if test_p_val < SIGNIFICANCE_THRESHOLD_STRONG:
            # Strong significance: **
            if dem_mean > rep_mean:
                significance_symbol = "\\textcolor{blue}{$\\ast\\ast$}"  # Democrat > Republican
            else:
                significance_symbol = "\\textcolor{red}{$\\ast\\ast$}"   # Republican > Democrat
        elif test_p_val < SIGNIFICANCE_THRESHOLD_WEAK:
            # Weak significance: *
            if dem_mean > rep_mean:
                significance_symbol = "\\textcolor{blue}{$\\ast$}"  # Democrat > Republican
            else:
                significance_symbol = "\\textcolor{red}{$\\ast$}"   # Republican > Democrat
    
    return (test_type_str, significance_symbol, truth_bias)


def build_no_persona_small():
    """
    Create a small summary table with significance test results.
    """
    out_dir = Path("output/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load test results from shapiro_wilk_summary
    summary_df = pd.read_csv(f"data/interim_results/shapiro_wilk_summary_{PROMPT_TYPE}.csv")
    
    # Filter for ME metric and "both" axis only
    both_df = summary_df[(summary_df['Metric'] == 'ME') & (summary_df['Axis'] == 'both')].copy()

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
    
    # Process each LLM
    for llm in unique_llms:
        llm_data = both_df[both_df['LLM'] == llm]
        
        # Get data for prompt 1 and prompt 2
        # Note: Prompt column contains "prompt_1" or "prompt_2"
        prompt1_data = llm_data[llm_data['Prompt'].str.contains('1', na=False)]
        prompt2_data = llm_data[llm_data['Prompt'].str.contains('2', na=False)]
        
        # Process both prompts using the helper function
        prompt1_test, prompt1_symbol, prompt1_tb = _process_prompt_data(prompt1_data)
        prompt2_test, prompt2_symbol, prompt2_tb = _process_prompt_data(prompt2_data)
        
        # Format row
        llm_name = llm_info.get(llm, {}).get("name", llm.capitalize())
        background_color = _get_latex_color(llm)
        latex_code_small += (
            f"        {background_color}{llm_name} & "
            f"{background_color}{prompt1_test} & {background_color}{prompt1_symbol} & {background_color}{prompt1_tb} & "
            f"{background_color}{prompt2_test} & {background_color}{prompt2_symbol} & {background_color}{prompt2_tb} \\\\\n"
        )

    latex_code_small += """
            \\bottomrule
        \\end{tabular}
        }
        \\caption{Adaptive Significance Test Results: Republican vs Democrat Responses (Both Axes Combined)}
        \\label{tab:no_persona_small}
    \\end{table}
    """

    # Save table
    output_path = out_dir / "no_persona_small.tex"
    with open(output_path, 'w') as f:
        f.write(latex_code_small)
    
    print(f"Small table saved to: {output_path}")
    print(f"Processed {len(unique_llms)} LLMs")

if __name__ == "__main__":
    build_no_persona_small()