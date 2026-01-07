"""
Build a comprehensive LaTeX longtable with significance test results.
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

from utils import target_llms, opinion_llms, llm_info

# Constants
PROMPT_TYPES = ["simple", "chain_of_thought", "all"]
AXES = ["economic", "social", "both"]

def _safe_get_value(df: pd.DataFrame, col_name: str, default=None):
    """
    extract value from DataFrame, handling NaN, None, and empty strings.
    """
    if df.empty or col_name not in df.columns:
        return default
    try:
        val = df.iloc[0][col_name]
        # Check for NaN, None, or empty string
        if pd.isna(val) or val == '' or val is None:
            return default
        # Convert empty string to default
        if isinstance(val, str) and val.strip() == '':
            return default
        return val
    except (KeyError, IndexError):
        return default


def _format_p_value(val):
    """Format p-value, bold if < 0.05."""
    if pd.isna(val) or val is None:
        return "N/A"
    if val < 0.05:
        return f"\\textbf{{{val:.4f}}}"
    return f"{val:.4f}"


def _format_mean(val):
    """Format mean value."""
    if pd.isna(val) or val is None:
        return "N/A"
    return f"{val:.4f}"


def _format_test_type(test_type):
    """Format test type - t-test (t) or Mann-Whitney U (U)."""
    if pd.isna(test_type) or test_type is None:
        return "N/A"
    if test_type == 't':
        return "t"
    elif test_type == 'U':
        return "U"
    return str(test_type)


def _format_test_p(test_p_value):
    """Format test p-value from t-test or Mann-Whitney U, bold if < 0.05."""
    if pd.isna(test_p_value) or test_p_value is None:
        return "N/A"
    if test_p_value < 0.05:
        return f"\\textbf{{{test_p_value:.4f}}}"
    return f"{test_p_value:.4f}"


def build_no_persona_big():
    """
    Create a LaTeX longtable
    """
    out_dir = Path("output/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load shapiro_wilk_summary for each prompt type
    all_data = {}
    for prompt_type in PROMPT_TYPES:
        try:
            df = pd.read_csv(
                f"data/interim_results/shapiro_wilk_summary_{prompt_type}.csv",
                na_values=['', 'nan', 'NaN', 'None', 'null'],
                keep_default_na=True
            )
            if 'Prompt_Type' not in df.columns:
                df['Prompt_Type'] = prompt_type
            all_data[prompt_type] = df
        except FileNotFoundError:
            print(f"Warning: Could not find summary file for {prompt_type}")
            continue
 
    
    # Combine all data
    if not all_data:
        print("Error: No data loaded!")
        return
    
    combined_df = pd.concat(all_data.values(), ignore_index=True)
    
    if combined_df.empty:
        print("Warning: Combined dataframe is empty. Creating empty table.")
        combined_df = pd.DataFrame(columns=[
            'LLM', 'Prompt', 'Axis', 'Metric', 
            'Republican_mean', 'Republican_p_value', 'Democrat_mean', 'Democrat_p_value', 'Both_Normal',
            'Test_p_value', 'Test_type', 'Prompt_Type'
        ])
    
    # Mapping dictionaries
    axis_mapping = {
        'economic': 'economic',
        'social': 'social',
        'both': 'combined'
    }
    
    prompt_type_mapping = {
        'simple': 'direct',
        'chain_of_thought': 'chain of thought',
        'all': 'combined'
    }
    
    # Get unique LLMs, sorted according to target_llms order
    available_llms = list(combined_df['LLM'].unique()) if 'LLM' in combined_df.columns else []
    unique_llms = [llm for llm in target_llms if llm in available_llms]
    remaining_llms = [llm for llm in available_llms if llm not in target_llms]
    unique_llms.extend(sorted(remaining_llms))
    
    # Create LaTeX longtable header
    latex_code = f"""
\\begin{{center}}

\\setlength{{\\LTleft}}{{3pt}}
\\setlength{{\\LTright}}{{0pt}}
\\setlength{{\\tabcolsep}}{{4pt}}

\\begin{{longtable}}{{@{{}}lll|cccccc|cccccc@{{}}}}

\\caption{{Shapiro-Wilk Normality Test Results}}
\\label{{tab:no_persona_normality_big}}\\\\

\\toprule
\\multirow{{3}}{{*}}{{\\textbf{{Model}}}} &
\\multirow{{3}}{{*}}{{\\textbf{{Dimension}}}} &
\\multirow{{3}}{{*}}{{\\textbf{{Prompt Type}}}} &
\\multicolumn{{6}}{{c}}{{\\textbf{{party-agnostic}}}} &
\\multicolumn{{6}}{{c}}{{\\textbf{{party-aware}}}} \\\\
\\cmidrule(lr){{4-9}} \\cmidrule(lr){{10-15}}
& & &
\\multicolumn{{2}}{{c}}{{\\textbf{{Republican}}}} &
\\multicolumn{{2}}{{c}}{{\\textbf{{Democrat}}}} &
\\multicolumn{{2}}{{c}}{{\\textbf{{Test}}}} &
\\multicolumn{{2}}{{c}}{{\\textbf{{Republican}}}} &
\\multicolumn{{2}}{{c}}{{\\textbf{{Democrat}}}} &
\\multicolumn{{2}}{{c}}{{\\textbf{{Test}}}} \\\\
\\cmidrule(lr){{4-5}} \\cmidrule(lr){{6-7}} \\cmidrule(lr){{8-9}} \\cmidrule(lr){{10-11}} \\cmidrule(lr){{12-13}} \\cmidrule(lr){{14-15}}
& & &
\\textbf{{mean}} & \\textbf{{p}} &
\\textbf{{mean}} & \\textbf{{p}} &
\\textbf{{type}} & \\textbf{{p}} &
\\textbf{{mean}} & \\textbf{{p}} &
\\textbf{{mean}} & \\textbf{{p}} &
\\textbf{{type}} & \\textbf{{p}} \\\\
\\midrule
\\endfirsthead

\\toprule
\\multirow{{3}}{{*}}{{\\textbf{{Model}}}} &
\\multirow{{3}}{{*}}{{\\textbf{{Dimension}}}} &
\\multirow{{3}}{{*}}{{\\textbf{{Prompt Type}}}} &
\\multicolumn{{6}}{{c}}{{\\textbf{{party-agnostic}}}} &
\\multicolumn{{6}}{{c}}{{\\textbf{{party-aware}}}} \\\\
\\cmidrule(lr){{4-9}} \\cmidrule(lr){{10-15}}
& & &
\\multicolumn{{2}}{{c}}{{\\textbf{{Republican}}}} &
\\multicolumn{{2}}{{c}}{{\\textbf{{Democrat}}}} &
\\multicolumn{{2}}{{c}}{{\\textbf{{Test}}}} &
\\multicolumn{{2}}{{c}}{{\\textbf{{Republican}}}} &
\\multicolumn{{2}}{{c}}{{\\textbf{{Democrat}}}} &
\\multicolumn{{2}}{{c}}{{\\textbf{{Test}}}} \\\\
\\cmidrule(lr){{4-5}} \\cmidrule(lr){{6-7}} \\cmidrule(lr){{8-9}} \\cmidrule(lr){{10-11}} \\cmidrule(lr){{12-13}} \\cmidrule(lr){{14-15}}
& & &
\\textbf{{mean}} & \\textbf{{p}} &
\\textbf{{mean}} & \\textbf{{p}} &
\\textbf{{type}} & \\textbf{{p}} &
\\textbf{{mean}} & \\textbf{{p}} &
\\textbf{{mean}} & \\textbf{{p}} &
\\textbf{{type}} & \\textbf{{p}} \\\\
\\midrule
\\endhead

\\midrule
\\multicolumn{{15}}{{r}}{{\\small Continued on next page}}\\\\
\\endfoot

\\bottomrule
\\endlastfoot
"""
    
    # Process each LLM, axis, and prompt type combination
    for llm in unique_llms:
        for axis in AXES:
            axis_display = axis_mapping.get(axis, axis)
            
            # Process each prompt type in separate rows
            for prompt_type in PROMPT_TYPES:
                # Filter data for this combination
                llm_data = combined_df[
                    (combined_df['LLM'] == llm) & 
                    (combined_df['Axis'] == axis) &
                    (combined_df['Prompt_Type'] == prompt_type)
                ]
                
                if llm_data.empty:
                    continue
                
                llm_name = llm_info.get(llm, {}).get("name", llm.capitalize())
                
                # Process Prompt 1 data - only ME metric
                prompt1_data = llm_data[llm_data['Prompt'].str.contains('1', na=False)]
                me_p1 = prompt1_data[prompt1_data['Metric'] == 'ME']
                
                # Extract Prompt 1 values
                p1_rep_mean = _safe_get_value(me_p1, 'Republican_mean')
                p1_rep_p = _safe_get_value(me_p1, 'Republican_p_value')
                p1_dem_mean = _safe_get_value(me_p1, 'Democrat_mean')
                p1_dem_p = _safe_get_value(me_p1, 'Democrat_p_value')
                p1_test_p = _safe_get_value(me_p1, 'Test_p_value')
                p1_test_type = _safe_get_value(me_p1, 'Test_type')
                
                # Process Prompt 2 data - only ME metric
                prompt2_data = llm_data[llm_data['Prompt'].str.contains('2', na=False)]
                me_p2 = prompt2_data[prompt2_data['Metric'] == 'ME']
                
                # Extract Prompt 2 values
                p2_rep_mean = _safe_get_value(me_p2, 'Republican_mean')
                p2_rep_p = _safe_get_value(me_p2, 'Republican_p_value')
                p2_dem_mean = _safe_get_value(me_p2, 'Democrat_mean')
                p2_dem_p = _safe_get_value(me_p2, 'Democrat_p_value')
                p2_test_p = _safe_get_value(me_p2, 'Test_p_value')
                p2_test_type = _safe_get_value(me_p2, 'Test_type')
                
                # Determine if row should be colored (only rows with both combined AND all)
                row_color = ""
                if axis_display == 'combined' and prompt_type == 'all':
                    row_color = "\\rowcolor{gray!20} "
                
                # Map prompt type for display
                prompt_type_display = prompt_type_mapping.get(prompt_type, prompt_type)
                
                # Format and add row to LaTeX table
                latex_code += (
                    f"        {row_color}{llm_name} & {axis_display} & {prompt_type_display} & "
                    f"{_format_mean(p1_rep_mean)} & {_format_p_value(p1_rep_p)} & "
                    f"{_format_mean(p1_dem_mean)} & {_format_p_value(p1_dem_p)} & "
                    f"{_format_test_type(p1_test_type)} & {_format_test_p(p1_test_p)} & "
                    f"{_format_mean(p2_rep_mean)} & {_format_p_value(p2_rep_p)} & "
                    f"{_format_mean(p2_dem_mean)} & {_format_p_value(p2_dem_p)} & "
                    f"{_format_test_type(p2_test_type)} & {_format_test_p(p2_test_p)} \\\\\n"
                )
                row_color = ""
                
    latex_code += """

\\end{longtable}
\\end{center}
"""
    
    # Save LaTeX table
    output_path = out_dir / "no_persona_big.tex"
    with open(output_path, 'w') as f:
        f.write(latex_code)
    
    print(f"Large table saved to: {output_path}")


if __name__ == "__main__":
    build_no_persona_big()
