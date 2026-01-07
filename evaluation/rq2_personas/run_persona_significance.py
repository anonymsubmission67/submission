"""
Statistical significance testing for persona experiments using ANOVA.
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

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from utils import target_llms, llm_info, PROMPT_TYPE, PERSONAS_PATH, make_tertile

# Constants
USE_BACKGROUND_COLORS = True  # Feature toggle for background colors in small table
PROMPT_TYPES = ["simple", "chain_of_thought", "all"]
PROMPTS = ["1", "2"]
AXES = ["economic", "social", "both"]
FACTORS = ['sex', 'ethnicity', 'education', 'political_view', 'age_bin', 'income_bin']
LABEL_TO_NUM = {
    "pants-fire": 0, "false": 1, "mostly-false": 2,
    "half-true": 3, "mostly-true": 4, "true": 5,
}
SIGNIFICANCE_THRESHOLD_STRONG = 0.005  # p < 0.005: **
SIGNIFICANCE_THRESHOLD_WEAK = 0.025    # p < 0.025: *


def _get_latex_color(llm: str) -> str:
    """
    Convert LLM edge color to LaTeX color with low opacity.
    
    Args:
        llm: LLM identifier
        
    Returns:
        LaTeX cellcolor command or empty string if colors disabled
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


def _format_p_value(p_val: float) -> str:
    """
    Format p-value for LaTeX table, bold if significant.
    
    Args:
        p_val: P-value to format
        
    Returns:
        Formatted p-value string
    """
    if pd.isna(p_val):
        return "na"
    if p_val < 0.05:
        return f"\\textbf{{{p_val:.3f}}}"
    return f"{p_val:.3f}"


def _format_significance_symbol(p_val: float) -> str:
    """
    Format significance symbol based on p-value.
    
    Args:
        p_val: P-value
        
    Returns:
        Significance symbol: **, *, or ---
    """
    if pd.isna(p_val):
        return "---"
    if p_val < SIGNIFICANCE_THRESHOLD_STRONG:
        return "$\\ast\\ast$"
    elif p_val < SIGNIFICANCE_THRESHOLD_WEAK:
        return "$\\ast$"
    return "---"


def _prepare_personas_data(personas: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare personas data by cleaning income and creating bins.
    
    Args:
        personas: Raw personas DataFrame
        
    Returns:
        Prepared personas DataFrame with age_bin and income_bin
    """
    personas = personas.copy()
    
    # Clean income then bin
    personas["income_num"] = (
        personas["income"].astype(str).str.replace(r"[$,]", "", regex=True).astype(float)
    )
    
    personas["age_bin"] = make_tertile(personas["age"])
    personas["income_bin"] = make_tertile(personas["income_num"])
    
    return personas


def _calculate_me_diff_for_personas(df: pd.DataFrame, claims: pd.DataFrame, 
                                     axis: str, personas: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ME differences for personas and merge with persona metadata.
    
    Args:
        df: DataFrame with persona responses
        claims: Claims metadata DataFrame
        axis: Axis to filter ('economic', 'social', or 'both')
        personas: Personas metadata DataFrame
        
    Returns:
        DataFrame with me_diff and persona attributes
    """
    # Filter by axis if not "both"
    df_long = df.T.merge(
        claims[["party", "label", "axis"]],
        left_index=True,
        right_index=True,
        how="left"
    )
    
    if axis != "both":
        df_long = df_long[df_long["axis"] == axis]
    else:
        df_long = df_long[df_long["axis"].isin(["economic", "social"])]
    
    df_long["label"] = df_long["label"].map(LABEL_TO_NUM)
    
    # Calculate Mean Error (subtract ground truth label)
    for col in df_long.columns:
        if col not in ["party", "label", "axis"]:
            df_long[col] = df_long[col] - df_long["label"]
    
    df_long = df_long.drop(columns=["label", "axis"])
    
    # Calculate me_diff for each persona
    bias_by_party = df_long.groupby(["party"]).mean().T
    me_diff = bias_by_party["Republican"] - bias_by_party["Democrat"]
    
    # Filter to only include personas from PERSONAS_PATH
    me_diff_filtered = me_diff[me_diff.index.isin(personas.index)]
    
    # Merge with persona metadata
    df_me_diff = me_diff_filtered.to_frame(name="me_diff").merge(
        personas[FACTORS],
        left_index=True,
        right_index=True,
        how="inner"
    )
    
    return df_me_diff


def _perform_anova(df_me_diff: pd.DataFrame) -> dict:
    """
    Perform multi-factor ANOVA on ME differences.
    
    Args:
        df_me_diff: DataFrame with me_diff and persona attributes
        
    Returns:
        Dictionary with p-values for each factor
    """
    # Multi-factor ANOVA
    formula = "me_diff ~ " + " + ".join([f"C({c})" for c in FACTORS])
    model = smf.ols(formula, data=df_me_diff).fit()
    anova = anova_lm(model, typ=2)
    
    # Extract p-values for each factor
    result = {}
    for factor in FACTORS:
        factor_key = f"C({factor})"
        if factor_key in anova.index:
            result[factor] = anova.loc[factor_key, "PR(>F)"]
        else:
            result[factor] = None
    
    return result



def run_persona_significance():
    """
    Main function to run ANOVA significance tests for persona experiments.
    
    Processes data for each prompt type, LLM, prompt, and axis combination,
    performs multi-factor ANOVA tests, and generates LaTeX tables.
    """
    out_dir = Path("output/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    claims = pd.read_csv("data/claims_metadata.csv").set_index("claim_id")
    personas_raw = pd.read_csv(PERSONAS_PATH).set_index("id")
    personas = _prepare_personas_data(personas_raw)

    # Process all combinations
    all_results = []
    
    for prompt_type in PROMPT_TYPES:
        for llm in target_llms:
            for prompt in PROMPTS:
                for axis in AXES:
                    print(f"Processing: {prompt_type}, {llm}, Prompt {prompt}, Axis {axis}")
                    
                    try:
                        # Load data
                        df_path = f"data/claim_matrices/{prompt_type}/personas_{prompt}/{llm}_mean.csv"
                        df = pd.read_csv(df_path, index_col=0)
                        
                        # Calculate ME differences and merge with persona metadata
                        df_me_diff = _calculate_me_diff_for_personas(df, claims, axis, personas)
                        
                        # Perform ANOVA
                        anova_results = _perform_anova(df_me_diff)
                        
                        # Create result row
                        result_row = {
                            'prompt_type': prompt_type,
                            'llm': llm,
                            'prompt': prompt,
                            'axis': axis
                        }
                        result_row.update(anova_results)
                        all_results.append(result_row)
                    
                    except Exception as e:
                        print(f"Error with {prompt_type}, {llm}, Prompt {prompt}, Axis {axis}: {e}")
                        # Add row with None values
                        result_row = {
                            'prompt_type': prompt_type,
                            'llm': llm,
                            'prompt': prompt,
                            'axis': axis
                        }
                        for factor in FACTORS:
                            result_row[factor] = None
                        all_results.append(result_row)

    # Create comprehensive DataFrame
    comprehensive_df = pd.DataFrame(all_results)

    # Create table with only "both" axis and "all" prompt_type for small table
    both_df = comprehensive_df[
        (comprehensive_df['axis'] == 'both') & 
        (comprehensive_df['prompt_type'] == 'all')
    ].copy()
    both_df.to_csv("data/personas_significance_both.csv", index=False)
    
    # Generate LaTeX tables
    _build_latex_big_table(comprehensive_df, out_dir)
    _build_latex_small_table(both_df, out_dir)
    
    print(f"\nComprehensive results shape: {comprehensive_df.shape}")
    print(f"Both axis results shape: {both_df.shape}")


def _build_latex_big_table(comprehensive_df: pd.DataFrame, out_dir: Path):
    """
    Build large LaTeX table with all ANOVA results.
    
    Args:
        comprehensive_df: DataFrame with all ANOVA results
        out_dir: Output directory for LaTeX file
    """

    # Create comprehensive LaTeX table - all dimensions and prompt types, with Prompt 1 and Prompt 2 in separate columns
    latex_code_big = f"""
\\begin{{center}}

\\setlength{{\LTleft}}{{3pt}}
\\setlength{{\LTright}}{{0pt}}
\\setlength{{\\tabcolsep}}{{4pt}}

\\begin{{longtable}}{{@{{}}lll|{'c' * len(FACTORS)}|{'c' * len(FACTORS)}@{{}}}}

\\caption{{ANOVA Results: P-values by LLM, Dimension, and Prompt Type}}
\\label{{tab:personas_big}}\\\\

\\toprule
\\multirow{{2}}{{*}}{{\\textbf{{LLM}}}} & \\multirow{{2}}{{*}}{{\\textbf{{Dimension}}}} & \\multirow{{2}}{{*}}{{\\textbf{{Prompt Type}}}} & \\multicolumn{{{len(FACTORS)}}}{{c|}}{{\\textbf{{party-agnostic}}}} & \\multicolumn{{{len(FACTORS)}}}{{c}}{{\\textbf{{party-aware}}}} \\\\
\\cmidrule(lr){{4-{3 + len(FACTORS)}}} \\cmidrule(lr){{{4 + len(FACTORS)}-{3 + 2 * len(FACTORS)}}}
& & & \\textbf{{Sex}} & \\textbf{{Ethnicity}} & \\textbf{{Education}} & \\textbf{{Pol View}} & \\textbf{{Age}} & \\textbf{{Income}} & \\textbf{{Sex}} & \\textbf{{Ethnicity}} & \\textbf{{Education}} & \\textbf{{Pol View}} & \\textbf{{Age}} & \\textbf{{Income}} \\\\
\\midrule
\\endfirsthead

\\toprule
\\multirow{{2}}{{*}}{{\\textbf{{LLM}}}} & \\multirow{{2}}{{*}}{{\\textbf{{Dimension}}}} & \\multirow{{2}}{{*}}{{\\textbf{{Prompt Type}}}} & \\multicolumn{{{len(FACTORS)}}}{{c|}}{{\\textbf{{party-agnostic}}}} & \\multicolumn{{{len(FACTORS)}}}{{c}}{{\\textbf{{party-aware}}}} \\\\
\\cmidrule(lr){{4-{3 + len(FACTORS)}}} \\cmidrule(lr){{{4 + len(FACTORS)}-{3 + 2 * len(FACTORS)}}}
& & & \\textbf{{Sex}} & \\textbf{{Ethnicity}} & \\textbf{{Education}} & \\textbf{{Pol View}} & \\textbf{{Age}} & \\textbf{{Income}} & \\textbf{{Sex}} & \\textbf{{Ethnicity}} & \\textbf{{Education}} & \\textbf{{Pol View}} & \\textbf{{Age}} & \\textbf{{Income}} \\\\
\\midrule
\\endhead

\\midrule
\\multicolumn{{{3 + 2 * len(FACTORS)}}}{{r}}{{\\small Continued on next page}}\\\\
\\endfoot

\\bottomrule
\\endlastfoot
"""

    # Sort LLMs according to target_llms order
    available_llms = list(comprehensive_df['llm'].unique())
    unique_llms = [llm for llm in target_llms if llm in available_llms]
    remaining_llms = [llm for llm in available_llms if llm not in target_llms]
    unique_llms.extend(sorted(remaining_llms))
    
    # Mapping dictionaries
    axis_display_mapping = {
        'economic': 'economic',
        'social': 'social',
        'both': 'combined'
    }
    
    # Define prompt type display mapping
    prompt_type_display_mapping = {
        'simple': 'direct',
        'chain_of_thought': 'chain-of-thought',
        'all': 'combined'
    }
    
    for llm in unique_llms:
        llm_data = comprehensive_df[comprehensive_df['llm'] == llm]
        
        for axis in AXES:
            axis_data = llm_data[llm_data['axis'] == axis]
            
            for prompt_type in PROMPT_TYPES:
                prompt_type_data = axis_data[axis_data['prompt_type'] == prompt_type]
                
                # Get prompt 1 and prompt 2 data
                prompt1_data = prompt_type_data[prompt_type_data['prompt'] == '1']
                prompt2_data = prompt_type_data[prompt_type_data['prompt'] == '2']
                
                # Format p-values for both prompts
                p1_values = _extract_p_values(prompt1_data)
                p2_values = _extract_p_values(prompt2_data)
                
                # Show LLM name in each row
                llm_name = llm_info.get(llm, {}).get("name", llm.capitalize())
                axis_display = axis_display_mapping.get(axis, axis)
                prompt_type_display = prompt_type_display_mapping.get(prompt_type, prompt_type)
                
                # Color row gray if dimension is combined and prompt type is combined
                row_color = "\\rowcolor{gray!20}" if (axis == 'both' and prompt_type == 'all') else ""
                
                latex_code_big += (
                    f"        {row_color}{llm_name} & {axis_display} & {prompt_type_display} & "
                    f"{' & '.join(p1_values)} & {' & '.join(p2_values)} \\\\\n"
                )

    latex_code_big += """
\\end{longtable}
\\end{center}
"""

    # Save table
    output_path = out_dir / "personas_big.tex"
    with open(output_path, 'w') as f:
        f.write(latex_code_big)
    print(f"LaTeX table saved to {output_path}")


    # Create LaTeX table header
    latex_code_small = f"""
    \\begin{{table}}[h]
        \\centering
        \\caption{{ANOVA Results: P-values by LLM (Both Axes Combined)}}
        \\label{{tab:personas_small}}
        {{\\small
        \\begin{{tabular}}{{@{{}}l@{{\\hspace{{0.5em}}}}c@{{\\hspace{{0.5em}}}}c@{{\\hspace{{0.5em}}}}c@{{\\hspace{{0.5em}}}}c@{{\\hspace{{0.5em}}}}c@{{\\hspace{{0.5em}}}}c@{{\\hspace{{0.5em}}}}c@{{\\hspace{{0.5em}}}}c@{{\\hspace{{0.5em}}}}c@{{\\hspace{{0.5em}}}}c@{{\\hspace{{0.5em}}}}c@{{\\hspace{{0.5em}}}}c@{{\\hspace{{0.5em}}}}c@{{}}}}
            \\toprule
            \\multirow{{2}}{{*}}{{\\textbf{{LLM}}}} & \\multicolumn{{6}}{{c|}}{{\\textbf{{party-agnostic}}}} & \\multicolumn{{6}}{{c}}{{\\textbf{{party-aware}}}} \\\\
            \\cmidrule(lr){{2-7}} \\cmidrule(lr){{8-13}}
            & \\attricon{{\\faVenusMars}} & \\attricon{{\\faGlobe}} & \\attricon{{\\faGraduationCap}} & \\attricon{{\\faDemocrat}}\\attricon{{\\faRepublican}} & \\attricon{{\\faBirthdayCake}} & \\attricon{{\\faMoneyBillWave}} & \\attricon{{\\faVenusMars}} & \\attricon{{\\faGlobe}} & \\attricon{{\\faGraduationCap}} & \\attricon{{\\faDemocrat}}\\attricon{{\\faRepublican}} & \\attricon{{\\faBirthdayCake}} & \\attricon{{\\faMoneyBillWave}} \\\\
            \\midrule
    """

    # Sort LLMs according to target_llms order
    available_llms = list(both_df['llm'].unique())
    unique_llms = [llm for llm in target_llms if llm in available_llms]
    remaining_llms = [llm for llm in available_llms if llm not in target_llms]
    unique_llms.extend(sorted(remaining_llms))

    # Generate table rows
    for llm in unique_llms:
        llm_data = both_df[both_df['llm'] == llm]
        
        # Get prompt 1 and prompt 2 data (handle both string and int)
        prompt1_data = llm_data[(llm_data['prompt'] == '1') | (llm_data['prompt'] == 1)]
        prompt2_data = llm_data[(llm_data['prompt'] == '2') | (llm_data['prompt'] == 2)]
        
        # Extract significance symbols for both prompts
        p1_symbols = _extract_significance_symbols(prompt1_data)
        p2_symbols = _extract_significance_symbols(prompt2_data)
        
        # Format row
        llm_name = llm_info.get(llm, {}).get("name", llm.capitalize())
        background_color = _get_latex_color(llm)
        all_symbols = p1_symbols + p2_symbols
        symbols_str = " & ".join([f"{background_color}{sym}" for sym in all_symbols])
        latex_code_small += f"        {background_color}{llm_name} & {symbols_str} \\\\\n"

    latex_code_small += """
            \\bottomrule
        \\end{tabular}
        }
    \\end{table}
    """

    # Save table
    output_path = out_dir / "personas_small.tex"
    with open(output_path, 'w') as f:
        f.write(latex_code_small)
    print(f"LaTeX table saved to {output_path}")


def _extract_significance_symbols(prompt_data: pd.DataFrame) -> list:
    """
    Extract and format significance symbols from prompt data.
    
    Args:
        prompt_data: DataFrame with ANOVA results for one prompt
        
    Returns:
        List of significance symbols (**, *, or ---)
    """
    if prompt_data.empty:
        return ["---"] * len(FACTORS)
    
    row = prompt_data.iloc[0]
    return [_format_significance_symbol(row[factor]) for factor in FACTORS]

if __name__ == "__main__":
    run_persona_significance()