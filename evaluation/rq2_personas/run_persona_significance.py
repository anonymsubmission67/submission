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
from utils import target_llms, opinion_llms, llm_colors, llm_info, PROMPT_TYPE, PERSONAS_PATH

# Feature toggle for background colors in small table
USE_BACKGROUND_COLORS = True



def make_tertile(s):
    s = pd.to_numeric(s, errors="coerce")
    cats, bins = pd.qcut(s, q=3, labels=None, retbins=True, duplicates="drop")
    n = len(bins) - 1  # actual number of bins after dropping duplicates
    labels = ["low", "middle", "high"][:n]
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True)


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
        'grok_small':'magenta',        # violet -> magenta
        'grok':  'violet',             # magenta -> violet
        'phi': 'gray',                 # gray -> gray
        'conservative': 'red',         # red -> red
        'liberal': 'blue',             # blue -> blue
        'american': 'black'            # black -> black
    }
    
    latex_color = color_map.get(llm, 'gray')
    return f"\\cellcolor{{{latex_color}!10}}"  # 10% opacity



def run_persona_significance():

    out_dir = Path("output/tables")
    out_dir.mkdir(parents=True, exist_ok=True)


    claims = pd.read_csv("data/claims_metadata.csv").set_index("claim_id")
    personas = pd.read_csv(PERSONAS_PATH).set_index("id")

    # Clean income then bin
    personas["income_num"] = (
        personas["income"].astype(str).str.replace(r"[$,]", "", regex=True).astype(float)
    )

    personas["age_bin"] = make_tertile(personas["age"])
    personas["income_bin"] = make_tertile(personas["income_num"])


    # Map dataset labels to numeric 0..5
    label_to_num = {
        "pants-fire": 0, "false": 1, "mostly-false": 2,
        "half-true": 3, "mostly-true": 4, "true": 5,
    }

    # Create comprehensive results table
    all_results = []
    factors = ['sex', 'ethnicity', 'education', 'political_view', 'age_bin', 'income_bin']
    
    # Process all prompt types
    prompt_types = ["simple", "chain_of_thought", "all"]

    for prompt_type in prompt_types:
        for llm in target_llms:
            for prompt in ["1", "2"]:
                for axis in ["economic", "social", "both"]:
                    print(f"Processing: {prompt_type}, {llm}, Prompt {prompt}, Axis {axis}")
                    
                    try:
                        df_path = f"data/claim_matrices/{prompt_type}/personas_{prompt}/{llm}_mean.csv"
                        df = pd.read_csv(df_path, index_col=0)
                        
                        # Filter by axis if not "both"
                        if axis != "both":
                            df_long = df.T.merge(claims[["party", "label", "axis"]], left_index=True, right_index=True, how="left")
                            df_long = df_long[df_long["axis"] == axis]
                        else:
                            df_long = df.T.merge(claims[["party", "label", "axis"]], left_index=True, right_index=True, how="left")
                            df_long = df_long[df_long["axis"].isin(["economic", "social"])]
                        
                        df_long["label"] = df_long["label"].map(label_to_num)
                        
                        # Subtract dataset_label from all columns except "party", "label", "axis"
                        for col in df_long.columns:
                            if col not in ["party", "label", "axis"]:
                                df_long[col] = df_long[col] - df_long["label"]
                        
                        df_long = df_long.drop(columns=["label", "axis"])
                        
                        # Calculate me_diff for each persona
                        bias_by_party = df_long.groupby(["party"]).mean().T
                        me_diff = bias_by_party["Republican"] - bias_by_party["Democrat"]
                        
                        # Filter me_diff to only include personas from PERSONAS_PATH
                        me_diff_filtered = me_diff[me_diff.index.isin(personas.index)]
                        
                        # Merge with persona metadata (inner join to ensure only personas from PERSONAS_PATH)
                        df_me_diff = me_diff_filtered.to_frame(name="me_diff").merge(
                            personas[["sex", "ethnicity", "education", "political_view", "age_bin", "income_bin"]], 
                            left_index=True, 
                            right_index=True, 
                            how="inner"
                        )
                        
                        # Multi-factor ANOVA
                        formula = "me_diff ~ " + " + ".join([f"C({c})" for c in factors])
                        model = smf.ols(formula, data=df_me_diff).fit()
                        anova = anova_lm(model, typ=2)
                        
                        # Extract p-values for each factor
                        result_row = {
                            'prompt_type': prompt_type,
                            'llm': llm,
                            'prompt': prompt,
                            'axis': axis
                        }
                        
                        for factor in factors:
                            factor_key = f"C({factor})"
                            if factor_key in anova.index:
                                p_value = anova.loc[factor_key, "PR(>F)"]
                                result_row[factor] = p_value
                            else:
                                result_row[factor] = None
                        
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
                        for factor in factors:
                            result_row[factor] = None
                        all_results.append(result_row)

    # Create comprehensive DataFrame
    comprehensive_df = pd.DataFrame(all_results)

    # Create table with only "both" axis and "all" prompt_type for small table
    both_df = comprehensive_df[(comprehensive_df['axis'] == 'both') & (comprehensive_df['prompt_type'] == 'all')].copy()
    both_df.to_csv("data/personas_significance_both.csv", index=False)

    # Create comprehensive LaTeX table - all dimensions and prompt types, with Prompt 1 and Prompt 2 in separate columns
    latex_code_big = f"""
\\begin{{center}}

\\setlength{{\LTleft}}{{3pt}}
\\setlength{{\LTright}}{{0pt}}
\\setlength{{\\tabcolsep}}{{4pt}}

\\begin{{longtable}}{{@{{}}lll|{'c' * len(factors)}|{'c' * len(factors)}@{{}}}}

\\caption{{ANOVA Results: P-values by LLM, Dimension, and Prompt Type}}
\\label{{tab:personas_big}}\\\\

\\toprule
\\multirow{{2}}{{*}}{{\\textbf{{LLM}}}} & \\multirow{{2}}{{*}}{{\\textbf{{Dimension}}}} & \\multirow{{2}}{{*}}{{\\textbf{{Prompt Type}}}} & \\multicolumn{{{len(factors)}}}{{c|}}{{\\textbf{{party-agnostic}}}} & \\multicolumn{{{len(factors)}}}{{c}}{{\\textbf{{party-aware}}}} \\\\
\\cmidrule(lr){{4-{3 + len(factors)}}} \\cmidrule(lr){{{4 + len(factors)}-{3 + 2 * len(factors)}}}
& & & \\textbf{{Sex}} & \\textbf{{Ethnicity}} & \\textbf{{Education}} & \\textbf{{Pol View}} & \\textbf{{Age}} & \\textbf{{Income}} & \\textbf{{Sex}} & \\textbf{{Ethnicity}} & \\textbf{{Education}} & \\textbf{{Pol View}} & \\textbf{{Age}} & \\textbf{{Income}} \\\\
\\midrule
\\endfirsthead

\\toprule
\\multirow{{2}}{{*}}{{\\textbf{{LLM}}}} & \\multirow{{2}}{{*}}{{\\textbf{{Dimension}}}} & \\multirow{{2}}{{*}}{{\\textbf{{Prompt Type}}}} & \\multicolumn{{{len(factors)}}}{{c|}}{{\\textbf{{party-agnostic}}}} & \\multicolumn{{{len(factors)}}}{{c}}{{\\textbf{{party-aware}}}} \\\\
\\cmidrule(lr){{4-{3 + len(factors)}}} \\cmidrule(lr){{{4 + len(factors)}-{3 + 2 * len(factors)}}}
& & & \\textbf{{Sex}} & \\textbf{{Ethnicity}} & \\textbf{{Education}} & \\textbf{{Pol View}} & \\textbf{{Age}} & \\textbf{{Income}} & \\textbf{{Sex}} & \\textbf{{Ethnicity}} & \\textbf{{Education}} & \\textbf{{Pol View}} & \\textbf{{Age}} & \\textbf{{Income}} \\\\
\\midrule
\\endhead

\\midrule
\\multicolumn{{{3 + 2 * len(factors)}}}{{r}}{{\\small Continued on next page}}\\\\
\\endfoot

\\bottomrule
\\endlastfoot
"""

    # Sort LLMs according to target_llms order
    available_llms = list(comprehensive_df['llm'].unique())
    unique_llms = [llm for llm in target_llms if llm in available_llms]
    remaining_llms = [llm for llm in available_llms if llm not in target_llms]
    unique_llms.extend(sorted(remaining_llms))
    
    # Define dimension order and mapping for display
    dimensions = ['economic', 'social', 'both']
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
        
        for axis in dimensions:
            axis_data = llm_data[llm_data['axis'] == axis]
            
            for prompt_type in prompt_types:
                prompt_type_data = axis_data[axis_data['prompt_type'] == prompt_type]
                
                # Get prompt 1 and prompt 2 data
                prompt1_data = prompt_type_data[prompt_type_data['prompt'] == '1']
                prompt2_data = prompt_type_data[prompt_type_data['prompt'] == '2']
                
                # Format p-values for Prompt 1
                p1_values = []
                if not prompt1_data.empty:
                    p1_row = prompt1_data.iloc[0]
                    for factor in factors:
                        p_val = p1_row[factor]
                        if pd.isna(p_val):
                            p1_values.append("na")
                        elif p_val < 0.05:
                            p1_values.append(f"\\textbf{{{p_val:.3f}}}")
                        else:
                            p1_values.append(f"{p_val:.3f}")
                else:
                    p1_values = ["na"] * len(factors)
                
                # Format p-values for Prompt 2
                p2_values = []
                if not prompt2_data.empty:
                    p2_row = prompt2_data.iloc[0]
                    for factor in factors:
                        p_val = p2_row[factor]
                        if pd.isna(p_val):
                            p2_values.append("na")
                        elif p_val < 0.05:
                            p2_values.append(f"\\textbf{{{p_val:.3f}}}")
                        else:
                            p2_values.append(f"{p_val:.3f}")
                else:
                    p2_values = ["na"] * len(factors)
                
                # Show LLM name in each row
                llm_name = llm_info.get(llm, {}).get("name", llm.capitalize())
                axis_display = axis_display_mapping.get(axis, axis)
                prompt_type_display = prompt_type_display_mapping.get(prompt_type, prompt_type)
                
                # Color row gray if dimension is combined and prompt type is combined
                row_color = ""
                if axis == 'both' and prompt_type == 'all':
                    row_color = "\\rowcolor{gray!20}"
                
                latex_code_big += f"        {row_color}{llm_name} & {axis_display} & {prompt_type_display} & {' & '.join(p1_values)} & {' & '.join(p2_values)} \\\\\n"

    latex_code_big += """
\\end{longtable}
\\end{center}
"""

    with open(out_dir / "personas_big.tex", 'w') as f:
        f.write(latex_code_big)
    print(f"LaTeX table saved to {out_dir / 'personas_big.tex'}")


    # Create small LaTeX table (12 columns: 6 for prompt 1, 6 for prompt 2)
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

    # Sort LLMs according to target_llms order for small table
    available_llms_small = list(both_df['llm'].unique())
    unique_llms_small = [llm for llm in target_llms if llm in available_llms_small]
    remaining_llms_small = [llm for llm in available_llms_small if llm not in target_llms]
    unique_llms_small.extend(sorted(remaining_llms_small))

    for llm in unique_llms_small:
        both_df_llm = both_df[both_df['llm'] == llm]
        
        # Debug: Print llama4 data
        if llm == 'llama4':
            print(f"\nDebug llama4 in both_df:")
            print(f"both_df_llm shape: {both_df_llm.shape}")
            print(f"both_df_llm:\n{both_df_llm}")
            print(f"Prompts in both_df_llm: {both_df_llm['prompt'].unique()}")
        
        # Get prompt 1 and prompt 2 data (handle both string and int)
        prompt1_data = both_df_llm[(both_df_llm['prompt'] == '1') | (both_df_llm['prompt'] == 1)]
        prompt2_data = both_df_llm[(both_df_llm['prompt'] == '2') | (both_df_llm['prompt'] == 2)]
        
        # Debug: Print llama4 prompt data
        if llm == 'llama4':
            print(f"prompt1_data shape: {prompt1_data.shape}")
            print(f"prompt2_data shape: {prompt2_data.shape}")
            if not prompt1_data.empty:
                print(f"prompt1_data:\n{prompt1_data[['sex', 'ethnicity', 'education', 'political_view', 'age_bin', 'income_bin']]}")
            if not prompt2_data.empty:
                print(f"prompt2_data:\n{prompt2_data[['sex', 'ethnicity', 'education', 'political_view', 'age_bin', 'income_bin']]}")
        
        # Process each factor for prompt 1
        p1_symbols = []
        for factor in ['sex', 'ethnicity', 'education', 'political_view', 'age_bin', 'income_bin']:
            if not prompt1_data.empty:
                p1_val = prompt1_data.iloc[0][factor]
                if pd.isna(p1_val):
                    p1_symbols.append("---")
                elif p1_val < 0.005:
                    p1_symbols.append("$\\ast\\ast$")
                elif p1_val < 0.025:
                    p1_symbols.append("$\\ast$")
                else:
                    p1_symbols.append("---")
            else:
                p1_symbols.append("---")
        
        # Process each factor for prompt 2
        p2_symbols = []
        for factor in ['sex', 'ethnicity', 'education', 'political_view', 'age_bin', 'income_bin']:
            if not prompt2_data.empty:
                p2_val = prompt2_data.iloc[0][factor]
                if pd.isna(p2_val):
                    p2_symbols.append("---")
                elif p2_val < 0.005:
                    p2_symbols.append("$\\ast\\ast$")
                elif p2_val < 0.025:
                    p2_symbols.append("$\\ast$")
                else:
                    p2_symbols.append("---")
            else:
                p2_symbols.append("---")
        
        llm_name = llm_info.get(llm, {}).get("name", llm.capitalize())
        background_color = get_latex_color(llm)
        # Combine all 12 symbols (6 for prompt 1, 6 for prompt 2)
        all_symbols = p1_symbols + p2_symbols
        symbols_str = " & ".join([f"{background_color}{sym}" for sym in all_symbols])
        latex_code_small += f"        {background_color}{llm_name} & {symbols_str} \\\\\n"


    latex_code_small += """
            \\bottomrule
        \\end{tabular}
        }
    \\end{table}
    """

    with open(out_dir / "personas_small.tex", 'w') as f:
        f.write(latex_code_small)
    print(f"LaTeX table saved to {out_dir / 'personas_small.tex'}")

    print(f"\nComprehensive results shape: {comprehensive_df.shape}")
    print(f"Both axis results shape: {both_df.shape}")
    print(f"\nFirst few rows of comprehensive results:")
    print(comprehensive_df.head())

if __name__ == "__main__":
    run_persona_significance()