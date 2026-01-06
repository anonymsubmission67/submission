import pandas as pd
import numpy as np
import json
from sklearn.metrics import f1_score
from pathlib import Path
import sys

# Add evaluation directory to path to allow importing utils
script_dir = Path(__file__).resolve().parent
evaluation_dir = script_dir.parent
if str(evaluation_dir) not in sys.path:
    sys.path.insert(0, str(evaluation_dir))

from utils import target_llms, opinion_llms, llm_info, PROMPT_TYPE, PERSONAS_PATH

def build_persona_correlation_table():
        
    # Load claims metadata
    claims = pd.read_csv("data/claims_metadata.csv").set_index("claim_id")
    label_to_num = {
        "pants-fire": 0, "false": 1, "mostly-false": 2,
        "half-true": 3, "mostly-true": 4, "true": 5,
    }

    # Load personas metadata
    personas_metadata = pd.read_csv(PERSONAS_PATH)
    # Set index to 'id' for filtering
    personas_metadata_indexed = personas_metadata.set_index("id")

    llms = target_llms + opinion_llms
    results = []

    print("Processing persona correlations...")

    for llm in llms:
        print(f"\nProcessing {llm}...")
        
        try:
            # Load personas compass data
            df_personas_compass = pd.read_csv(f"data/claim_matrices/{PROMPT_TYPE}/personas_compass/{llm}.csv", index_col=0)

            # Load personas prompt 1 and prompt 2 data
            df_personas_1 = pd.read_csv(f"data/claim_matrices/{PROMPT_TYPE}/personas_1/{llm}.csv", index_col=0)
            df_personas_2 = pd.read_csv(f"data/claim_matrices/{PROMPT_TYPE}/personas_2/{llm}.csv", index_col=0)
            
            # Get personas that exist in all datasets
            common_personas = df_personas_compass.index.intersection(df_personas_1.index).intersection(df_personas_2.index)
            
            # Filter to only include personas from PERSONAS_PATH
            common_personas = common_personas[common_personas.isin(personas_metadata_indexed.index)]
            
            if len(common_personas) == 0:
                print(f"  Warning: No common personas found for {llm} (after filtering by PERSONAS_PATH)")
                continue
            
            print(f"  Found {len(common_personas)} common personas (filtered by PERSONAS_PATH)")
            
            # Helper function to calculate MED and F1 for a given prompt dataframe
            def calculate_metrics_for_prompt(df_personas_prompt, persona_id):
                persona_predictions = df_personas_prompt.loc[persona_id].dropna()
                
                # Get corresponding true labels
                df_personas_with_labels = df_personas_prompt.T.merge(claims[["party", "label"]], left_index=True, right_index=True, how="left")
                df_personas_with_labels["label"] = df_personas_with_labels["label"].map(label_to_num)
                
                # Get true labels for this persona
                true_labels = df_personas_with_labels.loc[persona_predictions.index, "label"].dropna()
                
                # Align predictions and true labels
                common_indices = persona_predictions.index.intersection(true_labels.index)
                if len(common_indices) > 0:
                    aligned_predictions = persona_predictions.loc[common_indices]
                    aligned_true_labels = true_labels.loc[common_indices]
                    
                    # Get party information for these claims
                    df_with_party = df_personas_with_labels.loc[common_indices]
                    
                    # Group by party and calculate mean bias
                    bias_by_party = df_with_party.groupby("party")[persona_id].mean()
                    
                    # Calculate ME difference (Republican - Democrat)
                    me_diff = bias_by_party.get("Republican", 0) - bias_by_party.get("Democrat", 0)
                    
                    # Calculate F1 Score
                    rounded_predictions = np.round(aligned_predictions).astype(int)
                    rounded_predictions = np.clip(rounded_predictions, 0, 5)
                    f1 = f1_score(aligned_true_labels, rounded_predictions, average='macro')
                    
                    return me_diff, f1
                else:
                    return None, None
            
            # Initialize lists for correlation calculation (for both prompts)
            pol_bias_values = []
            me_diff_values_p1 = []
            f1_values_p1 = []
            me_diff_values_p2 = []
            f1_values_p2 = []
            
            # Process each persona
            for persona_id in common_personas:
                # Calculate political bias from compass data
                persona_compass_scores = df_personas_compass.loc[persona_id].dropna()
                
                if len(persona_compass_scores) > 0:
                    # Map from 0-4 scale to -1 to +1 scale
                    # Original: 1=strongly disagree, 2=disagree, 3=agree, 4=strongly agree
                    # Mapped: -1=strongly disagree, -0.33=disagree, +0.33=agree, +1=strongly agree
                    mapped_scores = (persona_compass_scores - 2.5) / 1.5
                    pol_bias = mapped_scores.mean()
                    pol_bias_values.append(pol_bias)
                else:
                    continue
                
                # Calculate MED and F1 for prompt 1
                me_diff_p1, f1_p1 = calculate_metrics_for_prompt(df_personas_1, persona_id)
                if me_diff_p1 is not None and f1_p1 is not None:
                    me_diff_values_p1.append(me_diff_p1)
                    f1_values_p1.append(f1_p1)
                else:
                    continue
                
                # Calculate MED and F1 for prompt 2
                me_diff_p2, f1_p2 = calculate_metrics_for_prompt(df_personas_2, persona_id)
                if me_diff_p2 is not None and f1_p2 is not None:
                    me_diff_values_p2.append(me_diff_p2)
                    f1_values_p2.append(f1_p2)
                else:
                    continue
            
            # Calculate correlations for both prompts
            min_length = min(len(pol_bias_values), len(me_diff_values_p1), len(f1_values_p1), 
                           len(me_diff_values_p2), len(f1_values_p2))
            
            if min_length >= 3:
                # Ensure all arrays have the same length
                pol_bias_array = np.array(pol_bias_values[:min_length])
                me_diff_array_p1 = np.array(me_diff_values_p1[:min_length])
                f1_array_p1 = np.array(f1_values_p1[:min_length])
                me_diff_array_p2 = np.array(me_diff_values_p2[:min_length])
                f1_array_p2 = np.array(f1_values_p2[:min_length])
                
                # Calculate correlations for prompt 1
                corr_pol_bias_med_p1 = np.corrcoef(pol_bias_array, me_diff_array_p1)[0, 1]
                corr_pol_bias_f1_p1 = np.corrcoef(pol_bias_array, f1_array_p1)[0, 1]
                
                # Calculate correlations for prompt 2
                corr_pol_bias_med_p2 = np.corrcoef(pol_bias_array, me_diff_array_p2)[0, 1]
                corr_pol_bias_f1_p2 = np.corrcoef(pol_bias_array, f1_array_p2)[0, 1]
                
                # Get LLM name
                llm_name = llm_info.get(llm, {}).get("name", llm.capitalize())
                
                result = {
                    "llm": llm_name,
                    "n_personas": min_length,
                    "corr_pol_bias_med_p1": round(corr_pol_bias_med_p1, 3),
                    "corr_pol_bias_f1_p1": round(corr_pol_bias_f1_p1, 3),
                    "corr_pol_bias_med_p2": round(corr_pol_bias_med_p2, 3),
                    "corr_pol_bias_f1_p2": round(corr_pol_bias_f1_p2, 3)
                }
                
                results.append(result)
                print(f"  {llm_name}: n={min_length}, P1: corr(PB, MED)={corr_pol_bias_med_p1:.3f}, corr(PB, F1)={corr_pol_bias_f1_p1:.3f}, P2: corr(PB, MED)={corr_pol_bias_med_p2:.3f}, corr(PB, F1)={corr_pol_bias_f1_p2:.3f}")
            else:
                print(f"  Warning: Insufficient data for {llm} (n={min_length})")
        
        except FileNotFoundError as e:
            print(f"  Warning: File not found for {llm}: {e}")
            continue

    # Calculate averages
    if results:
        avg_corr_pol_bias_med_p1 = np.mean([r["corr_pol_bias_med_p1"] for r in results])
        avg_corr_pol_bias_f1_p1 = np.mean([r["corr_pol_bias_f1_p1"] for r in results])
        avg_corr_pol_bias_med_p2 = np.mean([r["corr_pol_bias_med_p2"] for r in results])
        avg_corr_pol_bias_f1_p2 = np.mean([r["corr_pol_bias_f1_p2"] for r in results])
        
        # Add average row
        avg_result = {
            "llm": "Average",
            "n_personas": "",
            "corr_pol_bias_med_p1": round(avg_corr_pol_bias_med_p1, 3),
            "corr_pol_bias_f1_p1": round(avg_corr_pol_bias_f1_p1, 3),
            "corr_pol_bias_med_p2": round(avg_corr_pol_bias_med_p2, 3),
            "corr_pol_bias_f1_p2": round(avg_corr_pol_bias_f1_p2, 3)
        }
        results.append(avg_result)

    # Create LaTeX table
    output_dir = Path("output/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort results according to target_llms order (excluding Average)
    results_without_avg = [r for r in results if r["llm"] != "Average"]
    avg_result = [r for r in results if r["llm"] == "Average"]
    
    # Create a mapping from LLM name to result
    results_dict = {r["llm"]: r for r in results_without_avg}
    
    # Get LLM names from results and sort according to target_llms
    available_llm_names = list(results_dict.keys())
    # Create reverse mapping from LLM name to original llm key
    name_to_llm = {}
    for llm_key in target_llms + opinion_llms:
        llm_name = llm_info.get(llm_key, {}).get("name", llm_key.capitalize())
        if llm_name in available_llm_names:
            name_to_llm[llm_name] = llm_key
    
    # Sort by target_llms order
    sorted_results = []
    for llm_key in target_llms + opinion_llms:
        llm_name = llm_info.get(llm_key, {}).get("name", llm_key.capitalize())
        if llm_name in results_dict:
            sorted_results.append(results_dict[llm_name])
    
    # Add any remaining LLMs not in target_llms
    remaining_llm_names = [name for name in available_llm_names if name not in [r["llm"] for r in sorted_results]]
    for llm_name in sorted(remaining_llm_names):
        sorted_results.append(results_dict[llm_name])
    
    # Add average at the end
    if avg_result:
        sorted_results.extend(avg_result)

    latex_code = f"""
    \\begin{{table}}[h]
        \\centering
        \\caption{{Correlations between Political Bias and Performance Metrics (Personas)}}
        \\label{{tab:persona_correlations}}
        \\begin{{tabular}}{{@{{}}l|cc|cc@{{}}}}
            \\toprule
            \\multirow{{2}}{{*}}{{\\textbf{{LLM}}}} & \\multicolumn{{2}}{{c|}}{{\\textbf{{Prompt 1 (without source)}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{Prompt 2 (with source)}}}} \\\\
            \\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}}
            & \\textbf{{Corr(PB-A, PB-T)}} & \\textbf{{Corr(PB-A, F1)}} & \\textbf{{Corr(PB-A, PB-T)}} & \\textbf{{Corr(PB-A, F1)}} \\\\
            \\midrule
    """

    for result in sorted_results:
        if result["llm"] == "Average":
            latex_code += f"        \\midrule\n"
            latex_code += f"        \\textbf{{{result['llm']}}} & \\textbf{{{result['corr_pol_bias_med_p1']}}} & \\textbf{{{result['corr_pol_bias_f1_p1']}}} & \\textbf{{{result['corr_pol_bias_med_p2']}}} & \\textbf{{{result['corr_pol_bias_f1_p2']}}} \\\\\n"
        else:
            latex_code += f"        {result['llm']} & {result['corr_pol_bias_med_p1']} & {result['corr_pol_bias_f1_p1']} & {result['corr_pol_bias_med_p2']} & {result['corr_pol_bias_f1_p2']} \\\\\n"

    latex_code += """
            \\bottomrule
        \\end{tabular}
    \\end{table}
    """

    # Save LaTeX table
    with open(output_dir / "persona_correlations.tex", 'w') as f:
        f.write(latex_code)

    print(f"\nLaTeX table saved to: {output_dir / 'persona_correlations.tex'}")
    print(f"Processed {len(results)-1} LLMs")  # -1 because we added the average row

if __name__ == "__main__":
    build_persona_correlation_table()