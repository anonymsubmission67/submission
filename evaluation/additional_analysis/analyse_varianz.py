"""
Script to analyze variance in LLM responses.

Calculates:
- For no_persona: Variance across claims for each model (Prompt 1 and Prompt 2)

Creates a table with LLMs in rows and 2 columns:
- var_p1: Variance across claims for Prompt 1 (no_persona)
- var_p2: Variance across claims for Prompt 2 (no_persona)

Outputs the table as LaTeX format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import utils
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from utils import target_llms, llm_info


def calculate_no_persona_variance(prompt_num):
    """
    Calculate variance across claims for each model in no_persona data.
    Returns a Series with model_id as index and variance as values.
    """
    # Try both api and server format variance files
    paths_to_try = [
        f"data/claim_matrices/all/no_persona_api_{prompt_num}_variance.csv",
        f"data/claim_matrices/all/no_persona_{prompt_num}_variance.csv",
    ]
    
    dataframes = []
    for path in paths_to_try:
        if Path(path).exists():
            df = pd.read_csv(path, index_col=0)
            if not df.empty:
                dataframes.append(df)
                print(f"  Loaded: {path} ({len(df)} models)")
    
    if not dataframes:
        print(f"Warning: No no_persona variance data found for prompt {prompt_num}")
        return pd.Series(dtype=float)
    
    # Concatenate dataframes, combining data for models that appear in both
    # For models in both, we'll take the mean of their values
    if len(dataframes) > 1:
        # Align indices and columns, then combine
        df_matrix = pd.concat(dataframes)
        # Group by index (model_id) and take mean for overlapping models
        df_matrix = df_matrix.groupby(df_matrix.index).mean()
    else:
        df_matrix = dataframes[0]
    
    # Convert to numeric, coercing errors to NaN
    df_matrix = df_matrix.apply(pd.to_numeric, errors='coerce')
    
    # Calculate variance across claims for each model (mean of variances across claims)
    # This gives us the average variance per model across all claims
    # Skip NaN values when calculating mean
    model_variances = df_matrix.mean(axis=1, skipna=True)
    
    return model_variances


def get_model_variance_no_persona(model_name, prompt_num, var_series):
    """Get variance for a specific model from no_persona data."""
    if model_name in var_series.index:
        return var_series[model_name]
    return np.nan


def format_latex_table(df_results):
    """
    Convert DataFrame to LaTeX table format.
    Uses model names from llm_info if available.
    """
    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\begin{tabular}{lcc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Model & party-agnostic & party-aware \\\\")
    latex_lines.append("\\midrule")
    
    for model in df_results.index:
        var_p1 = df_results.loc[model, 'var_p1']
        var_p2 = df_results.loc[model, 'var_p2']
        
        # Format values (handle NaN)
        if pd.isna(var_p1):
            var_p1_str = "---"
        else:
            var_p1_str = f"{var_p1:.4f}"
        
        if pd.isna(var_p2):
            var_p2_str = "---"
        else:
            var_p2_str = f"{var_p2:.4f}"
        
        # Get display name from llm_info if available, otherwise use model name
        if model in llm_info and 'name' in llm_info[model]:
            model_display = llm_info[model]['name']
        else:
            model_display = model
        
        # Escape special characters for LaTeX
        model_escaped = model_display.replace('_', '\\_').replace('&', '\\&').replace('%', '\\%')
        
        latex_lines.append(f"{model_escaped} & {var_p1_str} & {var_p2_str} \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\caption{Variance across claims for each model (Prompt 1 and Prompt 2)}")
    latex_lines.append("\\label{tab:variance_analysis}")
    latex_lines.append("\\end{table}")
    
    return "\n".join(latex_lines)


def main():
    """Main function to create variance analysis table."""
    print("=" * 80)
    print("Variance Analysis")
    print("=" * 80)
    
    # Calculate variances for no_persona (across claims)
    print("\nCalculating no_persona variances (across claims)...")
    var_p1 = calculate_no_persona_variance("1")
    var_p2 = calculate_no_persona_variance("2")
    
    # Print diagnostic information about available models
    print(f"\nDiagnostic Information:")
    print(f"Number of models in var_p1: {len(var_p1)}")
    print(f"Models in var_p1: {sorted(var_p1.index.tolist())}")
    print(f"Number of models in var_p2: {len(var_p2)}")
    print(f"Models in var_p2: {sorted(var_p2.index.tolist())}")
    
    # Process each model from target_llms
    print(f"\n{'=' * 80}")
    print("Processing models from target_llms:")
    print(f"{'=' * 80}")
    
    results = []
    for model in target_llms:
        print(f"\n--- Processing model: {model} ---")
        
        # Get no_persona variances
        var_p1_val = get_model_variance_no_persona(model, "1", var_p1)
        var_p2_val = get_model_variance_no_persona(model, "2", var_p2)
        
        # Debug output
        print(f"  var_p1: {var_p1_val}")
        print(f"  var_p2: {var_p2_val}")
        
        results.append({
            'model': model,
            'var_p1': var_p1_val,
            'var_p2': var_p2_val,
        })
    
    # Also include any models not in target_llms but present in the data
    all_models_in_data = set(var_p1.index) | set(var_p2.index)
    models_not_in_target = sorted(all_models_in_data - set(target_llms))
    
    if models_not_in_target:
        print(f"\n{'=' * 80}")
        print(f"Additional models found (not in target_llms): {models_not_in_target}")
        print(f"{'=' * 80}")
        
        for model in models_not_in_target:
            print(f"\n--- Processing model: {model} ---")
            
            var_p1_val = get_model_variance_no_persona(model, "1", var_p1)
            var_p2_val = get_model_variance_no_persona(model, "2", var_p2)
            
            print(f"  var_p1: {var_p1_val}")
            print(f"  var_p2: {var_p2_val}")
            
            results.append({
                'model': model,
                'var_p1': var_p1_val,
                'var_p2': var_p2_val,
            })
    
    df_results = pd.DataFrame(results)
    df_results = df_results.set_index('model')
    
    # Sort by target_llms order if available
    if target_llms:
        # Create ordered list: first target_llms, then others alphabetically
        ordered_models = [m for m in target_llms if m in df_results.index]
        ordered_models.extend([m for m in sorted(df_results.index) if m not in ordered_models])
        df_results = df_results.reindex(ordered_models)
    
    # Print summary
    print(f"\n{'=' * 80}")
    print("Summary:")
    print(f"{'=' * 80}")
    print(f"Total models processed: {len(df_results)}")
    print(f"\nNaN counts:")
    print(f"  var_p1: {df_results['var_p1'].isna().sum()}")
    print(f"  var_p2: {df_results['var_p2'].isna().sum()}")
    

    # Generate and save LaTeX table
    latex_table = format_latex_table(df_results)
    output_path_tex = Path("output/tables/variance_analysis.tex")
    output_path_tex.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path_tex, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {output_path_tex}")
    
    print("\nVariance Analysis Table:")
    print("=" * 80)
    print(df_results.to_string())
    print("=" * 80)
    
    # Print LaTeX table
    print("\nLaTeX Table:")
    print("=" * 80)
    print(latex_table)
    print("=" * 80)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Number of models: {len(df_results)}")
    print(f"\nMean variance across claims (Prompt 1): {df_results['var_p1'].mean(skipna=True):.4f}")
    print(f"Mean variance across claims (Prompt 2): {df_results['var_p2'].mean(skipna=True):.4f}")


if __name__ == "__main__":
    main()

