import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import shapiro, mannwhitneyu, ttest_ind
import numpy as np
import sys

# Add evaluation directory to path to allow importing utils
script_dir = Path(__file__).resolve().parent
evaluation_dir = script_dir.parent
if str(evaluation_dir) not in sys.path:
    sys.path.insert(0, str(evaluation_dir))

from utils import target_llms, opinion_llms, llm_colors, llm_info

def run_no_persona_significance():
    # Get LLMs from the CSV file instead of hardcoding
    out_dir = Path("data/interim_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    claims = pd.read_csv("data/claims_metadata.csv").set_index("claim_id")

    label_to_num = {
        "false": 1, "mostly-false": 2,
        "half-true": 3, "mostly-true": 4, "true": 5,
    }

    # Process each prompt type
    prompt_types = ["simple", "chain_of_thought", "all"]
    
    for prompt_type in prompt_types:
        print(f"\n{'='*80}")
        print(f"Processing prompt type: {prompt_type}")
        print(f"{'='*80}\n")
        
        # Perform Shapiro-Wilk tests for normality for each LLM and prompt
        test_results = {}

        for prompt in ["1", "2"]:
            hpc_path = f"data/claim_matrices/{prompt_type}/no_persona_{prompt}.csv"
            api_path = f"data/claim_matrices/{prompt_type}/no_persona_api_{prompt}.csv"
        
            try:
                df_hpc = pd.read_csv(hpc_path, index_col=0)
                df_api = pd.read_csv(api_path, index_col=0)
                df_long = pd.concat([df_hpc, df_api], axis=0)
            except FileNotFoundError as e:
                print(f"Warning: Could not find data files for {prompt_type}, prompt {prompt}: {e}")
                continue

            df_long = df_long.T.merge(claims[["party", "label", "axis"]], left_index=True, right_index=True, how="left")
            df_long["label"] = df_long["label"].map(label_to_num)

            # Calculate Mean Error and Mean Absolute Error for all columns except "party", "label", "axis"
            for col in df_long.columns:
                if col not in ["party", "label", "axis"]:
                    df_long[f"{col}_me"] = df_long[col] - df_long["label"]  # Mean Error
                    df_long[f"{col}_mae"] = abs(df_long[col] - df_long["label"])  # Mean Absolute Error

            # Remove original columns and label column
            cols_to_remove = [col for col in df_long.columns if col not in ["party", "axis"] and not col.endswith("_me") and not col.endswith("_mae")]
            df_long = df_long.drop(columns=cols_to_remove)

            # Initialize results for this prompt
            prompt_results = {}
            
            # Test each LLM
            for llm in target_llms + opinion_llms:
                llm_results = {}
                
                # Test both ME and MAE for each axis
                for axis in ["economic", "social", "both"]:
                    axis_results = {}
                    
                    for metric in ["me", "mae"]:
                        col_name = f"{llm}_{metric}"
                        
                        # Check if column exists in dataframe
                        if col_name not in df_long.columns:
                            # Create empty results for missing column
                            metric_results = {
                                'rep': {
                                    'statistic': np.nan,
                                    'p_value': np.nan,
                                    'n': 0,
                                    'mean': np.nan,
                                    'std': np.nan,
                                    'normal': 'Insufficient data'
                                },
                                'dem': {
                                    'statistic': np.nan,
                                    'p_value': np.nan,
                                    'n': 0,
                                    'mean': np.nan,
                                    'std': np.nan,
                                    'normal': 'Insufficient data'
                                }
                            }
                            axis_results[metric] = metric_results
                            continue
                        
                        if axis == "both":
                            rep_data = df_long[(df_long["party"] == "Republican") & (df_long["axis"].isin(["economic", "social"]))][col_name].dropna()
                            dem_data = df_long[(df_long["party"] == "Democrat") & (df_long["axis"].isin(["economic", "social"]))][col_name].dropna()
                        else:
                            rep_data = df_long[(df_long["party"] == "Republican") & (df_long["axis"] == axis)][col_name].dropna()
                            dem_data = df_long[(df_long["party"] == "Democrat") & (df_long["axis"] == axis)][col_name].dropna()
                        
                        # Sort by index to ensure consistent ordering (important for reproducible test results)
                        rep_data = rep_data.sort_index()
                        dem_data = dem_data.sort_index()
                        
                        metric_results = {}
                        
                        # Test normality for Republican group
                        if len(rep_data) > 3:  # Shapiro-Wilk requires at least 3 observations
                            stat_rep, p_val_rep = shapiro(rep_data)
                            metric_results['rep'] = {
                                'statistic': stat_rep,
                                'p_value': p_val_rep,
                                'n': len(rep_data),
                                'mean': rep_data.mean(),
                                'std': rep_data.std(),
                                'normal': 'Yes' if p_val_rep >= 0.05 else 'No'
                            }
                        else:
                            metric_results['rep'] = {
                                'statistic': np.nan,
                                'p_value': np.nan,
                                'n': len(rep_data),
                                'mean': rep_data.mean() if len(rep_data) > 0 else np.nan,
                                'std': rep_data.std() if len(rep_data) > 0 else np.nan,
                                'normal': 'Insufficient data'
                            }
                        
                        # Test normality for Democrat group
                        if len(dem_data) > 3:  # Shapiro-Wilk requires at least 3 observations
                            stat_dem, p_val_dem = shapiro(dem_data)
                            metric_results['dem'] = {
                                'statistic': stat_dem,
                                'p_value': p_val_dem,
                                'n': len(dem_data),
                                'mean': dem_data.mean(),
                                'std': dem_data.std(),
                                'normal': 'Yes' if p_val_dem >= 0.05 else 'No'
                            }
                        else:
                            metric_results['dem'] = {
                                'statistic': np.nan,
                                'p_value': np.nan,
                                'n': len(dem_data),
                                'mean': dem_data.mean() if len(dem_data) > 0 else np.nan,
                                'std': dem_data.std() if len(dem_data) > 0 else np.nan,
                                'normal': 'Insufficient data'
                            }
                        
                        # Perform adaptive significance test (t-test or Mann-Whitney U)
                        rep_normal = metric_results['rep']['normal'] == 'Yes'
                        dem_normal = metric_results['dem']['normal'] == 'Yes'
                        both_normal = rep_normal and dem_normal
                        
                        if len(rep_data) > 0 and len(dem_data) > 0:
                            # Choose test based on normality
                            if both_normal:
                                # Use t-test (Welch's t-test for unequal variances)
                                stat, test_p_val = ttest_ind(rep_data, dem_data, equal_var=False)
                                test_type = 't'
                            else:
                                # Use Mann-Whitney U test
                                stat, test_p_val = mannwhitneyu(rep_data, dem_data, alternative='two-sided')
                                test_type = 'U'
                            
                            metric_results['test_statistic'] = stat
                            metric_results['test_p_value'] = test_p_val
                            metric_results['test_type'] = test_type
                        else:
                            metric_results['test_statistic'] = np.nan
                            metric_results['test_p_value'] = np.nan
                            metric_results['test_type'] = None
                        
                        axis_results[metric] = metric_results
                    
                    llm_results[axis] = axis_results
                
                prompt_results[llm] = llm_results
            
            test_results[f"prompt_{prompt}"] = prompt_results

        # Create comprehensive results table
        results_data = []

        for prompt, results in test_results.items():
            for llm, llm_results in results.items():
                for axis, axis_data in llm_results.items():
                    for metric, test_data in axis_data.items():
                        # Republican results
                        rep_data = test_data['rep']
                        results_data.append({
                            'Prompt': prompt,
                            'LLM': llm,
                            'Axis': axis,
                            'Metric': metric.upper(),  # ME or MAE
                            'Party': 'Republican',
                            'W_statistic': rep_data['statistic'],
                            'p_value': rep_data['p_value'],
                            'n': rep_data['n'],
                            'mean': rep_data['mean'],
                            'std': rep_data['std'],
                            'normal': rep_data['normal'],
                            'p_value_rounded': round(rep_data['p_value'], 4) if not np.isnan(rep_data['p_value']) else np.nan
                        })
                        
                        # Democrat results
                        dem_data = test_data['dem']
                        results_data.append({
                            'Prompt': prompt,
                            'LLM': llm,
                            'Axis': axis,
                            'Metric': metric.upper(),  # ME or MAE
                            'Party': 'Democrat',
                            'W_statistic': dem_data['statistic'],
                            'p_value': dem_data['p_value'],
                            'n': dem_data['n'],
                            'mean': dem_data['mean'],
                            'std': dem_data['std'],
                            'normal': dem_data['normal'],
                            'p_value_rounded': round(dem_data['p_value'], 4) if not np.isnan(dem_data['p_value']) else np.nan
                        })

        # Create DataFrame
        results_df = pd.DataFrame(results_data)

        # Create simplified table with one row per statistical test
        test_summary_data = []

        # Group by LLM, Prompt, Axis, Metric to create one row per test
        for (llm, prompt, axis, metric), group in results_df.groupby(['LLM', 'Prompt', 'Axis', 'Metric']):
            rep_row = group[group['Party'] == 'Republican'].iloc[0] if len(group[group['Party'] == 'Republican']) > 0 else None
            dem_row = group[group['Party'] == 'Democrat'].iloc[0] if len(group[group['Party'] == 'Democrat']) > 0 else None
            
            if rep_row is not None and dem_row is not None:
                rep_p_value = rep_row['p_value']
                dem_p_value = dem_row['p_value']
                rep_mean = rep_row['mean']
                dem_mean = dem_row['mean']
                rep_normal = rep_row['normal'] == 'Yes'
                dem_normal = dem_row['normal'] == 'Yes'
                both_normal = rep_normal and dem_normal
                
                # Get test results from the metric data
                # We need to access the test results from the nested structure
                # Find the corresponding test data from test_results
                # Note: prompt is already "prompt_1" or "prompt_2" from the groupby
                try:
                    test_data = test_results[prompt][llm][axis][metric.lower()]
                    test_p_value = test_data.get('test_p_value', np.nan)
                    test_type = test_data.get('test_type', None)
                except (KeyError, TypeError) as e:
                    # If test data is not available, set to None
                    # Debug: print error for troubleshooting
                    print(f"Warning: Could not find test data for {prompt}, {llm}, {axis}, {metric.lower()}: {e}")
                    test_p_value = None
                    test_type = None
                
                # Helper function to safely check for NaN or None
                def safe_value(val):
                    if val is None:
                        return None
                    if isinstance(val, (int, float)):
                        return val if not np.isnan(val) else None
                    return val
                
                test_summary_data.append({
                    'LLM': llm,
                    'Prompt': prompt,
                    'Axis': axis,
                    'Metric': metric.upper(),  # ME or MAE
                    'Republican_mean': safe_value(rep_mean),
                    'Republican_p_value': safe_value(rep_p_value),
                    'Democrat_mean': safe_value(dem_mean),
                    'Democrat_p_value': safe_value(dem_p_value),
                    'Both_Normal': both_normal,
                    'Test_p_value': safe_value(test_p_value),
                    'Test_type': test_type
                })

        # Create DataFrame - ensure it has the correct structure even if empty
        if len(test_summary_data) == 0:
            # Create empty DataFrame with correct column structure
            test_summary_df = pd.DataFrame(columns=[
                'LLM', 'Prompt', 'Axis', 'Metric',
                'Republican_mean', 'Republican_p_value', 'Democrat_mean', 'Democrat_p_value', 'Both_Normal',
                'Test_p_value', 'Test_type'
            ])
            print(f"Warning: No data found for {prompt_type}. Creating empty summary file.")
        else:
            test_summary_df = pd.DataFrame(test_summary_data)

        # Save as CSV with prompt_type in filename
        output_path = out_dir / f"shapiro_wilk_summary_{prompt_type}.csv"
        # Replace None with np.nan for proper CSV handling
        test_summary_df = test_summary_df.replace({None: np.nan})
        test_summary_df.to_csv(output_path, index=False, na_rep='')
        print(f"Shapiro-Wilk summary saved to: {output_path} ({len(test_summary_df)} rows)")
        # Debug: Check if Test_p_value has any non-null values
        if 'Test_p_value' in test_summary_df.columns:
            non_null_count = test_summary_df['Test_p_value'].notna().sum()
            print(f"  - Test_p_value: {non_null_count} non-null values out of {len(test_summary_df)}")

        # Calculate percentage of True values in Both_Normal column
        if len(test_summary_df) > 0:
            both_normal_count = test_summary_df['Both_Normal'].sum()
            total_tests = len(test_summary_df)
            percentage_both_normal = (both_normal_count / total_tests) * 100
            print(f"Percentage of tests where both groups are normally distributed: {percentage_both_normal:.1f}% ({both_normal_count}/{total_tests})")
        else:
            print("No tests to calculate percentage for.")

if __name__ == "__main__":
    run_no_persona_significance()



