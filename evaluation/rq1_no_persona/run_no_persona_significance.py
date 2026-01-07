"""
Statistical significance testing for no-persona experiments.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import shapiro, mannwhitneyu, ttest_ind
import sys

# Add evaluation directory to path to allow importing utils
script_dir = Path(__file__).resolve().parent
evaluation_dir = script_dir.parent
if str(evaluation_dir) not in sys.path:
    sys.path.insert(0, str(evaluation_dir))

from utils import target_llms, opinion_llms

# Constants
LABEL_TO_NUM = {
    "false": 1,
    "mostly-false": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5,
}
PROMPT_TYPES = ["simple", "chain_of_thought", "all"]
PROMPTS = ["1", "2"]
AXES = ["economic", "social", "both"]
METRICS = ["me", "mae"]
MIN_SAMPLE_SIZE = 3  # Minimum for Shapiro-Wilk test
NORMALITY_THRESHOLD = 0.05


def load_data(prompt_type: str, prompt: str) -> pd.DataFrame:
    """
    Load and prepare data for a specific prompt type and prompt number.
    """
    hpc_path = f"data/claim_matrices/{prompt_type}/no_persona_{prompt}.csv"
    api_path = f"data/claim_matrices/{prompt_type}/no_persona_api_{prompt}.csv"
    
    try:
        df_hpc = pd.read_csv(hpc_path, index_col=0)
        df_api = pd.read_csv(api_path, index_col=0)
        df_long = pd.concat([df_hpc, df_api], axis=0)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find data files for {prompt_type}, prompt {prompt}: {e}"
        )
    
    # Load claims metadata and merge
    claims = pd.read_csv("data/claims_metadata.csv").set_index("claim_id")
    df_long = df_long.T.merge(
        claims[["party", "label", "axis"]],
        left_index=True,
        right_index=True,
        how="left"
    )
    df_long["label"] = df_long["label"].map(LABEL_TO_NUM)
    
    # Calculate Mean Error (ME) and Mean Absolute Error (MAE)
    for col in df_long.columns:
        if col not in ["party", "label", "axis"]:
            df_long[f"{col}_me"] = df_long[col] - df_long["label"]
            df_long[f"{col}_mae"] = abs(df_long[col] - df_long["label"])
    
    # Remove original columns and label, keep only ME/MAE columns
    cols_to_remove = [
        col for col in df_long.columns
        if col not in ["party", "axis"]
        and not col.endswith("_me")
        and not col.endswith("_mae")
    ]
    df_long = df_long.drop(columns=cols_to_remove)
    
    return df_long


def test_normality(data: pd.Series) -> dict:
    """
    Perform Shapiro-Wilk normality test on a data series.
    """
    if len(data) < MIN_SAMPLE_SIZE:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'n': len(data),
            'mean': data.mean() if len(data) > 0 else np.nan,
            'std': data.std() if len(data) > 0 else np.nan,
            'normal': 'Insufficient data'
        }
    
    stat, p_value = shapiro(data)
    return {
        'statistic': stat,
        'p_value': p_value,
        'n': len(data),
        'mean': data.mean(),
        'std': data.std(),
        'normal': 'Yes' if p_value >= NORMALITY_THRESHOLD else 'No'
    }


def perform_significance_test(rep_data: pd.Series, dem_data: pd.Series,
                             rep_normal: bool, dem_normal: bool) -> dict:
    """
    Perform adaptive significance test (t-test or Mann-Whitney U).
    """
    if len(rep_data) == 0 or len(dem_data) == 0:
        return {
            'test_statistic': np.nan,
            'test_p_value': np.nan,
            'test_type': None
        }
    
    both_normal = rep_normal and dem_normal
    
    if both_normal:
        # Use Welch's t-test (handles unequal variances)
        stat, p_value = ttest_ind(rep_data, dem_data, equal_var=False)
        test_type = 't'
    else:
        # Use Mann-Whitney U test (non-parametric)
        stat, p_value = mannwhitneyu(rep_data, dem_data, alternative='two-sided')
        test_type = 'U'
    
    return {
        'test_statistic': stat,
        'test_p_value': p_value,
        'test_type': test_type
    }


def extract_party_data(df: pd.DataFrame, col_name: str, axis: str,
                      party: str) -> pd.Series:
    """
    Extract data for a specific party and axis.
    """
    if axis == "both":
        data = df[
            (df["party"] == party) &
            (df["axis"].isin(["economic", "social"]))
        ][col_name].dropna()
    else:
        data = df[
            (df["party"] == party) &
            (df["axis"] == axis)
        ][col_name].dropna()
    
    return data.sort_index()  # Ensure consistent ordering


def process_llm_metric(df: pd.DataFrame, llm: str, axis: str,
                       metric: str) -> dict:
    """
    Process statistical tests for a specific LLM, axis, and metric.
    """
    col_name = f"{llm}_{metric}"
    
    if col_name not in df.columns:
        return {
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
            },
            'test_statistic': np.nan,
            'test_p_value': np.nan,
            'test_type': None
        }
    
    # Extract data for both parties
    rep_data = extract_party_data(df, col_name, axis, "Republican")
    dem_data = extract_party_data(df, col_name, axis, "Democrat")
    
    # Test normality
    rep_results = test_normality(rep_data)
    dem_results = test_normality(dem_data)
    
    # Perform significance test
    rep_normal = rep_results['normal'] == 'Yes'
    dem_normal = dem_results['normal'] == 'Yes'
    test_results = perform_significance_test(rep_data, dem_data, rep_normal, dem_normal)
    
    return {
        'rep': rep_results,
        'dem': dem_results,
        **test_results
    }




def _safe_value(val):
    """Helper function to safely handle NaN and None values."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return val if not np.isnan(val) else None
    return val


def _create_summary_table(test_results: dict) -> pd.DataFrame:
    """
    Create a summary table from nested test results.
    """
    summary_data = []
    
    for prompt_key, prompt_results in test_results.items():
        for llm, llm_results in prompt_results.items():
            for axis, axis_data in llm_results.items():
                for metric, metric_data in axis_data.items():
                    rep_data = metric_data['rep']
                    dem_data = metric_data['dem']
                    
                    rep_normal = rep_data['normal'] == 'Yes'
                    dem_normal = dem_data['normal'] == 'Yes'
                    both_normal = rep_normal and dem_normal
                    
                    summary_data.append({
                        'LLM': llm,
                        'Prompt': prompt_key,
                        'Axis': axis,
                        'Metric': metric.upper(),
                        'Republican_mean': _safe_value(rep_data['mean']),
                        'Republican_p_value': _safe_value(rep_data['p_value']),
                        'Democrat_mean': _safe_value(dem_data['mean']),
                        'Democrat_p_value': _safe_value(dem_data['p_value']),
                        'Both_Normal': both_normal,
                        'Test_p_value': _safe_value(metric_data.get('test_p_value')),
                        'Test_type': metric_data.get('test_type')
                    })
    
    return pd.DataFrame(summary_data)


def run_no_persona_significance():
    """
    Main function to run statistical significance tests for no-persona experiments.
    """
    out_dir = Path("data/interim_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process each prompt type
    for prompt_type in PROMPT_TYPES:
        print(f"\n{'='*80}")
        print(f"Processing prompt type: {prompt_type}")
        print(f"{'='*80}\n")
        
        test_results = {}

        # Process each prompt
        for prompt in PROMPTS:
            # Load and prepare data (load_data already handles merge and ME/MAE calculation)
            try:
                df_long = load_data(prompt_type, prompt)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue

            # Test each LLM, axis, and metric combination
            prompt_results = {}
            for llm in target_llms + opinion_llms:
                llm_results = {}
                for axis in AXES:
                    axis_results = {}
                    for metric in METRICS:
                        # Use the helper function to process this combination
                        metric_results = process_llm_metric(df_long, llm, axis, metric)
                        axis_results[metric] = metric_results
                    llm_results[axis] = axis_results
                prompt_results[llm] = llm_results
            
            test_results[f"prompt_{prompt}"] = prompt_results

        # Create summary table
        test_summary_df = _create_summary_table(test_results)

        # Save results
        if len(test_summary_df) == 0:
            test_summary_df = pd.DataFrame(columns=[
                'LLM', 'Prompt', 'Axis', 'Metric',
                'Republican_mean', 'Republican_p_value', 'Democrat_mean', 'Democrat_p_value', 'Both_Normal',
                'Test_p_value', 'Test_type'
            ])
            print(f"Warning: No data found for {prompt_type}. Creating empty summary file.")
        else:
            # Replace None with np.nan for proper CSV handling
            test_summary_df = test_summary_df.replace({None: np.nan})
        
        output_path = out_dir / f"shapiro_wilk_summary_{prompt_type}.csv"
        test_summary_df.to_csv(output_path, index=False, na_rep='')
        print(f"Shapiro-Wilk summary saved to: {output_path} ({len(test_summary_df)} rows)")
        
        # Print statistics
        if 'Test_p_value' in test_summary_df.columns:
            non_null_count = test_summary_df['Test_p_value'].notna().sum()
            print(f"  - Test_p_value: {non_null_count} non-null values out of {len(test_summary_df)}")

        if len(test_summary_df) > 0:
            both_normal_count = test_summary_df['Both_Normal'].sum()
            total_tests = len(test_summary_df)
            percentage_both_normal = (both_normal_count / total_tests) * 100
            print(f"Percentage of tests where both groups are normally distributed: "
                  f"{percentage_both_normal:.1f}% ({both_normal_count}/{total_tests})")
        else:
            print("No tests to calculate percentage for.")

if __name__ == "__main__":
    run_no_persona_significance()



