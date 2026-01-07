"""
Data preparation for correlation analysis.
"""

import pandas as pd
import json
from sklearn.metrics import f1_score
import numpy as np
import sys
from pathlib import Path

# Add evaluation directory to path to allow importing utils
script_dir = Path(__file__).resolve().parent
evaluation_dir = script_dir.parent
if str(evaluation_dir) not in sys.path:
    sys.path.insert(0, str(evaluation_dir))

from utils import target_llms, opinion_llms, PERSONAS_PATH

# Constants
LABEL_TO_NUM = {
    "pants-fire": 0, "false": 1, "mostly-false": 2,
    "half-true": 3, "mostly-true": 4, "true": 5,
}
PROMPTS = ["1", "2"]
COMPASS_SCALE_OFFSET = 2.5
COMPASS_SCALE_DIVISOR = 1.5


def _map_compass_scale(scores: pd.Series) -> pd.Series:
    """
    Map compass scores from 0-4 scale to -1 to +1 scale.
    
    Original: 1=strongly disagree, 2=disagree, 3=agree, 4=strongly agree
    Mapped: -1=strongly disagree, -0.33=disagree, +0.33=agree, +1=strongly agree
    
    Args:
        scores: Series of compass scores (0-4 scale)
        
    Returns:
        Series of mapped scores (-1 to +1 scale)
    """
    return (scores - COMPASS_SCALE_OFFSET) / COMPASS_SCALE_DIVISOR


def _calculate_political_bias(compass_scores: pd.Series) -> float:
    """
    Calculate political bias as mean of mapped compass scores.
    
    Args:
        compass_scores: Series of compass scores
        
    Returns:
        Mean political bias value
    """
    if len(compass_scores) == 0:
        return 0.0
    mapped_scores = _map_compass_scale(compass_scores)
    return mapped_scores.mean()


def _calculate_me_diff_and_f1(df_long: pd.DataFrame, claims: pd.DataFrame, 
                               llm: str, prompt: str) -> tuple:
    """
    Calculate ME difference and F1 score for a given LLM and prompt.
    
    Args:
        df_long: DataFrame with LLM predictions
        claims: Claims metadata DataFrame
        llm: LLM identifier
        prompt: Prompt number ('1' or '2')
        
    Returns:
        Tuple of (me_diff, f1) or (None, None) if insufficient data
    """
    if llm not in df_long.index:
        return None, None
    
    # Get LLM predictions
    llm_predictions = df_long.loc[llm].dropna()
    
    # Get corresponding true labels
    df_long_with_labels = df_long.T.merge(
        claims[["label"]], left_index=True, right_index=True, how="left"
    )
    df_long_with_labels["label"] = df_long_with_labels["label"].map(LABEL_TO_NUM)
    
    true_labels = df_long_with_labels.loc[llm_predictions.index, "label"].dropna()
    
    # Align predictions and true labels
    common_indices = llm_predictions.index.intersection(true_labels.index)
    if len(common_indices) == 0:
        return None, None
    
    aligned_predictions = llm_predictions.loc[common_indices]
    aligned_true_labels = true_labels.loc[common_indices]
    
    # Calculate ME Difference (Republican - Democrat)
    df_long_with_party = df_long.T.merge(
        claims[["party", "label"]], left_index=True, right_index=True, how="left"
    )
    df_long_with_party["label"] = df_long_with_party["label"].map(LABEL_TO_NUM)
    df_long_with_party[llm] = df_long_with_party[llm] - df_long_with_party["label"]
    
    # Group by party and calculate mean bias
    bias_by_party = df_long_with_party.groupby("party")[llm].mean()
    me_diff = bias_by_party.get("Republican", 0) - bias_by_party.get("Democrat", 0)
    
    # Calculate F1 Score
    rounded_predictions = np.round(aligned_predictions).astype(int)
    rounded_predictions = np.clip(rounded_predictions, 0, 5)
    f1 = f1_score(aligned_true_labels, rounded_predictions, average='macro')
    
    return me_diff, f1


def _process_no_persona_data(llms: list, claims: pd.DataFrame, 
                             df_compass: pd.DataFrame) -> list:
    """
    Process no-persona data for all LLMs.
    
    Args:
        llms: List of LLM identifiers
        claims: Claims metadata DataFrame
        df_compass: Political compass data DataFrame
        
    Returns:
        List of result dictionaries
    """
    results = []
    
    for llm in llms:
        print(f"\nProcessing {llm}...")
        
        # Calculate political bias
        if llm in df_compass.index:
            all_scores = df_compass.loc[llm].dropna()
            pol_bias = _calculate_political_bias(all_scores)
        else:
            pol_bias = 0.0
        
        # Initialize result structure
        result = {
            "llm": llm,
            "persona": "no_persona",
            "pol_bias": round(pol_bias, 3),
            "prompt_1": {},
            "prompt_2": {}
        }
        
        # Process each prompt
        for prompt in PROMPTS:
            print(f"  Processing prompt {prompt}...")
            
            # Load data
            df_hpc = pd.read_csv(f"data/claim_matrices/all/no_persona_{prompt}.csv", index_col=0)
            df_api = pd.read_csv(f"data/claim_matrices/all/no_persona_api_{prompt}.csv", index_col=0)
            df_long = pd.concat([df_hpc, df_api], axis=0)
            
            # Calculate metrics
            me_diff, f1 = _calculate_me_diff_and_f1(df_long, claims, llm, prompt)
            
            if me_diff is not None and f1 is not None:
                result[f"prompt_{prompt}"] = {
                    "me_diff": round(me_diff, 3),
                    "f1": round(f1, 3)
                }
                print(f"    ME Diff: {me_diff:.3f}, F1: {f1:.3f}")
            else:
                print(f"    Warning: {llm} not found in prompt {prompt} data")
        
        results.append(result)
    
    return results


def _process_persona_data(llms: list, claims: pd.DataFrame, 
                          personas_metadata: pd.DataFrame) -> list:
    """
    Process persona data grouped by political view.
    
    Args:
        llms: List of LLM identifiers
        claims: Claims metadata DataFrame
        personas_metadata: Personas metadata DataFrame
        
    Returns:
        List of result dictionaries
    """
    results = []
    personas_metadata_indexed = personas_metadata.set_index("id")
    political_groups = personas_metadata.groupby("political_view")
    
    for llm in llms:
        print(f"\nProcessing personas for {llm}...")
        
        # Process each political group
        for political_view, group_personas in political_groups:
            print(f"  Processing {political_view} personas...")
            
            result = {
                "llm": llm,
                "persona": political_view.lower().replace(" ", "_"),
                "pol_bias": 0.0,
                "prompt_1": {},
                "prompt_2": {}
            }
            
            # Collect values for averaging
            pol_bias_values = []
            me_diff_values = {prompt: [] for prompt in PROMPTS}
            f1_values = {prompt: [] for prompt in PROMPTS}
            
            # Process each persona in this political group
            for _, persona_row in group_personas.iterrows():
                persona_id = persona_row["id"]
                
                # Only process personas that are in PERSONAS_PATH
                if persona_id not in personas_metadata_indexed.index:
                    continue
                
                # Calculate political bias from compass data
                try:
                    df_personas_compass = pd.read_csv(
                        f"data/claim_matrices/all/personas_compass/{llm}.csv", index_col=0
                    )
                    
                    if persona_id in df_personas_compass.index:
                        persona_compass_scores = df_personas_compass.loc[persona_id].dropna()
                        if len(persona_compass_scores) > 0:
                            pol_bias = _calculate_political_bias(persona_compass_scores)
                            pol_bias_values.append(pol_bias)
                except FileNotFoundError:
                    print(f"    Warning: personas_compass/{llm}.csv not found")
                    continue
                
                # Process each prompt
                for prompt in PROMPTS:
                    try:
                        df_personas = pd.read_csv(
                            f"data/claim_matrices/all/personas_{prompt}/{llm}.csv", index_col=0
                        )
                        
                        if persona_id in df_personas.index:
                            persona_predictions = df_personas.loc[persona_id].dropna()
                            
                            # Get corresponding true labels
                            df_personas_with_labels = df_personas.T.merge(
                                claims[["party", "label"]],
                                left_index=True, right_index=True, how="left"
                            )
                            df_personas_with_labels["label"] = (
                                df_personas_with_labels["label"].map(LABEL_TO_NUM)
                            )
                            
                            true_labels = df_personas_with_labels.loc[
                                persona_predictions.index, "label"
                            ].dropna()
                            
                            # Align predictions and true labels
                            common_indices = persona_predictions.index.intersection(true_labels.index)
                            if len(common_indices) > 0:
                                aligned_predictions = persona_predictions.loc[common_indices]
                                aligned_true_labels = true_labels.loc[common_indices]
                                
                                # Calculate bias
                                bias_values = aligned_predictions - aligned_true_labels
                                df_with_party = df_personas_with_labels.loc[common_indices]
                                bias_by_party = df_with_party.groupby("party")[persona_id].mean()
                                me_diff = bias_by_party.get("Republican", 0) - bias_by_party.get("Democrat", 0)
                                
                                # Calculate F1
                                rounded_predictions = np.round(aligned_predictions).astype(int)
                                rounded_predictions = np.clip(rounded_predictions, 0, 5)
                                f1 = f1_score(aligned_true_labels, rounded_predictions, average='macro')
                                
                                me_diff_values[prompt].append(me_diff)
                                f1_values[prompt].append(f1)
                    
                    except FileNotFoundError:
                        print(f"    Warning: personas_{prompt}/{llm}.csv not found")
                        continue
            
            # Calculate averages
            if pol_bias_values:
                result["pol_bias"] = round(sum(pol_bias_values) / len(pol_bias_values), 3)
            
            for prompt in PROMPTS:
                if me_diff_values[prompt] and f1_values[prompt]:
                    result[f"prompt_{prompt}"] = {
                        "me_diff": round(sum(me_diff_values[prompt]) / len(me_diff_values[prompt]), 3),
                        "f1": round(sum(f1_values[prompt]) / len(f1_values[prompt]), 3)
                    }
            
            results.append(result)
            print(f"    {political_view}: pol_bias={result['pol_bias']}, "
                  f"prompt_1_me_diff={result['prompt_1'].get('me_diff', 'N/A')}, "
                  f"prompt_2_me_diff={result['prompt_2'].get('me_diff', 'N/A')}")
    
    return results


def run_data_preperation():
    """
    Main function to prepare overview data for correlation analysis.
    
    Processes both no-persona and persona data, calculating political bias,
    ME differences, and F1 scores. Results are saved to JSON file.
    """
    llms = target_llms + opinion_llms
    
    # Load data
    claims = pd.read_csv("data/claims_metadata.csv").set_index("claim_id")
    
    # Load political compass data
    df_hpc = pd.read_csv("data/claim_matrices/all/no_persona_compass.csv", index_col=0)
    df_api = pd.read_csv("data/claim_matrices/all/no_persona_api_compass.csv", index_col=0)
    df_compass = pd.concat([df_hpc, df_api], axis=0)
    
    # Load personas metadata
    personas_metadata = pd.read_csv(PERSONAS_PATH)
    
    # Process data
    results = []
    results.extend(_process_no_persona_data(llms, claims, df_compass))
    results.extend(_process_persona_data(llms, claims, personas_metadata))
    
    # Save results
    output_path = Path("data/interim_results/overview_data.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Processed {len(results)} LLM-persona combinations")

    # Load political compass data
    df_hpc = pd.read_csv("data/claim_matrices/all/no_persona_compass.csv", index_col=0)
    df_api = pd.read_csv("data/claim_matrices/all/no_persona_api_compass.csv", index_col=0)
    df_compass = pd.concat([df_hpc, df_api], axis=0)

    # Load political compass statements metadata (for reference)
    df_statements_meta = pd.read_csv("data/political_compass_statements.csv")

    # Load personas metadata
    personas_metadata = pd.read_csv(PERSONAS_PATH)
    # Set index to 'id' for filtering
    personas_metadata_indexed = personas_metadata.set_index("id")

    results = []

    for llm in llms:
        print(f"\nProcessing {llm}...")
        
        # Calculate political bias (average of all statements)
        if llm in df_compass.index:
            # Get all statement scores for this LLM
            all_scores = df_compass.loc[llm].dropna()
            
            # Map from 0-4 scale to -1 to +1 scale
            # Original: 1=strongly disagree, 2=disagree, 3=agree, 4=strongly agree
            # Mapped: -1=strongly disagree, -0.33=disagree, +0.33=agree, +1=strongly agree
            mapped_scores = (all_scores - 2.5) / 1.5
            pol_bias = mapped_scores.mean()
        else:
            pol_bias = 0.0
        
        # Initialize result structure
        result = {
            "llm": llm,
            "persona": "no_persona",
            "pol_bias": round(pol_bias, 3),
            "prompt_1": {},
            "prompt_2": {}
        }
        
        # Process each prompt
        for prompt in ["1", "2"]:
            print(f"  Processing prompt {prompt}...")
            
            # Load no_persona data
            df_hpc = pd.read_csv(f"data/claim_matrices/all/no_persona_{prompt}.csv", index_col=0)
            df_api = pd.read_csv(f"data/claim_matrices/all/no_persona_api_{prompt}.csv", index_col=0)
            df_long = pd.concat([df_hpc, df_api], axis=0)

            if llm not in df_long.index:
                print(f"    Warning: {llm} not found in prompt {prompt} data")
                continue
            
            # Get LLM predictions
            llm_predictions = df_long.loc[llm].dropna()
            
            # Get corresponding true labels
            df_long_with_labels = df_long.T.merge(claims[["label"]], left_index=True, right_index=True, how="left")
            df_long_with_labels["label"] = df_long_with_labels["label"].map(label_to_num)
            
            # Get true labels for this LLM
            true_labels = df_long_with_labels.loc[llm_predictions.index, "label"].dropna()
            
            # Align predictions and true labels
            common_indices = llm_predictions.index.intersection(true_labels.index)
            if len(common_indices) == 0:
                print(f"    Warning: No common indices for {llm} in prompt {prompt}")
                continue
            
            aligned_predictions = llm_predictions.loc[common_indices]
            aligned_true_labels = true_labels.loc[common_indices]
            
            # Calculate ME Difference (Republican - Democrat)
            # First, get the bias by party
            df_long_with_party = df_long.T.merge(claims[["party", "label"]], left_index=True, right_index=True, how="left")
            df_long_with_party["label"] = df_long_with_party["label"].map(label_to_num)
            
            # Calculate bias (prediction - true_label) for each claim
            df_long_with_party[llm] = df_long_with_party[llm] - df_long_with_party["label"]
            
            # Group by party and calculate mean bias
            bias_by_party = df_long_with_party.groupby("party")[llm].mean()
            
            # Calculate ME difference (Republican - Democrat)
            me_diff = bias_by_party.get("Republican", 0) - bias_by_party.get("Democrat", 0)
            
            # Calculate F1 Score
            # Round predictions to nearest integer and map to classes
            rounded_predictions = np.round(aligned_predictions).astype(int)
            rounded_predictions = np.clip(rounded_predictions, 0, 5)  # Ensure valid range
            
            # Calculate F1 score (macro average)
            f1 = f1_score(aligned_true_labels, rounded_predictions, average='macro')
            
            # Store results
            result[f"prompt_{prompt}"] = {
                "me_diff": round(me_diff, 3),
                "f1": round(f1, 3)
            }
            
            print(f"    ME Diff: {me_diff:.3f}, F1: {f1:.3f}")
        
        results.append(result)

    # Process personas data
    print(f"\nProcessing personas data...")

    # Group personas by political_view
    political_groups = personas_metadata.groupby("political_view")

    for llm in llms:
        print(f"\nProcessing personas for {llm}...")
        
        # Process each political group
        for political_view, group_personas in political_groups:
            print(f"  Processing {political_view} personas...")
            
            # Initialize result structure for this political group
            result = {
                "llm": llm,
                "persona": political_view.lower().replace(" ", "_"),
                "pol_bias": 0.0,
                "prompt_1": {},
                "prompt_2": {}
            }
            
            # Calculate political bias for this group (average across personas)
            pol_bias_values = []
            me_diff_values_1 = []
            me_diff_values_2 = []
            f1_values_1 = []
            f1_values_2 = []
            
            # Process each persona in this political group
            for _, persona_row in group_personas.iterrows():
                persona_id = persona_row["id"]
                
                # Only process personas that are in PERSONAS_PATH
                if persona_id not in personas_metadata_indexed.index:
                    continue
                
                # Check if this persona exists in personas compass data
                try:
                    df_personas_compass = pd.read_csv(f"data/claim_matrices/all/personas_compass/{llm}.csv", index_col=0)
                    
                    if persona_id in df_personas_compass.index:
                        # Get all statement scores for this persona
                        persona_compass_scores = df_personas_compass.loc[persona_id].dropna()
                        
                        if len(persona_compass_scores) > 0:
                            # Map from 0-4 scale to -1 to +1 scale
                            mapped_scores = (persona_compass_scores - 2.5) / 1.5
                            pol_bias = mapped_scores.mean()
                            pol_bias_values.append(pol_bias)
                except FileNotFoundError:
                    print(f"    Warning: personas_compass/{llm}.csv not found")
                    continue
                
                # Process each prompt
                for prompt in ["1", "2"]:
                    try:
                        # Load personas data for this prompt
                        df_personas = pd.read_csv(f"data/claim_matrices/all/personas_{prompt}/{llm}.csv", index_col=0)
                        
                        if persona_id in df_personas.index:
                            # Get persona predictions
                            persona_predictions = df_personas.loc[persona_id].dropna()
                            
                            # Get corresponding true labels
                            df_personas_with_labels = df_personas.T.merge(claims[["party", "label"]], left_index=True, right_index=True, how="left")
                            df_personas_with_labels["label"] = df_personas_with_labels["label"].map(label_to_num)
                            
                            # Get true labels for this persona
                            true_labels = df_personas_with_labels.loc[persona_predictions.index, "label"].dropna()
                            
                            # Align predictions and true labels
                            common_indices = persona_predictions.index.intersection(true_labels.index)
                            if len(common_indices) > 0:
                                aligned_predictions = persona_predictions.loc[common_indices]
                                aligned_true_labels = true_labels.loc[common_indices]
                                
                                # Calculate bias (prediction - true_label) for each claim
                                bias_values = aligned_predictions - aligned_true_labels
                                
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
                                
                                # Store values for averaging
                                if prompt == "1":
                                    me_diff_values_1.append(me_diff)
                                    f1_values_1.append(f1)
                                else:
                                    me_diff_values_2.append(me_diff)
                                    f1_values_2.append(f1)
                    
                    except FileNotFoundError:
                        print(f"    Warning: personas_{prompt}/{llm}.csv not found")
                        continue
            
            # Calculate averages for this political group
            if pol_bias_values:
                result["pol_bias"] = round(sum(pol_bias_values) / len(pol_bias_values), 3)
            
            if me_diff_values_1:
                result["prompt_1"]["me_diff"] = round(sum(me_diff_values_1) / len(me_diff_values_1), 3)
                result["prompt_1"]["f1"] = round(sum(f1_values_1) / len(f1_values_1), 3)
            
            if me_diff_values_2:
                result["prompt_2"]["me_diff"] = round(sum(me_diff_values_2) / len(me_diff_values_2), 3)
                result["prompt_2"]["f1"] = round(sum(f1_values_2) / len(f1_values_2), 3)
            
            results.append(result)
            print(f"    {political_view}: pol_bias={result['pol_bias']}, prompt_1_me_diff={result['prompt_1'].get('me_diff', 'N/A')}, prompt_2_me_diff={result['prompt_2'].get('me_diff', 'N/A')}")

    # Save results to JSON
    with open("data/interim_results/overview_data.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: output/overview_data.json")
    print(f"Processed {len(results)} LLM-persona combinations")


if __name__ == "__main__":
    run_data_preperation()