#!/usr/bin/env python3
"""
Script to calculate variance table for LLM consistency analysis.

Calculates:
- Variance for no_persona: variance across claims per LLM per prompt
- Average variance for personas: average variance across personas per LLM per prompt

Output: Table with LLMs in rows and 4 columns:
- Variance prompt_1 (no_persona)
- Variance prompt_2 (no_persona)
- Average variance across personas prompt_1
- Average variance across personas prompt_2
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add evaluation directory to path to allow importing utils
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from utils import target_llms, opinion_llms, llm_info, PROMPT_TYPE


def calculate_variance_table():
    """Calculate variance table for LLM consistency."""
    
    output_dir = Path("output/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all LLMs
    all_llms = target_llms + opinion_llms
    
    results = []
    
    print("Calculating variance table...")
    print(f"Using PROMPT_TYPE: {PROMPT_TYPE}")
    
    for llm in all_llms:
        print(f"\nProcessing {llm}...")
        
        result = {
            "llm": llm,
            "var_no_persona_p1": None,
            "var_no_persona_p2": None,
            "avg_var_personas_p1": None,
            "avg_var_personas_p2": None
        }
        
        # Calculate variance for no_persona (variance across claims)
        for prompt in ["1", "2"]:
            try:
                # Load HPC and API no_persona matrices from "all" directory (aggregated)
                hpc_path = Path(f"data/claim_matrices/all/no_persona_{prompt}.csv")
                api_path = Path(f"data/claim_matrices/all/no_persona_api_{prompt}.csv")
                
                df_hpc = pd.DataFrame()
                df_api = pd.DataFrame()
                
                if hpc_path.exists():
                    df_hpc = pd.read_csv(hpc_path, index_col=0)
                if api_path.exists():
                    df_api = pd.read_csv(api_path, index_col=0)
                
                # Combine HPC and API data
                if not df_hpc.empty and not df_api.empty:
                    df_no_persona = pd.concat([df_hpc, df_api], axis=0)
                elif not df_hpc.empty:
                    df_no_persona = df_hpc
                elif not df_api.empty:
                    df_no_persona = df_api
                else:
                    print(f"  Warning: No no_persona data found for prompt {prompt}")
                    continue
                
                # Check if LLM exists in the data
                if llm not in df_no_persona.index:
                    print(f"  Warning: {llm} not found in no_persona prompt {prompt} data")
                    continue
                
                # Get LLM row (responses across all claims)
                llm_responses = df_no_persona.loc[llm].dropna()
                
                if len(llm_responses) > 1:
                    # Calculate variance across claims
                    variance = llm_responses.var()
                    result[f"var_no_persona_p{prompt}"] = variance
                    print(f"  no_persona prompt {prompt}: variance = {variance:.4f} (n={len(llm_responses)} claims)")
                else:
                    print(f"  Warning: Insufficient data for {llm} in no_persona prompt {prompt} (n={len(llm_responses)})")
                    
            except Exception as e:
                print(f"  Error processing no_persona prompt {prompt} for {llm}: {e}")
                continue
        
        # Calculate average variance for personas (variance across personas, averaged per claim)
        for prompt in ["1", "2"]:
            try:
                # Load personas matrix for this LLM from "all" directory (aggregated)
                personas_path = Path(f"data/claim_matrices/all/personas_{prompt}/{llm}.csv")
                
                if not personas_path.exists():
                    print(f"  Warning: No personas data found for {llm} prompt {prompt}")
                    continue
                
                df_personas = pd.read_csv(personas_path, index_col=0)
                
                if df_personas.empty:
                    print(f"  Warning: Empty personas matrix for {llm} prompt {prompt}")
                    continue
                
                # Calculate variance across personas for each claim
                # Then average these variances
                claim_variances = []
                for claim in df_personas.columns:
                    claim_responses = df_personas[claim].dropna()
                    if len(claim_responses) > 1:
                        claim_var = claim_responses.var()
                        claim_variances.append(claim_var)
                
                if len(claim_variances) > 0:
                    avg_variance = np.mean(claim_variances)
                    result[f"avg_var_personas_p{prompt}"] = avg_variance
                    print(f"  personas prompt {prompt}: avg variance = {avg_variance:.4f} (n={len(claim_variances)} claims)")
                else:
                    print(f"  Warning: No valid variance calculations for {llm} personas prompt {prompt}")
                    
            except Exception as e:
                print(f"  Error processing personas prompt {prompt} for {llm}: {e}")
                continue
        
        results.append(result)
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Sort by target_llms order
    available_llms = df_results['llm'].tolist()
    sorted_llms = [llm for llm in target_llms if llm in available_llms]
    remaining_llms = [llm for llm in available_llms if llm not in target_llms]
    sorted_llms.extend(sorted(remaining_llms))
    
    # Reorder DataFrame
    df_results['llm_order'] = df_results['llm'].map({llm: i for i, llm in enumerate(sorted_llms)})
    df_results = df_results.sort_values('llm_order').drop(columns='llm_order')
    
    # Add LLM display names
    df_results['llm_name'] = df_results['llm'].apply(
        lambda x: llm_info.get(x, {}).get("name", x.capitalize())
    )
    
    # Reorder columns
    df_results = df_results[['llm', 'llm_name', 'var_no_persona_p1', 'var_no_persona_p2', 
                              'avg_var_personas_p1', 'avg_var_personas_p2']]
    
    # Save to CSV
    output_path = output_dir / "llm_variance_table.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\nVariance table saved to: {output_path}")
    
    # Print summary
    print("\nSummary:")
    print(df_results.to_string(index=False))
    
    return df_results


if __name__ == "__main__":
    calculate_variance_table()

