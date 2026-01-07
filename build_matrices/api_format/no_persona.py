import json
import sys
import pandas as pd
from pathlib import Path

# Add parent directory to path to import utils
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from utils import AGREEMENT_MAPPING, MODEL_MAPPING, PROMPT_STYLES, LABEL_MAPPING


PROMPTS = ["1", "2"]

def _load_model_records(model_dir: Path) -> list:
    """Load all baseline JSONL files from a model directory.
    """
    baseline_files = list(model_dir.glob("baseline*.jsonl"))
    
    if not baseline_files:
        return []
    
    records = []
    for file_path in baseline_files:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse JSON in {file_path}")
                        continue
    
    return records


def _process_records_to_matrix_data(model_records: list, model_name: str, df_claims_meta: pd.DataFrame) -> list:
    """Process model records and extract matrix data.
    """
    matrix_records = []
    
    for record in model_records:
        claim_id = record.get("Claim ID")
        verdict = record.get("verdict", "").strip().lower()
        
        if claim_id is None:
            print(f"Warning: No Claim ID found in record")
            continue
        
        if verdict not in LABEL_MAPPING:
            print(f"Warning: Unknown verdict '{verdict}' for claim {claim_id}")
            continue
        
        if claim_id not in df_claims_meta.index:
            print(f"Warning: Claim ID {claim_id} not found in metadata")
            continue
        
        matrix_records.append({
            "model_id": model_name,
            "claim_id": claim_id,
            "label": LABEL_MAPPING[verdict],
        })
    
    return matrix_records


def _create_matrices(matrix_records: list) -> tuple:
    """Create mean and variance matrices from matrix records.
    """
    if not matrix_records:
        return pd.DataFrame(), pd.DataFrame()
    
    df_matrix_long = pd.DataFrame(matrix_records)
    
    # Calculate mean and variance across runs
    df_agg = (
        df_matrix_long
        .groupby(["model_id", "claim_id"], as_index=False)["label"]
        .agg(['mean', 'var'])
    )
    df_agg.columns = ['model_id', 'claim_id', 'mean', 'var']
    df_agg['var'] = df_agg['var'].fillna(0.0)
    
    # Create pivot tables
    df_matrix_mean = df_agg[['model_id', 'claim_id', 'mean']].pivot(
        index="model_id", columns="claim_id", values="mean"
    ).sort_index(axis=0).sort_index(axis=1)
    
    df_matrix_var = df_agg[['model_id', 'claim_id', 'var']].pivot(
        index="model_id", columns="claim_id", values="var"
    ).sort_index(axis=0).sort_index(axis=1)
    
    return df_matrix_mean, df_matrix_var


def _save_matrices(df_mean: pd.DataFrame, df_var: pd.DataFrame, prompt_style: str, prompt: str):
    """Save matrices to CSV files.
    """
    output_path_mean = Path(f"data/claim_matrices/{prompt_style}/no_persona_api_{prompt}_mean.csv")
    output_path_var = Path(f"data/claim_matrices/{prompt_style}/no_persona_api_{prompt}_variance.csv")
    output_path_mean.parent.mkdir(parents=True, exist_ok=True)
    
    df_mean.to_csv(output_path_mean)
    df_var.to_csv(output_path_var)
    
    print(f"Mean matrix saved to: {output_path_mean}")
    print(f"Variance matrix saved to: {output_path_var}")
    print(f"Matrix shape: {df_mean.shape}")
    print(f"Models: {list(df_mean.index)}, Claims: {len(df_mean.columns)}")


def _aggregate_prompt_styles(matrices_by_prompt: dict, prompt: str):
    """Aggregate matrices across prompt styles for 'all' output.

    """
    if prompt not in matrices_by_prompt:
        return
    
    matrices = matrices_by_prompt[prompt]
    print(f"\n--- Creating aggregated 'all' matrix for prompt {prompt} ---")
    
    # Collect data from all prompt styles
    combined_mean_data = []
    combined_var_data = []
    
    for prompt_style in PROMPT_STYLES:
        style_data = matrices.get(prompt_style, {})
        df_mean = style_data.get('mean', pd.DataFrame())
        df_var = style_data.get('var', pd.DataFrame())
        
        if not df_mean.empty:
            for model in df_mean.index:
                for claim in df_mean.columns:
                    mean_val = df_mean.loc[model, claim]
                    var_val = df_var.loc[model, claim] if not df_var.empty else 0.0
                    if pd.notna(mean_val):
                        combined_mean_data.append({
                            "model_id": model,
                            "claim_id": claim,
                            "value": mean_val,
                        })
                        combined_var_data.append({
                            "model_id": model,
                            "claim_id": claim,
                            "value": var_val if pd.notna(var_val) else 0.0,
                        })
    
    # Aggregate
    if combined_mean_data:
        df_mean_long = pd.DataFrame(combined_mean_data)
        df_var_long = pd.DataFrame(combined_var_data)
        
        df_mean_agg = df_mean_long.groupby(["model_id", "claim_id"], as_index=False).agg({'value': 'mean'})
        df_var_agg = df_var_long.groupby(["model_id", "claim_id"], as_index=False).agg({'value': 'mean'})
        
        df_all_mean = df_mean_agg.pivot(index="model_id", columns="claim_id", values="value").sort_index(axis=0).sort_index(axis=1)
        df_all_var = df_var_agg.pivot(index="model_id", columns="claim_id", values="value").sort_index(axis=0).sort_index(axis=1)
    else:
        # Fallback to single style if available
        if not matrices.get("simple", {}).get('mean', pd.DataFrame()).empty:
            df_all_mean = matrices["simple"]['mean']
            df_all_var = matrices["simple"]['var']
        elif not matrices.get("chain_of_thought", {}).get('mean', pd.DataFrame()).empty:
            df_all_mean = matrices["chain_of_thought"]['mean']
            df_all_var = matrices["chain_of_thought"]['var']
        else:
            df_all_mean = pd.DataFrame()
            df_all_var = pd.DataFrame()
    
    _save_matrices(df_all_mean, df_all_var, "all", prompt)


def process_no_persona_api(prompt_styles=None):
    """Process no_persona API format data and create matrices.

    """
    if prompt_styles is None:
        base_path = Path("data/api_outputs")
        prompt_styles = []
        if (base_path / "simple").exists():
            prompt_styles.append("simple")
        if (base_path / "chain_of_thought").exists():
            prompt_styles.append("chain_of_thought")
        if not prompt_styles:
            prompt_styles = ["simple"]
    
    df_claims_meta = pd.read_csv("data/claims_metadata.csv").set_index("claim_id")
    matrices_by_prompt = {}
    
    for prompt_style in prompt_styles:
        print(f"\nProcessing prompt style: {prompt_style}")
        
        for prompt in PROMPTS:
            print(f"\nProcessing Prompt {prompt}...")
            
            # Load all model data
            model_data = {}
            for model_dir_name in MODEL_MAPPING.keys():
                display_name = MODEL_MAPPING[model_dir_name]
                model_dir = Path(f"data/api_outputs/{prompt_style}/api_prompting_jing/{model_dir_name}/prompt_{prompt}")
                
                if not model_dir.exists():
                    print(f"Warning: Directory not found: {model_dir}")
                    continue
                
                model_records = _load_model_records(model_dir)
                if model_records:
                    model_data[display_name] = model_records
                    print(f"Loaded {display_name}: {len(model_records)} records")
            
            # Process records to matrix data
            all_matrix_records = []
            for model_name, model_records in model_data.items():
                print(f"Processing {model_name}...")
                matrix_records = _process_records_to_matrix_data(model_records, model_name, df_claims_meta)
                all_matrix_records.extend(matrix_records)
            
            # Create matrices
            df_mean, df_var = _create_matrices(all_matrix_records)
            
            if not df_mean.empty:
                _save_matrices(df_mean, df_var, prompt_style, prompt)
                
                # Store for aggregation
                if prompt not in matrices_by_prompt:
                    matrices_by_prompt[prompt] = {}
                matrices_by_prompt[prompt][prompt_style] = {'mean': df_mean, 'var': df_var}
    
    # Create aggregated "all" matrices
    for prompt in PROMPTS:
        _aggregate_prompt_styles(matrices_by_prompt, prompt)


if __name__ == "__main__":
    process_no_persona_api()
