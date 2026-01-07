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
    """Load all persona JSONL files from a model directory.
    """
    persona_files = list(model_dir.glob("persona*.jsonl"))
    
    if not persona_files:
        return []
    
    records = []
    for file_path in persona_files:
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


def _extract_persona_id(record: dict) -> str:
    """Extract persona ID from record.
    """
    # Try 'id' first, then 'persona_id'
    persona_id = record.get("id") or record.get("persona_id")
    if persona_id is not None:
        return str(persona_id)
    return None


def _process_records_to_matrix_data(model_records: list, model_name: str, df_claims_meta: pd.DataFrame) -> list:
    """Process model records and extract matrix data.
    """
    matrix_records = []
    
    for record in model_records:
        claim_id = record.get("Claim ID")
        verdict_raw = record.get("verdict")
        persona_id = _extract_persona_id(record)
        
        if claim_id is None:
            print(f"Warning: No Claim ID found in record")
            continue
        
        if persona_id is None:
            print(f"Warning: No persona_id found for claim {claim_id}")
            continue
        
        if verdict_raw is None:
            print(f"Warning: No verdict found for claim {claim_id}, persona {persona_id}")
            continue
        
        verdict = verdict_raw.strip().lower()
        
        if verdict not in LABEL_MAPPING:
            print(f"Warning: Unknown verdict '{verdict}' for claim {claim_id}, persona {persona_id}")
            continue
        
        if claim_id not in df_claims_meta.index:
            print(f"Warning: Claim ID {claim_id} not found in metadata")
            continue
        
        matrix_records.append({
            "model_id": model_name,
            "persona_id": persona_id,
            "claim_id": claim_id,
            "label": LABEL_MAPPING[verdict],
        })
    
    return matrix_records


def _create_matrices(matrix_records: list, prompt_style: str, prompt: str) -> dict:
    """Create mean and variance matrices from matrix records.
    """
    if not matrix_records:
        return {}
    
    df_matrix_long = pd.DataFrame(matrix_records)
    matrices = {}
    
    for model_dir_name, display_name in MODEL_MAPPING.items():
        model_data_filtered = df_matrix_long[df_matrix_long["model_id"] == display_name]
        
        if model_data_filtered.empty:
            continue
        
        # Calculate mean and variance across runs
        df_agg = (
            model_data_filtered
            .groupby(["persona_id", "claim_id"], as_index=False)["label"]
            .agg(['mean', 'var'])
        )
        df_agg.columns = ['persona_id', 'claim_id', 'mean', 'var']
        df_agg['var'] = df_agg['var'].fillna(0.0)
        
        # Create pivot tables
        df_matrix_mean = df_agg[['persona_id', 'claim_id', 'mean']].pivot(
            index="persona_id", columns="claim_id", values="mean"
        ).sort_index(axis=0).sort_index(axis=1)
        
        df_matrix_var = df_agg[['persona_id', 'claim_id', 'var']].pivot(
            index="persona_id", columns="claim_id", values="var"
        ).sort_index(axis=0).sort_index(axis=1)
        
        matrices[display_name] = {'mean': df_matrix_mean, 'var': df_matrix_var}
    
    return matrices


def _save_matrices(matrices: dict, prompt_style: str, prompt: str):
    """Save matrices to CSV files.
    """
    for display_name, matrix_data in matrices.items():
        df_mean = matrix_data['mean']
        df_var = matrix_data['var']
        
        output_path_mean = Path(f"data/claim_matrices/{prompt_style}/personas_{prompt}/{display_name}_mean.csv")
        output_path_var = Path(f"data/claim_matrices/{prompt_style}/personas_{prompt}/{display_name}_variance.csv")
        output_path_mean.parent.mkdir(parents=True, exist_ok=True)
        
        df_mean.to_csv(output_path_mean)
        df_var.to_csv(output_path_var)
        
        print(f"Mean matrix saved to: {output_path_mean}")
        print(f"Variance matrix saved to: {output_path_var}")
        print(f"Matrix shape: {df_mean.shape}")
        print(f"Personas: {len(df_mean.index)}, Claims: {len(df_mean.columns)}")


def _aggregate_prompt_styles(matrices_by_prompt_and_model: dict, prompt: str):
    """Aggregate matrices across prompt styles for 'all' output.
    """
    if prompt not in matrices_by_prompt_and_model:
        return
    
    print(f"\n--- Creating aggregated 'all' matrices for prompt {prompt} ---")
    
    for display_name, matrices in sorted(matrices_by_prompt_and_model[prompt].items()):
        available_styles = [s for s in PROMPT_STYLES if s in matrices]
        
        if not available_styles:
            print(f"Warning: No data found for {display_name}, skipping 'all' matrix")
            continue
        
        # Collect data from all prompt styles
        combined_mean_data = []
        combined_var_data = []
        
        for prompt_style in available_styles:
            style_data = matrices[prompt_style]
            df_mean = style_data.get('mean', pd.DataFrame())
            df_var = style_data.get('var', pd.DataFrame())
            
            if not df_mean.empty:
                for persona in df_mean.index:
                    for claim in df_mean.columns:
                        mean_val = df_mean.loc[persona, claim]
                        var_val = df_var.loc[persona, claim] if not df_var.empty else 0.0
                        if pd.notna(mean_val):
                            combined_mean_data.append({
                                "persona_id": persona,
                                "claim_id": claim,
                                "value": mean_val,
                            })
                            combined_var_data.append({
                                "persona_id": persona,
                                "claim_id": claim,
                                "value": var_val if pd.notna(var_val) else 0.0,
                            })
        
        # Aggregate
        if combined_mean_data:
            df_mean_long = pd.DataFrame(combined_mean_data)
            df_var_long = pd.DataFrame(combined_var_data)
            
            if len(available_styles) > 1:
                df_mean_agg = df_mean_long.groupby(["persona_id", "claim_id"], as_index=False).agg({'value': 'mean'})
                df_var_agg = df_var_long.groupby(["persona_id", "claim_id"], as_index=False).agg({'value': 'mean'})
                df_all_mean = df_mean_agg.pivot(index="persona_id", columns="claim_id", values="value").sort_index(axis=0).sort_index(axis=1)
                df_all_var = df_var_agg.pivot(index="persona_id", columns="claim_id", values="value").sort_index(axis=0).sort_index(axis=1)
            else:
                df_all_mean = matrices[available_styles[0]]['mean']
                df_all_var = matrices[available_styles[0]]['var']
        else:
            df_all_mean = pd.DataFrame()
            df_all_var = pd.DataFrame()
        
        # Save aggregated matrices
        output_path_all_mean = Path(f"data/claim_matrices/all/personas_{prompt}/{display_name}_mean.csv")
        output_path_all_var = Path(f"data/claim_matrices/all/personas_{prompt}/{display_name}_variance.csv")
        output_path_all_mean.parent.mkdir(parents=True, exist_ok=True)
        df_all_mean.to_csv(output_path_all_mean)
        df_all_var.to_csv(output_path_all_var)
        print(f"Mean matrix saved to: {output_path_all_mean} (from {available_styles})")
        print(f"Variance matrix saved to: {output_path_all_var}")
        print(f"Matrix shape: {df_all_mean.shape}")
        print(f"Personas: {len(df_all_mean.index)}, Claims: {len(df_all_mean.columns)}")


def process_personas_api(prompt_styles=None):
    """Process personas API format data and create matrices.
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
    matrices_by_prompt_and_model = {}
    
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
            matrices = _create_matrices(all_matrix_records, prompt_style, prompt)
            
            if matrices:
                _save_matrices(matrices, prompt_style, prompt)
                
                # Store for aggregation
                if prompt not in matrices_by_prompt_and_model:
                    matrices_by_prompt_and_model[prompt] = {}
                for display_name, matrix_data in matrices.items():
                    if display_name not in matrices_by_prompt_and_model[prompt]:
                        matrices_by_prompt_and_model[prompt][display_name] = {}
                    matrices_by_prompt_and_model[prompt][display_name][prompt_style] = matrix_data
    
    # Create aggregated "all" matrices
    for prompt in PROMPTS:
        _aggregate_prompt_styles(matrices_by_prompt_and_model, prompt)


if __name__ == "__main__":
    process_personas_api()
