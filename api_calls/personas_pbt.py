from openai import OpenAI
import argparse
import csv
import json
import random
import time
import importlib.util
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# --- mappings ---
map_education = {
    "Secondary Level Education": "No I don't have.",
    "Bachelor's or equivalent level": "Yes, I have an Undergraduate Degree.",
    "Master's or equivalent level": "Yes, I have a Graduate Degree.",
}

map_political_view = {
    "No Specific Political View": "I don't identify with either the Democrats or the Republicans.",
    "Democrat": "I identify as a Democrat.",
    "Republican": "I identify as a Republican.",
}

ethnicity_names = {
    "White": ["Olson", "Snyder", "Wagner", "Meyer", "Schmidt", "Ryan", "Hansen", "Hoffman", "Johnston", "Larson"],
    "African American": ["Smalls", "Jeanbaptiste", "Diallo", "Kamara", "Pierrelouis", "Gadson", "Jeanlouis", "Bah", "Desir", "Mensah"],
    "Asian": ["Nguyen", "Kim", "Patel", "Tran", "Chen", "Li", "Le", "Wang", "Yang", "Pham"],
    "Hispanic": ["Garcia", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Perez", "Sanchez", "Ramirez", "Torres"],
}

ALLOWED_VERDICTS = {"false", "mostly-false", "half-true", "mostly-true", "true"}

def load_prompt_module(prompt_style: str, prompt_name: str):
    """Load a prompt module from api_calls/prompts/{prompt_style}/{prompt_name}.py"""
    prompts_dir = Path(__file__).parent / "prompts" / prompt_style
    prompt_path = prompts_dir / f"{prompt_name}.py"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    spec = importlib.util.spec_from_file_location(prompt_name, prompt_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if not hasattr(module, 'TEMPLATE'):
        raise ValueError(f"Prompt module {prompt_name} does not have TEMPLATE attribute")
    
    return module.TEMPLATE


def make_user_prompt(persona, claim_text, claim_polview, prompt_type, prompt_style="simple"):
    """
    Generate user prompt using template from prompts directory.
    
    Args:
        persona: Persona dictionary with attributes
        claim_text: The claim to evaluate
        claim_polview: Political view of the claim (for prompt_type 1)
        prompt_type: 0 for prompt_personas_1, 1 for prompt_personas_2
        prompt_style: "simple" or "chain_of_thought"
    """
    income = persona["income"]
    age = persona["age"]
    sex = persona["sex"]
    political_view = persona["political_view"]
    ethnicity = persona["ethnicity"]
    education_per = persona["education"]

    # Generate name with ethnicity
    name_prefix = "Mr." if sex.lower() == "male" else "Ms."
    name = random.choice(ethnicity_names[ethnicity])

    # Map education and political view
    education = map_education.get(education_per, education_per)
    polview = map_political_view.get(political_view, political_view)
    
    # Load template based on prompt_type
    prompt_name = f"prompt_personas_{prompt_type + 1}"
    template = load_prompt_module(prompt_style, prompt_name)
    
    # Fill template placeholders
    pre = "<s>[INST]"
    post = "[/INST]"
    
    # Format template with available variables
    try:
        if "{claim_polview}" in template:
            prompt = template.format(
                pre=pre,
                post=post,
                name_prefix=name_prefix,
                name=name,
                age=age,
                sex=sex,
                ethnicity=ethnicity,
                income=income,
                education=education,
                polview=polview,
                claim_text=claim_text,
                claim_polview=claim_polview
            )
        else:
            prompt = template.format(
                pre=pre,
                post=post,
                name_prefix=name_prefix,
                name=name,
                age=age,
                sex=sex,
                ethnicity=ethnicity,
                income=income,
                education=education,
                polview=polview,
                claim_text=claim_text
            )
    except KeyError as e:
        raise ValueError(f"Template {prompt_name} has unsupported placeholder: {e}")
    
    return prompt


def create_message(user_prompt):
    return [{"role": "user", "content": user_prompt}]


def normalize_verdict(raw_verdict):
    if not isinstance(raw_verdict, str):
        return None
    verdict = raw_verdict.strip().lower()
    verdict = verdict.replace("_", "-").replace(" ", "-")
    return verdict if verdict in ALLOWED_VERDICTS else None


def get_existing_persona_indices(prompt_dir: Path) -> set:
    """Get set of persona indices that already have result files"""
    existing_indices = set()
    if not prompt_dir.exists():
        return existing_indices
    
    for file_path in prompt_dir.glob("persona_results*.jsonl"):
        # Extract persona index from filename like "persona_results91.jsonl"
        filename = file_path.stem  # "persona_results91"
        if filename.startswith("persona_results"):
            index_str = filename[14:]  # Remove "persona_results" prefix
            try:
                index = int(index_str)
                existing_indices.add(index)
            except ValueError:
                # Skip files that don't have numeric indices (old format with IDs)
                continue
    
    return existing_indices


def load_existing_claim_ids(path: Path) -> set:
    claim_ids = set()
    if not path.exists():
        return claim_ids

    valid_records = []
    invalid_count = 0

    with path.open("r", encoding="utf-8") as infile:
        for line_number, raw_line in enumerate(infile, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"{path.name} line {line_number}: dropping invalid JSON ({exc})")
                invalid_count += 1
                continue

            claim_id = data.get("Claim ID")
            verdict = normalize_verdict(data.get("verdict"))

            if not claim_id or verdict is None:
                print(f"{path.name} line {line_number}: dropping invalid record (Claim ID={claim_id})")
                invalid_count += 1
                continue

            data["verdict"] = verdict
            claim_ids.add(claim_id)
            valid_records.append(data)

    if invalid_count:
        with path.open("w", encoding="utf-8") as outfile:
            for record in valid_records:
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{path.name}: cleaned {invalid_count} invalid record(s); will retry those claims")

    return claim_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Generate fact-checking responses for personas")
    parser.add_argument("--prompt", choices=["prompt_1", "prompt_2"], required=True, help="Prompt variant to run")
    parser.add_argument("--prompt-style", choices=["simple", "chain_of_thought"], default="simple",
                        help="Prompt style: 'simple' or 'chain_of_thought' (default: simple)")
    parser.add_argument("--persona-start", type=int, default=0, help="Start at this position/index (0-based, excluding header)")
    parser.add_argument("--persona-end", type=int, default=None, help="End at this position/index (inclusive, 0-based)")
    parser.add_argument("--prompt-type", type=int, choices=[0, 1], default=None, help="Override prompt type mapping")
    parser.add_argument("--personas-path", default="data/personas_metadata.csv", help="Path to personas CSV")
    parser.add_argument("--claims-path", default="./data/claims_metadata.csv", help="Path to claims CSV")
    parser.add_argument("--model", default="deepseek/deepseek-chat-v3.1", help="Model identifier")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--wait-seconds", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()


    if "deepseek" in args.model:
        model_dir_name = "deepseek-v3.1"
    elif "grok-4-fast" in args.model:
        model_dir_name = "grok-4-fast"
    elif "grok-3" in args.model:
        model_dir_name = "grok-3"
    elif "gpt" in args.model:
        model_dir_name = "gpt-4.1"
    else:
        raise ValueError(f"Unsupported model '{args.model}'. Expected identifier containing deepseek, grok, or gpt.")
    
    
    base_dir = Path("data/api_outputs") / args.prompt_style / "api_prompting_jing"
    model_dir = base_dir / model_dir_name
    prompt_dir = model_dir / args.prompt
    prompt_dir.mkdir(parents=True, exist_ok=True)

    with open(args.personas_path, "r", newline="", encoding="utf-8") as file:
        personas_reader = csv.DictReader(file)
        personas = list(personas_reader)

    # Keep personas in original CSV order (no sorting)

    with open(args.claims_path, "r", newline="", encoding="utf-8") as claim_file:
        claims_reader = csv.DictReader(claim_file)
        claims = list(claims_reader)

    persona_start = args.persona_start
    persona_end = args.persona_end

    prompt_type_map = {"prompt_1": 0, "prompt_2": 1}
    prompt_type = args.prompt_type if args.prompt_type is not None else prompt_type_map.get(args.prompt, 0)

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

    # Get existing persona indices to skip
    existing_persona_indices = get_existing_persona_indices(prompt_dir)
    print(f"Found {len(existing_persona_indices)} existing persona result files")
    print(f"Starting from position {persona_start}" + (f" to {persona_end}" if persona_end is not None else " to end"))
    
    for persona_index, persona in enumerate(personas):
        # Skip based on position/index
        if persona_index < persona_start:
            continue
        if persona_end is not None and persona_index > persona_end:
            continue
        
        persona_id = persona.get("id")
        if persona_id is None:
            continue
        
        # Skip if persona already has a result file
        if persona_index in existing_persona_indices:
            print(f"Persona at index {persona_index} (ID: {persona_id}): already processed, skipping")
            continue

        output_path = prompt_dir / f"persona_results{persona_index}.jsonl"
        existing_claim_ids = load_existing_claim_ids(output_path)
        pending_claims = [claim for claim in claims if claim['claim_id'] not in existing_claim_ids]

        if not pending_claims:
            print(f"Persona {persona_id}: up to date")
            continue

        print(f"Persona {persona_id}: generating {len(pending_claims)} claims")

        total_claims = len(pending_claims)
        with output_path.open("a", encoding="utf-8") as outfile:
            for index, claim in enumerate(pending_claims, start=1):
                start_time = time.time()
                claim_text = claim["claim"]
                claim_polview = claim.get("party", "")
                print(f"Persona {persona_id} [{index}/{total_claims}] claim {claim['claim_id']}: start")
                user_prompt = make_user_prompt(persona, claim_text, claim_polview, prompt_type=prompt_type, prompt_style=args.prompt_style)
                messages = create_message(user_prompt)

                model_answer = None
                raw_answer = None
                for attempt in range(1, args.max_retries + 1):
                    try:
                        response = client.chat.completions.create(
                            model=args.model,
                            messages=messages,
                            stream=False,
                            temperature=args.temperature,
                            response_format={'type': 'json_object'},
                            extra_body={
                                "reasoning": {
                                    "enabled": False
                                }
                            },
                        )
                        raw_answer = response.choices[0].message.content
                        model_answer = json.loads(raw_answer)
                        verdict_value = normalize_verdict(model_answer.get("verdict"))
                        if verdict_value is None:
                            raise ValueError("response missing valid verdict")
                        reasoning_text = model_answer.get("reasoning")
                        if isinstance(reasoning_text, str):
                            model_answer["reasoning"] = reasoning_text.strip()
                        elif "reasoning" in model_answer:
                            # Remove non-string reasoning to avoid emitting invalid JSON later.
                            model_answer.pop("reasoning", None)
                        model_answer["verdict"] = verdict_value
                        break
                    except json.JSONDecodeError as e:
                        print(f"Persona {persona_id} claim {claim['claim_id']} retry {attempt}/{args.max_retries}: Invalid JSON {e}")
                        print(f"Raw answer: {raw_answer}")
                        time.sleep(args.wait_seconds)
                    except ValueError as e:
                        print(f"Persona {persona_id} claim {claim['claim_id']} retry {attempt}/{args.max_retries}: Invalid content {e}")
                        print(f"Raw answer: {raw_answer}")
                        time.sleep(args.wait_seconds)
                    except Exception as e:
                        print(f"Persona {persona_id} claim {claim['claim_id']} retry {attempt}/{args.max_retries}: API error {e}")
                        time.sleep(args.wait_seconds)

                if model_answer is None:
                    duration = time.time() - start_time
                    print(f"Persona {persona_id} [{index}/{total_claims}] claim {claim['claim_id']}: failed after retries ({duration:.2f}s), skipping")
                    continue

                dataset_verdict = claim.get('label')
                record = {
                    'age': int(persona.get('age', 0)),
                    'sex': persona.get('sex', ''),
                    'ethnicity': persona.get('ethnicity', ''),
                    'income': persona.get('income', ''),
                    'education': persona.get('education', ''),
                    'political views': persona.get('political_view', ''),
                    'name': persona.get('name', ''),
                    'ID': persona_index,
                    'id': persona_id,
                    'Claim ID': claim['claim_id'],
                    'verdict': model_answer.get("verdict"),
                    'reasoning': model_answer.get("reasoning"),
                    'dataset verdict': dataset_verdict,
                    'claim_case': claim_text,
                }

                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                duration = time.time() - start_time
                print(f"Persona {persona_id} [{index}/{total_claims}] claim {claim['claim_id']}: done in {duration:.2f}s")


if __name__ == "__main__":
    main()
