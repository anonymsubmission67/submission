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

ALLOWED_AGREEMENTS = {"strongly disagree", "disagree", "agree", "strongly agree"}


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


def make_user_prompt(persona, claim_text, prompt_style="simple"):
    """
    Generate user prompt using template from prompts directory.
    
    Args:
        persona: Persona dictionary with attributes
        claim_text: The claim/statement to evaluate
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
    
    # Load template
    prompt_name = "prompt_personas_compass"
    template = load_prompt_module(prompt_style, prompt_name)
    
    # Fill template placeholders
    pre = "<s>[INST]"
    post = "[/INST]"
    
    # Format template
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
    
    return prompt


def create_message(user_prompt):
    return [{"role": "user", "content": user_prompt}]


def normalize_agreement(raw_agreement):
    if not isinstance(raw_agreement, str):
        return None
    agreement = raw_agreement.strip().lower()
    return agreement if agreement in ALLOWED_AGREEMENTS else None


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
            agreement = normalize_agreement(data.get("agreement"))

            if not claim_id or agreement is None:
                print(f"{path.name} line {line_number}: dropping invalid record (Claim ID={claim_id})")
                invalid_count += 1
                continue

            data["agreement"] = agreement
            claim_ids.add(claim_id)
            valid_records.append(data)

    if invalid_count:
        with path.open("w", encoding="utf-8") as outfile:
            for record in valid_records:
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{path.name}: cleaned {invalid_count} invalid record(s); will retry those claims")

    return claim_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Generate political compass responses for personas")
    parser.add_argument("--prompt-style", choices=["simple", "chain_of_thought"], default="simple",
                        help="Prompt style: 'simple' or 'chain_of_thought' (default: simple)")
    parser.add_argument("--persona-start", type=int, default=0, help="First persona ID (inclusive)")
    parser.add_argument("--persona-end", type=int, default=None, help="Last persona ID (inclusive)")
    parser.add_argument("--personas-path", default="./data/personas_metadata.csv", help="Path to personas CSV")
    parser.add_argument("--statements-path", default="./data/political_compass_statements.csv", help="Path to statements CSV")
    parser.add_argument("--model", default="deepseek/deepseek-chat-v3.1", help="Model identifier")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--wait-seconds", type=float, default=1.0)
    parser.add_argument("--runs", type=int, default=20, help="How many repeated runs to perform")
    parser.add_argument("--run-start", type=int, default=1, help="First run index (1-based, inclusive)")
    parser.add_argument("--run-end", type=int, default=None, help="Last run index (1-based, inclusive)")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed to pass (model-dependent)")
    parser.add_argument("--sleep-between-runs", type=float, default=0.0, help="Pause between runs in seconds")
    return parser.parse_args()


def resolve_model_dir(model_name: str) -> str:
    if "deepseek" in model_name:
        return "deepseek-v3.1"
    if "grok-3" in model_name:
        return "grok-3"
    if "grok-4" in model_name:
        return "grok-4-fast"
    if "gpt" in model_name:
        return "gpt-4.1"
    raise ValueError(f"Unsupported model '{model_name}'. Expected identifier containing deepseek, grok, or gpt.")


def main():
    args = parse_args()

    base_dir = Path("data/api_outputs") / args.prompt_style / "api_prompting_compass"
    model_dir_name = resolve_model_dir(args.model)
    model_dir = base_dir / model_dir_name
    prompt_dir = model_dir / "personas_compass"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    # Load personas metadata
    import pandas as pd
    df_personas = pd.read_csv(args.personas_path)
    personas = df_personas.to_dict('records')

    # Load political compass statements
    with open(args.statements_path, "r", newline="", encoding="utf-8") as statements_file:
        statements_reader = csv.DictReader(statements_file)
        statements = list(statements_reader)

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

    total_runs = args.runs
    run_start = max(1, args.run_start)
    run_end = args.run_end if args.run_end is not None else total_runs
    run_end = min(run_end, total_runs)

    for run_idx in range(run_start, run_end + 1):
        # One file per run: persona_compass{run}.jsonl
        pad = max(2, len(str(total_runs)))
        run_tag = f"{run_idx:0{pad}d}"
        output_path = prompt_dir / f"persona_compass{run_tag}.jsonl"

        print(f"\n=== Run {run_idx}/{total_runs} -> {output_path.name} ===")
        existing_ids = load_existing_claim_ids(output_path)
        
        # Filter personas by ID range
        filtered_personas = []
        for persona_idx, persona in enumerate(personas):
            persona_id = persona.get("id")
            if persona_id is None:
                continue
            # Use index instead of persona_id for filtering since persona_id is a string hash
            if args.persona_start is not None and persona_idx < args.persona_start:
                continue
            if args.persona_end is not None and persona_idx > args.persona_end:
                continue
            filtered_personas.append(persona)

        if not filtered_personas:
            print(f"Run {run_idx}: no personas to process")
            continue

        print(f"Run {run_idx}: processing {len(filtered_personas)} personas with {len(statements)} statements each")

        # Process personas in batches of 10 to save progress regularly
        batch_size = 10
        persona_batches = [filtered_personas[i:i + batch_size] for i in range(0, len(filtered_personas), batch_size)]
        
        for batch_idx, persona_batch in enumerate(persona_batches, start=1):
            print(f"[run {run_idx}] Processing batch {batch_idx}/{len(persona_batches)} ({len(persona_batch)} personas)")
            
            with output_path.open("a", encoding="utf-8") as outfile:
                for persona_idx, persona in enumerate(persona_batch, start=1):
                    persona_id = persona.get("id")
                    global_persona_idx = (batch_idx - 1) * batch_size + persona_idx
                    print(f"[run {run_idx}] Persona {persona_id} ({global_persona_idx}/{len(filtered_personas)}): processing {len(statements)} statements")
                    
                    for statement_idx, statement in enumerate(statements, start=1):
                        start_time = time.time()
                        statement_text = statement["claim"]
                        statement_id = statement["claim_id"]
                        
                        # Check if this combination already exists
                        combo_id = f"{persona_id}_{statement_id}"
                        if combo_id in existing_ids:
                            continue
                        
                        print(f"[run {run_idx}] Persona {persona_id} [{statement_idx}/{len(statements)}] statement {statement_id}: start")
                        
                        user_prompt = make_user_prompt(persona, statement_text, prompt_style=args.prompt_style)
                        messages = create_message(user_prompt)

                        model_answer = None
                        raw_answer = None
                        for attempt in range(1, args.max_retries + 1):
                            try:
                                extra_body = {"reasoning": {"enabled": False}}
                                if args.seed is not None:
                                    extra_body["seed"] = args.seed

                                response = client.chat.completions.create(
                                    model=args.model,
                                    messages=messages,
                                    stream=False,
                                    temperature=args.temperature,
                                    response_format={'type': 'json_object'},
                                    extra_body=extra_body,
                                )
                                raw_answer = response.choices[0].message.content
                                model_answer = json.loads(raw_answer)

                                agreement_value = normalize_agreement(model_answer.get("agreement"))
                                if agreement_value is None:
                                    raise ValueError("response missing valid agreement")
                                model_answer["agreement"] = agreement_value

                                reasoning_text = model_answer.get("reasoning")
                                if isinstance(reasoning_text, str):
                                    model_answer["reasoning"] = reasoning_text.strip()
                                elif "reasoning" in model_answer:
                                    model_answer.pop("reasoning", None)

                                break
                            except json.JSONDecodeError as e:
                                print(f"[run {run_idx}] Persona {persona_id} statement {statement_id} retry {attempt}/{args.max_retries}: Invalid JSON {e}")
                                print(f"Raw answer: {raw_answer}")
                                time.sleep(args.wait_seconds)
                            except ValueError as e:
                                print(f"[run {run_idx}] Persona {persona_id} statement {statement_id} retry {attempt}/{args.max_retries}: Invalid content {e}")
                                print(f"Raw answer: {raw_answer}")
                                time.sleep(args.wait_seconds)
                            except Exception as e:
                                print(f"[run {run_idx}] Persona {persona_id} statement {statement_id} retry {attempt}/{args.max_retries}: API error {e}")
                                time.sleep(args.wait_seconds)

                        if model_answer is None:
                            duration = time.time() - start_time
                            print(f"[run {run_idx}] Persona {persona_id} [{statement_idx}/{len(statements)}] statement {statement_id}: failed after retries ({duration:.2f}s), skipping")
                            continue

                        record = {
                            'run': run_idx,
                            'persona_id': persona_id,
                            'Claim ID': statement_id,
                            'agreement': model_answer.get("agreement"),
                            'reasoning': model_answer.get("reasoning"),
                            'statement': statement_text,
                            'domain': statement.get('Domain'),
                            'agree': statement.get('Agree'),
                            **persona
                        }

                        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                        duration = time.time() - start_time
                        print(f"[run {run_idx}] Persona {persona_id} [{statement_idx}/{len(statements)}] statement {statement_id}: done in {duration:.2f}s")
            
            # Save progress after each batch
            print(f"[run {run_idx}] Batch {batch_idx} completed - progress saved to {output_path.name}")

        if args.sleep_between_runs and run_idx < run_end:
            time.sleep(args.sleep_between_runs)


if __name__ == "__main__":
    main()
