from openai import OpenAI
import argparse
import csv
import json
import time
import importlib.util
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
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


def make_user_prompt(claim_text, prompt_style="simple"):
    """
    Generate user prompt using template from prompts directory.
    
    Args:
        claim_text: The claim/statement to evaluate
        prompt_style: "simple" or "chain_of_thought"
    """
    prompt_name = "prompt_no_persona_compass"
    template = load_prompt_module(prompt_style, prompt_name)
    
    # Fill template placeholders
    pre = "<s>[INST]"
    post = "[/INST]"
    
    # Format template
    prompt = template.format(
        pre=pre,
        post=post,
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
    parser = argparse.ArgumentParser(description="Generate political compass responses (no persona baseline, repeated runs)")
    parser.add_argument("--prompt-style", choices=["simple", "chain_of_thought"], default="simple",
                        help="Prompt style: 'simple' or 'chain_of_thought' (default: simple)")
    parser.add_argument("--model", default="deepseek/deepseek-chat-v3.1", help="Model identifier")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--wait-seconds", type=float, default=1.0)
    parser.add_argument("--runs", type=int, default=20, help="How many repeated runs to perform")
    parser.add_argument("--run-start", type=int, default=1, help="First run index (1-based, inclusive)")
    parser.add_argument("--run-end", type=int, default=None, help="Last run index (1-based, inclusive)")
    # Optional: if your model supports seeds, you can provide one; otherwise ignored.
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
    prompt_dir = model_dir / "no_persona_compass"  # Fixed to prompt_1 since there's only one prompt
    prompt_dir.mkdir(parents=True, exist_ok=True)

    # Load political compass statements
    claims_path = "./data/political_compass_statements.csv"
    with open(claims_path, "r", newline="", encoding="utf-8") as claim_file:
        claims_reader = csv.DictReader(claim_file)
        claims = list(claims_reader)

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

    total_runs = args.runs
    run_start = max(1, args.run_start)
    run_end = args.run_end if args.run_end is not None else total_runs
    run_end = min(run_end, total_runs)

    for run_idx in range(run_start, run_end + 1):
        # One file per run: baseline_compassXX.jsonl
        pad = max(2, len(str(total_runs)))
        run_tag = f"{run_idx:0{pad}d}"
        output_path = prompt_dir / f"baseline_compass{run_tag}.jsonl"

        print(f"\n=== Run {run_idx}/{total_runs} -> {output_path.name} ===")
        existing_ids = load_existing_claim_ids(output_path)
        pending_claims = [c for c in claims if c['claim_id'] not in existing_ids]

        if not pending_claims:
            print(f"Run {run_idx}: up to date")
            if args.sleep_between_runs and run_idx < run_end:
                time.sleep(args.sleep_between_runs)
            continue

        print(f"Run {run_idx}: generating {len(pending_claims)} claim(s)")
        with output_path.open("a", encoding="utf-8") as outfile:
            for index, claim in enumerate(pending_claims, start=1):
                start_time = time.time()
                claim_text = claim["claim"]
                claim_polview = claim.get("party", "")
                claim_id = claim["claim_id"]
                print(f"[run {run_idx}] [{index}/{len(pending_claims)}] claim {claim_id}: start")

                user_prompt = make_user_prompt(claim_text, prompt_style=args.prompt_style)
                messages = create_message(user_prompt)

                model_answer = None
                raw_answer = None
                for attempt in range(1, args.max_retries + 1):
                    try:
                        extra_body = {"reasoning": {"enabled": False}}
                        if args.seed is not None:
                            # Not all providers honor this, but it's harmless if ignored.
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
                        print(f"[run {run_idx}] claim {claim_id} retry {attempt}/{args.max_retries}: Invalid JSON {e}")
                        print(f"Raw answer: {raw_answer}")
                        time.sleep(args.wait_seconds)
                    except ValueError as e:
                        print(f"[run {run_idx}] claim {claim_id} retry {attempt}/{args.max_retries}: Invalid content {e}")
                        print(f"Raw answer: {raw_answer}")
                        time.sleep(args.wait_seconds)
                    except Exception as e:
                        print(f"[run {run_idx}] claim {claim_id} retry {attempt}/{args.max_retries}: API error {e}")
                        time.sleep(args.wait_seconds)

                if model_answer is None:
                    duration = time.time() - start_time
                    print(f"[run {run_idx}] [{index}/{len(pending_claims)}] claim {claim_id}: failed after retries ({duration:.2f}s), skipping")
                    continue

                record = {
                    'run': run_idx,
                    'ID': -1,
                    'Claim ID': claim_id,
                    'agreement': model_answer.get("agreement"),
                    'reasoning': model_answer.get("reasoning"),
                    'statement': claim_text,
                    'domain': claim.get('Domain'),
                    'agree': claim.get('Agree'),
                }

                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                duration = time.time() - start_time
                print(f"[run {run_idx}] [{index}/{len(pending_claims)}] claim {claim_id}: done in {duration:.2f}s")

        if args.sleep_between_runs and run_idx < run_end:
            time.sleep(args.sleep_between_runs)


if __name__ == "__main__":
    main()
