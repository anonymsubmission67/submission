"""Free-text generations: each free-text template × each claim × each run, one JSON file per model."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

_API_CALLS_DIR = Path(__file__).resolve().parent
if str(_API_CALLS_DIR) not in sys.path:
    sys.path.insert(0, str(_API_CALLS_DIR))

from prompts.free_text_templates import get_free_text_templates  # noqa: E402


JSON_INSTRUCTION = (
    'Respond with a single JSON object using only the key "generated_text" '
    "(string). Put your full response in that string."
)


def model_slug(model_name: str) -> str:
    s = model_name.strip().replace("/", "_")
    s = re.sub(r"[^\w.\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "model"


def display_name_for_templates(model_id: str, override: str | None) -> str:
    if override:
        return override
    parts = model_id.split("/")
    return parts[-1] if parts else model_id


def make_user_prompt(instruction: str, claim_text: str) -> str:
    return (
        f"{instruction}\n\n"
        f"Statement:\n{claim_text}\n\n"
        f"{JSON_INSTRUCTION}"
    )


def create_message(user_prompt: str) -> list[dict]:
    return [{"role": "user", "content": user_prompt}]


def composite_id(claim_id: str, template_idx: int) -> str:
    return f"{claim_id}_{template_idx}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Free-text API runs: templates × claims × runs → one JSON per model",
    )
    p.add_argument("--questionnaire", choices=["compass", "coordinates"], required=True)
    p.add_argument("--model", default="deepseek/deepseek-chat-v3.1")
    p.add_argument(
        "--model-display-name",
        default=None,
        help="Name inserted in personalized templates (default: last segment of --model)",
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--wait-seconds", type=float, default=1.0)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--run-start", type=int, default=1)
    p.add_argument("--run-end", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--sleep-between-runs", type=float, default=0.0)
    return p.parse_args()


def load_output(path: Path, expected_model: str) -> tuple[dict, set[tuple[str, int]], int]:
    """
    Load JSON file or return empty structure.
    Returns (payload, keys_done, dropped_invalid).
    keys_done: set of (id, runs) for valid rows.
    """
    if not path.exists():
        return (
            {"model": expected_model, "results": []},
            set(),
            0,
        )
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"{path.name}: invalid JSON ({e}); starting fresh")
        return ({"model": expected_model, "results": []}, set(), 0)

    results = data.get("results")
    if not isinstance(results, list):
        print(f"{path.name}: missing or invalid 'results' list; starting fresh")
        return ({"model": expected_model, "results": []}, set(), 0)

    valid: list[dict] = []
    keys_done: set[tuple[str, int]] = set()
    dropped = 0
    for i, row in enumerate(results):
        if not isinstance(row, dict):
            dropped += 1
            continue
        rid = row.get("id")
        runs_val = row.get("runs")
        gtext = row.get("generated_text")
        m = row.get("model")
        if not isinstance(rid, str) or not isinstance(gtext, str):
            dropped += 1
            continue
        if not gtext.strip():
            dropped += 1
            continue
        if not isinstance(runs_val, int):
            try:
                runs_val = int(runs_val)
            except (TypeError, ValueError):
                dropped += 1
                continue
        if not isinstance(m, str) or not m.strip():
            m = expected_model
        valid.append(
            {
                "model": m,
                "id": rid,
                "runs": runs_val,
                "generated_text": gtext.strip(),
            }
        )
        keys_done.add((rid, runs_val))

    if dropped:
        print(f"{path.name}: dropped {dropped} invalid result row(s)")

    data["model"] = expected_model
    data["results"] = valid
    return data, keys_done, dropped


def save_output(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def main() -> None:
    args = parse_args()

    out_dir = Path("data/api_outputs/free_text") / args.questionnaire
    slug = model_slug(args.model)
    output_path = out_dir / f"{slug}.json"

    claims_path = Path("data") / f"political_{args.questionnaire}_statements.csv"
    with claims_path.open("r", newline="", encoding="utf-8") as f:
        claims = list(csv.DictReader(f))

    templates = get_free_text_templates(
        display_name_for_templates(args.model, args.model_display_name)
    )

    total_runs = max(1, args.runs)
    run_start = max(1, args.run_start)
    run_end = args.run_end if args.run_end is not None else total_runs
    run_end = min(run_end, total_runs)

    payload, keys_done, _ = load_output(output_path, args.model)
    results: list[dict] = payload["results"]

    pending: list[tuple[str, str, int, int, str]] = []
    for run_idx in range(run_start, run_end + 1):
        for claim in claims:
            claim_id = str(claim["claim_id"]).strip()
            claim_text = claim["claim"]
            for t_idx, instruction in enumerate(templates):
                cid = composite_id(claim_id, t_idx)
                if (cid, run_idx) in keys_done:
                    continue
                pending.append((claim_id, claim_text, t_idx, run_idx, instruction))

    if not pending:
        print("Nothing to do: all (id, runs) combinations already present.")
        save_output(output_path, payload)
        return

    print(f"Output: {output_path}")
    print(f"Pending API calls: {len(pending)}")

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

    last_run: int | None = None
    for j, (claim_id, claim_text, t_idx, run_idx, instruction) in enumerate(pending, start=1):
        if last_run is not None and run_idx != last_run and args.sleep_between_runs:
            time.sleep(args.sleep_between_runs)
        last_run = run_idx

        cid = composite_id(claim_id, t_idx)
        print(f"[{j}/{len(pending)}] id={cid} run={run_idx} claim={claim_id} template={t_idx}")

        user_prompt = make_user_prompt(instruction, claim_text)
        messages = create_message(user_prompt)

        generated: str | None = None
        raw_answer: str | None = None

        for attempt in range(1, args.max_retries + 1):
            try:
                extra_body: dict = {"reasoning": {"enabled": False}}
                if args.seed is not None:
                    extra_body["seed"] = args.seed

                response = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    stream=False,
                    temperature=args.temperature,
                    response_format={"type": "json_object"},
                    extra_body=extra_body,
                )
                raw_answer = response.choices[0].message.content
                parsed = json.loads(raw_answer)
                text = parsed.get("generated_text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError('missing non-empty generated_text')
                generated = text.strip()
                break
            except json.JSONDecodeError as e:
                print(f"  retry {attempt}/{args.max_retries}: invalid JSON {e}")
                if raw_answer is not None:
                    print(f"  raw: {raw_answer[:500]!r}...")
                time.sleep(args.wait_seconds)
            except ValueError as e:
                print(f"  retry {attempt}/{args.max_retries}: {e}")
                if raw_answer is not None:
                    print(f"  raw: {raw_answer[:500]!r}...")
                time.sleep(args.wait_seconds)
            except Exception as e:
                print(f"  retry {attempt}/{args.max_retries}: API error {e}")
                time.sleep(args.wait_seconds)

        if generated is None:
            print(f"  skip id={cid} run={run_idx} (failed after retries)")
            continue

        row = {
            "model": args.model,
            "id": cid,
            "runs": run_idx,
            "generated_text": generated,
        }
        results.append(row)
        keys_done.add((cid, run_idx))
        payload["model"] = args.model
        payload["results"] = results
        save_output(output_path, payload)

    print("Done.")


if __name__ == "__main__":
    main()
