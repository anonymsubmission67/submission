"""
Enrich Political Compass statements with GPT labels / reformulations.

- Adds columns: topic_label_gpt, pol_label_gpt, pol_opposite_gpt, pol_reformulation_gpt
  on the original rows.
- Appends rows for opposite / paraphrase: claim_id ``<original_id>_opp`` and ``<original_id>_ref``.

Reads ``OPENAI_API_KEY`` from the environment. If ``python-dotenv`` is installed, values are also
loaded from ``.env`` at the repo root (see ``.env.example``; that file must not contain real secrets).

Run from repository root:

    python claim_preprocessing/prepare_questionnaire_data.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = REPO_ROOT / "data/political_coordinates_statements.csv"
OUTPUT_CSV = REPO_ROOT / "data/political_coordinates_statements_extended.csv"


def parse_topic(reply: str) -> object:
    """Return 'cultural' | 'economic' | NA."""
    s = reply.lower().strip()
    has_cultural = bool(re.search(r"\bcultural\b", s))
    has_economic = bool(re.search(r"\beconomic\b", s))
    if has_cultural and not has_economic:
        return "cultural"
    if has_economic and not has_cultural:
        return "economic"
    if has_cultural and has_economic:
        idx_c = min((m.start() for m in re.finditer(r"\bcultural\b", s)), default=10**9)
        idx_e = min((m.start() for m in re.finditer(r"\beconomic\b", s)), default=10**9)
        return "cultural" if idx_c <= idx_e else "economic"
    return pd.NA


def parse_politics(reply: str) -> object:
    """Return 'left' | 'right' | NA."""
    s = reply.lower().strip()
    has_left = bool(re.search(r"\bleft\b", s))
    has_right = bool(re.search(r"\bright\b", s))
    if has_left and not has_right:
        return "left"
    if has_right and not has_left:
        return "right"
    if has_left and has_right:
        idx_l = min((m.start() for m in re.finditer(r"\bleft\b", s)), default=10**9)
        idx_r = min((m.start() for m in re.finditer(r"\bright\b", s)), default=10**9)
        return "left" if idx_l <= idx_r else "right"
    return pd.NA


def get_response(
    client: OpenAI,
    prefix: str,
    statement: str,
    *,
    task: str | None,
    model: str,
) -> object:
    if not str(statement).strip():
        return pd.NA

    user_content = prefix.rstrip() + "\n\nStatement:\n" + str(statement).strip()

    rsp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Follow the instruction exactly."},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )
    raw = (rsp.choices[0].message.content or "").strip()

    if task == "topic":
        return parse_topic(raw)
    if task == "politics":
        return parse_politics(raw)
    return raw


def append_derived_claim_rows(df_originals: pd.DataFrame) -> pd.DataFrame:
    """Append rows whose claim comes from GPT opposite / reformulation columns."""
    extra: list[pd.Series] = []
    gpt_cols = [
        "topic_label_gpt",
        "pol_label_gpt",
        "pol_opposite_gpt",
        "pol_reformulation_gpt",
    ]

    def claim_id_suffix(row_claim_id: object) -> str:
        cid_raw = row_claim_id
        if pd.notna(cid_raw):
            try:
                return str(int(cid_raw))
            except (ValueError, TypeError):
                s = str(cid_raw).strip()
                return s.replace(" ", "_")
        return ""

    for _, row in df_originals.iterrows():
        sid = claim_id_suffix(row["claim_id"])
        if not sid or sid.endswith("_opp") or sid.endswith("_ref"):
            continue
        opp_txt = row.get("pol_opposite_gpt")
        ref_txt = row.get("pol_reformulation_gpt")

        if pd.notna(opp_txt) and str(opp_txt).strip():
            r = row.copy()
            r["claim_id"] = f"{sid}_opp"
            r["claim"] = str(opp_txt).strip()
            for c in gpt_cols:
                r[c] = pd.NA
            extra.append(r)
        if pd.notna(ref_txt) and str(ref_txt).strip():
            r = row.copy()
            r["claim_id"] = f"{sid}_ref"
            r["claim"] = str(ref_txt).strip()
            for c in gpt_cols:
                r[c] = pd.NA
            extra.append(r)

    if not extra:
        return df_originals
    return pd.concat([df_originals, pd.DataFrame(extra)], ignore_index=True)


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    model_name = os.environ.get("OPENAI_QUESTIONNAIRE_MODEL", "gpt-4")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Add it to .env at the repo root (see .env.example) "
            "or export it in your shell."
        )

    client = OpenAI(api_key=api_key)
    data = pd.read_csv(INPUT_CSV)

    tqdm.pandas(desc="gpt")

    prefixes = [
        (
            'Please indicate whether the following statement is about economic or cultural '
            'issues by returning only the word "economic" or "cultural".'
        ),
        (
            'Please indicate whether the following statement is attributable to the right or '
            'left side of the political spectrum by returning only the word "left" or "right".'
        ),
        (
            "The statement may lean politically. Reformulate it so it reflects the opposite "
            "side of the spectrum. Reply with only the reformulated statement, no preamble."
        ),
        (
            "Rephrase the statement so the political meaning stays similar but the wording "
            "changes substantially. Reply with only the reformulated statement, no preamble."
        ),
    ]

    tasks: list[str | None] = ["topic", "politics", None, None]

    colnames = [
        "topic_label_gpt",
        "pol_label_gpt",
        "pol_opposite_gpt",
        "pol_reformulation_gpt",
    ]

    for i, colname in enumerate(colnames):

        data[colname] = data["claim"].progress_apply(
            lambda claim,
            pref=prefixes[i],
            tk=tasks[i],
            m=model_name: get_response(
                client,
                pref,
                claim,
                task=tk,
                model=m,
            )
        )

    data = append_derived_claim_rows(data)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Wrote {OUTPUT_CSV} ({len(data)} rows).")


if __name__ == "__main__":
    main()
