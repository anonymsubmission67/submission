"""
Pipeline step **1/5** — pooled PolitiFact no-persona (prompts 1–3).

Single pooled PoliticalFact no-persona matrices (server JSON + API JSONL, all prompt styles).

Writes:
  ``data/claim_matrices/{1|2|3}_mean.csv``
  ``data/claim_matrices/{1|2|3}_variance.csv``
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from utils.claim_matrix_roots import CLAIM_MATRICES, REPO_ROOT

from freetext_model_map import (
    has_server_json,
    reindex_matrices_to_target_llms,
    require_target_llm_coverage,
    target_llms,
    API_MODEL_MAPPING
)

_MAP = {
    "pants": 0,
    "pants-fire": 0,
    "the claim is **false**.": 1,
    "false": 1,
    "true": 5,
    "mostly-false": 2,
    "mostly-true": 4,
    "half-true": 3,
    "pants-on-fire": 0,
}

API_MODEL_MAPPING = {
    "deepseek-v3.1": "deepseek",
    "gpt-4.1": "gpt4",
    "grok-4-fast": "grok",
    "grok-3": "grok_small",
    "grok": "grok",
    "grok-4.3": "grok",
}


def _verdict_blob_to_keyword(raw) -> str:
    """Flatten JSON ``verdict`` values (often str; sometimes list of options)."""

    if raw is None:
        return ""
    if isinstance(raw, list):
        for item in raw:
            s = _verdict_blob_to_keyword(item)
            if s:
                return s
        return ""
    text = str(raw).strip().lower()
    return text


def verdict_from_run_text(run_txt: str) -> str | None:
    text = str(run_txt)
    json_pattern = r"\{[^}]*\"verdict\"[^}]*\}"
    for json_match in re.findall(json_pattern, text):
        try:
            blob = json.loads(json_match)
            v = _verdict_blob_to_keyword(blob.get("verdict"))
            if v in _MAP:
                return v
        except json.JSONDecodeError:
            continue
    for pattern in (
        r'"verdict":\s*"([^"]+)"',
        r"verdict:\s*\"([^\"]+)\"",
        r"verdict:\s*([a-z-]+)",
    ):
        for m in re.findall(pattern, text, re.IGNORECASE):
            vv = str(m).strip().lower()
            if vv in _MAP:
                return vv
    return None


def _load_server_bundle(data_dir: Path, model_name: str) -> list[dict]:
    combined: list[dict] = []
    for name in (f"{model_name}.json", f"{model_name}_additional.json"):
        p = data_dir / name
        if p.exists():
            combined.extend(json.loads(p.read_text(encoding="utf-8")))
    return combined


def _model_has_api_pbt(model_id: str, prompt_tag: str, prompt_styles: list[str]) -> bool:
    for api_model_key, disp in API_MODEL_MAPPING.items():
        if disp != model_id:
            continue
        for prompt_style in prompt_styles:
            mdir = (
                REPO_ROOT
                / "data/api_outputs"
                / prompt_style
                / "pbt"
                / api_model_key
                / f"prompt_{prompt_tag}"
            )
            if mdir.is_dir() and any(mdir.glob("baseline*.jsonl")):
                return True
    return False


def collect_server_records(prompt_styles: list[str], prompt_tag: str) -> list[dict]:
    meta = pd.read_csv(REPO_ROOT / "data/claims_metadata.csv").set_index("claim_id")
    meta_ix = meta.index
    rows: list[dict] = []
    server_subdir = f"prompt_{prompt_tag}"
    for prompt_style in prompt_styles:
        data_dir = REPO_ROOT / "data/server_outputs" / prompt_style / server_subdir
        if not data_dir.is_dir():
            continue
        for model in target_llms:
            if not has_server_json(data_dir, model):
                continue
            for record in _load_server_bundle(data_dir, model):
                cid = record.get("id")
                if cid is None or cid not in meta_ix:
                    continue
                for run_txt in record.get("runs") or []:
                    v = verdict_from_run_text(run_txt)
                    if v is None:
                        continue
                    rows.append(
                        {"model_id": model, "claim_id": cid, "label": _MAP[v]},
                    )
    return rows


def collect_api_records(prompt_styles: list[str], prompt_tag: str) -> list[dict]:
    meta = pd.read_csv(REPO_ROOT / "data/claims_metadata.csv").set_index("claim_id")
    meta_ix = meta.index
    rows: list[dict] = []
    for prompt_style in prompt_styles:
        root = REPO_ROOT / "data/api_outputs" / prompt_style
        for api_model_key, disp in API_MODEL_MAPPING.items():
            if disp not in target_llms:
                continue
            mdir = root / "pbt" / api_model_key / f"prompt_{prompt_tag}"
            if not mdir.exists():
                continue
            for fp in mdir.glob("baseline*.jsonl"):
                for line in fp.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    cid = rec.get("Claim ID")
                    verdict = _verdict_blob_to_keyword(rec.get("verdict"))
                    if cid is None or verdict not in _MAP:
                        continue
                    if cid not in meta_ix:
                        continue
                    rows.append({"model_id": disp, "claim_id": cid, "label": _MAP[verdict]})
    return rows


def _detect_prompt_styles() -> list[str]:
    base_path = REPO_ROOT / "data/api_outputs"
    out: list[str] = []
    if (base_path / "simple").exists():
        out.append("simple")
    if (base_path / "chain_of_thought").exists():
        out.append("chain_of_thought")
    return out if out else ["simple", "chain_of_thought"]


def pooled_matrices(prompt_tag: str, prompt_styles: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    context = f"PBT prompt {prompt_tag}"
    server_subdir = f"prompt_{prompt_tag}"
    require_target_llm_coverage(
        context=context,
        repo_root=REPO_ROOT,
        prompt_styles=prompt_styles,
        server_subdir=server_subdir,
        api_available=lambda model: _model_has_api_pbt(model, prompt_tag, prompt_styles),
    )

    all_rows = collect_server_records(prompt_styles, prompt_tag) + collect_api_records(
        prompt_styles, prompt_tag
    )
    if not all_rows:
        raise RuntimeError(f"{context}: no valid rows after pooling server and API sources")

    df = pd.DataFrame(all_rows)
    allowed = {0, 1, 2, 3, 4, 5}
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].notna() & df["label"].isin(allowed)]

    agg = df.groupby(["model_id", "claim_id"], as_index=False)["label"].agg(["mean", "var"])
    agg.columns = ["model_id", "claim_id", "mean", "var"]
    agg["var"] = agg["var"].fillna(0.0)
    mean_mx = agg.pivot(index="model_id", columns="claim_id", values="mean").sort_index(
        axis=0
    ).sort_index(axis=1)
    var_mx = agg.pivot(index="model_id", columns="claim_id", values="var").sort_index(
        axis=0
    ).sort_index(axis=1)
    mean_mx, var_mx = reindex_matrices_to_target_llms(mean_mx, var_mx)
    return mean_mx, var_mx


def process_politifact_no_persona(prompt_styles: list[str] | None = None) -> None:
    styles = _detect_prompt_styles() if prompt_styles is None else list(prompt_styles)
    CLAIM_MATRICES.mkdir(parents=True, exist_ok=True)
    for tag in ["1", "2", "3"]:
        mean_mx, var_mx = pooled_matrices(tag, styles)
        mp = CLAIM_MATRICES / f"{tag}_mean.csv"
        vp = CLAIM_MATRICES / f"{tag}_variance.csv"
        mean_mx.to_csv(mp)
        var_mx.to_csv(vp)
        print(f"politifact {tag}: wrote {mp} {mean_mx.shape}")


if __name__ == "__main__":
    process_politifact_no_persona()
