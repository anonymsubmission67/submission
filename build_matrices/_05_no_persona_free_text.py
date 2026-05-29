"""
Pipeline step **5/5** — no-persona free-text classified matrices.

Build no-persona free-text (classified) mean/variance matrices for compass and/or coordinates.

Label rows are keyed by statements-CSV ``claim_id`` (after stripping a trailing numeric
run index suffix such as ``..._42`` from the JSON ``id`` field). Per ``(model_id, claim_id)``
variant the script pools to one mean score, then folds ``claim_id`` variants onto numeric
bases (``42``, ``42_opp``, ``42_ref`` → column ``42``) with the same mean / between-variant
variance rule as ``_02_no_persona_pba.py``.

Single entrypoint for easy debugging (set breakpoints in ``process_no_persona_free_text``).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_bm_dir = Path(__file__).resolve().parent
if str(_bm_dir) not in sys.path:
    sys.path.insert(0, str(_bm_dir))

from _02_no_persona_pba import (
    collapse_to_bases,
    flip_score_free_text,
    members_by_numeric_base,
    pooled_item_mean_wide,
)
from freetext_model_map import (
    FREE_TEXT_MATRIX_REQUIRED_IDS,
    reindex_matrices_to_target_llms,
    resolve_free_text_matrix_row_id,
    target_llms,
)

from utils.claim_matrix_roots import FREE_TEXT_DIR

# JSON record id may end with ``_<digits>`` as a run copy index; strip before CSV lookup / folding.
_RUN_INDEX_SUFFIX = re.compile(r"_(\d+)$")


def _claim_id_from_record_id(rid: str) -> str:
    s = str(rid).strip()
    if not s:
        return s
    m = _RUN_INDEX_SUFFIX.search(s)
    if not m:
        return s
    prefix = s[: m.start()]
    return prefix if prefix else s


def process_no_persona_free_text(questionnaire: str | None = None) -> None:
    map_labels = {"disagree": 1, "neutral": 2, "agree": 3}
    qconf = {
        "compass": {
            "statements_csv": "data/political_compass_statements.csv",
            "classified_subdir": "compass",
            "output_suffix": "compass",
        },
        "coordinates": {
            "statements_csv": "data/political_coordinates_statements.csv",
            "classified_subdir": "coordinates",
            "output_suffix": "coordinates",
        },
    }

    for q in (["compass", "coordinates"] if questionnaire is None else [questionnaire]):
        if q not in qconf:
            raise ValueError(f"questionnaire must be {tuple(qconf)}, got {q!r}")
        cfg = qconf[q]

        data_dir = Path("data/server_outputs/free_text_classified") / cfg["classified_subdir"]
        if not data_dir.is_dir():
            raise FileNotFoundError(data_dir)

        stmt_raw = pd.read_csv(cfg["statements_csv"])
        stmt_raw["claim_id"] = stmt_raw["claim_id"].astype(str).str.strip()
        stmt_raw = stmt_raw[stmt_raw["claim_id"] != ""]
        stmt_df_for_members = stmt_raw[["claim_id"]].copy()
        meta_ix_set = set(stmt_df_for_members["claim_id"].tolist())
        agree = {str(r["claim_id"]): r["Agree"] for _, r in stmt_raw.iterrows()}

        # --- load one merged record list per matrix model_id (primary *.json + optional *_additional.json)
        primaries = sorted(
            p for p in data_dir.glob("*.json") if not p.name.endswith("_additional.json")
        )
        if not primaries:
            raise FileNotFoundError(f"no *.json files in {data_dir}")

        model_data: dict[str, list] = {}
        for path in primaries:
            stem = path.stem
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                chunk0 = payload
            elif isinstance(payload, dict) and isinstance(payload.get("results"), list):
                chunk0 = payload["results"]
            else:
                raise TypeError(
                    f"{path.name}: expected list or dict with 'results' list, got {type(payload).__name__}"
                )
            merged = list(chunk0)
            add_path = path.with_name(f"{stem}_additional.json")
            if add_path.exists():
                extra = json.loads(add_path.read_text(encoding="utf-8"))
                if isinstance(extra, list):
                    merged.extend(extra)
                elif isinstance(extra, dict) and isinstance(extra.get("results"), list):
                    merged.extend(extra["results"])
                else:
                    raise TypeError(
                        f"{add_path.name}: expected list or dict with 'results', got {type(extra).__name__}"
                    )
            if not merged:
                raise ValueError(f"{path.name}: merged records empty")
            model_id = resolve_free_text_matrix_row_id(stem)
            if model_id is None:
                print(
                    f"WARNING: skipping {path.name}: stem {stem!r} not in target_llms"
                )
                continue
            model_data.setdefault(model_id, []).extend(merged)

        need = [m for m in target_llms if m in FREE_TEXT_MATRIX_REQUIRED_IDS]
        missing = [m for m in need if not model_data.get(m)]
        if missing:
            raise ValueError(
                f"{q!r} {data_dir}: missing classified JSON for models {missing}; "
                f"have {sorted(model_data)}; require {need}."
            )

        # --- long frame: one row per (model, claim_id variant, single label token)
        rows: list[dict] = []
        for model_id, records in model_data.items():
            for ri, record in enumerate(records):
                if not isinstance(record, dict):
                    raise TypeError(f"{model_id}[{ri}] expected dict, got {type(record).__name__}")
                rid = record.get("id")
                if rid is None:
                    raise ValueError(f"{model_id}[{ri}] missing id")
                claim_id = _claim_id_from_record_id(str(rid))
                if claim_id not in meta_ix_set:
                    raise ValueError(
                        f"{model_id}[{ri}] resolved claim_id {claim_id!r} not in {cfg['statements_csv']} "
                        f"(record id={str(rid).strip()!r})"
                    )
                rlabels = record.get("runs_labels")
                if not rlabels:
                    raise ValueError(f"{model_id}[{ri}] missing/empty runs_labels")
                for lab in rlabels:
                    rows.append({"model_id": model_id, "claim_id": claim_id, "label": lab})

        df_long = pd.DataFrame(rows)
        if df_long.empty:
            raise ValueError("no label rows extracted")

        # --- numeric labels (1–3) + Left/Right flip on 3-point scale
        allowed_labels = {1, 2, 3}
        label_series = pd.Series(df_long["label"], dtype="string").str.strip().str.lower()
        mapped = label_series.map(map_labels).fillna(label_series)
        df = df_long.copy()
        df["label"] = mapped
        df.loc[~df["label"].isin(allowed_labels), "label"] = pd.NA
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        if not df["label"].notna().any():
            raise ValueError("all labels invalid after 1–3 filter")

        scores: list[float] = []
        for _, row in df.iterrows():
            lab = row["label"]
            if pd.isna(lab):
                scores.append(float("nan"))
                continue
            cid = str(row["claim_id"])
            scores.append(flip_score_free_text(float(lab), str(agree.get(cid, "Right"))))
        df["score"] = scores
        df = df[np.isfinite(df["score"])].copy()

        if df.empty:
            raise ValueError("no valid scores after flip / filter")

        mean_wide = pooled_item_mean_wide(df)
        memb = members_by_numeric_base(stmt_df_for_members)
        mean_mx, var_mx = collapse_to_bases(mean_wide, memb)
        if mean_mx.empty:
            raise ValueError("collapse_to_bases produced empty mean matrix")
        mean_mx, var_mx = reindex_matrices_to_target_llms(mean_mx, var_mx)
        mean_mx = mean_mx.sort_index(axis=0)
        var_mx = var_mx.sort_index(axis=0)

        FREE_TEXT_DIR.mkdir(parents=True, exist_ok=True)
        mp = FREE_TEXT_DIR / f"{cfg['output_suffix']}_mean.csv"
        vp = FREE_TEXT_DIR / f"{cfg['output_suffix']}_variance.csv"
        mean_mx.to_csv(mp)
        var_mx.to_csv(vp)
        print(f"{q}: wrote {mp} shape={mean_mx.shape}")


if __name__ == "__main__":
    process_no_persona_free_text(questionnaire=None)
