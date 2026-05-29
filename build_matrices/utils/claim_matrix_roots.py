"""Shared paths for claim-level matrix outputs."""

from pathlib import Path

# File lives in ``build_matrices/utils/`` → repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]
CLAIM_MATRICES = REPO_ROOT / "data" / "claim_matrices"
FREE_TEXT_DIR = CLAIM_MATRICES / "free_text"


def ensure_claim_matrices_dirs() -> None:
    CLAIM_MATRICES.mkdir(parents=True, exist_ok=True)
    FREE_TEXT_DIR.mkdir(parents=True, exist_ok=True)
