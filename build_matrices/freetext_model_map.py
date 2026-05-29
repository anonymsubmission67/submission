"""
Free-text classified JSON stem → matrix row id helpers.

Separate from ``utils/`` (package path roots) because a top-level ``utils.py`` cannot
coexist with a ``utils/`` subdirectory.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

target_llms = [
    "phi",
    "phi_small",
    "phi_medium",
    "liberal",
    "conservative",
    "ministral3",
    "ministral8",
    "ministral14",
    "mixtral",
    "qwen",
    "qwen_big",
    "qwen_very_big",
    "llama3",
    "llama4",
    "gpt-oss",
    "gpt4",
    "deepseek_small",
    "deepseek",
    "grok",
    "llama3_big"
]

API_MODEL_MAPPING = {
    "deepseek-v3.1": "deepseek",
    "gpt-4.1": "gpt4",
    "grok-4-fast": "grok",
    "grok-3": "grok_small",
    "grok": "grok",
    "grok-4.3": "grok",
}


FREETEXT_STEM_TO_CANONICAL: dict[str, str] = {
    "deepseek-chat-v3.1": "deepseek",
    "deepseek_deepseek-chat-v3.1": "deepseek",
    "gpt-4.1": "gpt4",
    "openai_gpt-4.1": "gpt4",
    "grok-4.3": "grok",
    "grok-4-fast": "grok",
    "x-ai_grok-4.3": "grok",
    "x-ai_grok-4-fast": "grok",
}

FREE_TEXT_MATRIX_REQUIRED_IDS: frozenset[str] = frozenset(FREETEXT_STEM_TO_CANONICAL.values())


def has_server_json(data_dir: Path, model: str) -> bool:
    """True if ``model.json`` or ``model_additional.json`` exists under ``data_dir``."""
    return (data_dir / f"{model}.json").is_file() or (
        data_dir / f"{model}_additional.json"
    ).is_file()


def server_styles_for_model(
    repo_root: Path,
    prompt_styles: list[str],
    server_subdir: str,
    model: str,
) -> list[str]:
    """Prompt styles (simple / chain_of_thought) with server JSON for ``model``."""
    found: list[str] = []
    for prompt_style in prompt_styles:
        data_dir = repo_root / "data/server_outputs" / prompt_style / server_subdir
        if data_dir.is_dir() and has_server_json(data_dir, model):
            found.append(prompt_style)
    return found


def require_target_llm_coverage(
    *,
    context: str,
    repo_root: Path,
    prompt_styles: list[str],
    server_subdir: str,
    api_available: Callable[[str], bool],
) -> None:
    """
    Each ``target_llms`` id must have server JSON in at least one prompt style
    (simple or chain_of_thought) or API data for the same context.
    """
    missing: list[str] = []
    for model in target_llms:
        if server_styles_for_model(repo_root, prompt_styles, server_subdir, model):
            continue
        if api_available(model):
            continue
        missing.append(model)
    if missing:
        raise RuntimeError(
            f"{context}: target_llms without server (simple/cot) or API data: {missing}"
        )


def reindex_matrices_to_target_llms(
    mean_df,
    var_df,
):
    """Ensure every ``target_llms`` row exists in mean/variance matrices (NaN if no scores)."""
    import pandas as pd

    mean_out = mean_df.reindex(target_llms)
    var_out = var_df.reindex(target_llms)
    if not isinstance(mean_out, pd.DataFrame):
        mean_out = pd.DataFrame(index=target_llms)
    if not isinstance(var_out, pd.DataFrame):
        var_out = pd.DataFrame(index=target_llms)
    return mean_out, var_out


def resolve_free_text_matrix_row_id(stem: str) -> str | None:
    """
    Map classified JSON filename stem to matrix ``model_id``.

    Returns ``None`` when the stem is not part of ``target_llms`` (file is skipped).
    """
    if stem in FREETEXT_STEM_TO_CANONICAL:
        canonical = FREETEXT_STEM_TO_CANONICAL[stem]
        if canonical in target_llms:
            return canonical
        return None
    if stem in target_llms:
        return stem
    return None


def normalize_free_text_matrix_row_id(stem: str) -> str:
    """Strict wrapper around :func:`resolve_free_text_matrix_row_id` (raises if unknown)."""
    model_id = resolve_free_text_matrix_row_id(stem)
    if model_id is None:
        raise ValueError(f"Unknown free-text classified JSON stem: {stem!r}")
    return model_id
