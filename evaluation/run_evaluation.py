"""
Orchestration entry-point for evaluation scripts.

Layouts (relative to repo root):

- ``evaluation/rq1_no_persona`` — RQ1 no-persona
- ``evaluation/rq2_pba_pbt`` — pooled PBA vs PBT / correlation artefacts
- ``personas/rq2_personas`` — RQ2 persona significance & boxplots
- ``personas/rq3_correlations`` — persona-facing correlation tables/plots
- ``evaluation/rq3_llm_size`` — LLM size scaling (Phi / Qwen)

Run from repo root, e.g. ``python evaluation/run_evaluation.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EVAL_ROOT = Path(__file__).resolve().parent
if str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))

from rq1_no_persona._01_run_pbt_significance import run_pbt_significance
from rq1_no_persona._02_build_pbt_small import build_pbt_small
from rq1_no_persona._03_build_pbt_big import build_pbt_big
from rq1_no_persona._04_build_claim_maker_regression import run_regression_analysis

from rq2_pba_pbt._01_run_data_preperation import run_data_preperation
from rq2_pba_pbt._02_build_correlation_matrix import main as build_correlation_matrix

from build_main_plot import build_main_plot

from rq3_llm_size.correlation import build_llm_size_correlation_plot


if __name__ == "__main__":
    # RQ1 (evaluation/rq1_no_persona)
    run_pbt_significance()
    build_pbt_small()
    build_pbt_big()
    run_regression_analysis()

    # RQ2
    run_data_preperation()
    build_correlation_matrix()

    # PBA & PBT Plot
    build_main_plot()

    # RQ3 LLM size (reads interim_results only for regression)
    build_llm_size_correlation_plot()

