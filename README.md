# Exploring Confirmation Bias: LLMs Perceive Democratic Claims as More Truthful Independent of their Content

Research codebase for studying confirmation bias in large language models using PolitiFact claims (PB-T) and political questionnaire agreement (PB-A).

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repository root:

```
OPENROUTER_API_KEY=your_key_here
```

Before evaluation or plots, set which models to include in [`config/models.py`](config/models.py) (`target_llms`, `llm_info`, `llm_colors`).

All commands below assume the **repository root** as the working directory.

## Data layout

| Path                               | Role                                                    |
| ---------------------------------- | ------------------------------------------------------- |
| `data/claims_metadata.csv`         | Main PolitiFact test set (committed)                    |
| `data/political_*_statements.csv`  | Questionnaire items (committed)                         |
| `data/claim_matrices/`             | Pooled model×claim matrices (committed for paper rerun) |
| `data/claim_preprocessing/`        | Raw scrape + intermediate CSVs (**not** in git)         |
| `data/api_outputs/`                | Your OpenRouter JSONL/JSON runs                         |
| `data/server_outputs/`             | Server-side model JSON (if used)                        |
| `data/interim_results/`            | Evaluation intermediates                                |
| `output/tables/`, `output/images/` | LaTeX tables and figures                                |

Legacy scrape path `data/claim_preporcessing/` is still resolved automatically if the correctly spelled folder is missing.

## Pipeline

### Step 0 — `claim_preprocessing/` (optional; needs raw scrape)

**Not runnable** without `data/claim_preprocessing/politifact_scraped.jsonl` (or the legacy typo directory).

Typical order:

1. `python claim_preprocessing/scraping_dump_to_table.py` — JSONL → `politifact_processed.csv`
2. `python claim_preprocessing/filter_claims.py` — filters and balancing (check script for which writes are enabled)
3. `python claim_preprocessing/prepare_questionnaire_data.py` — extended questionnaire labels

### Step 1 — `api_calls/` (OpenRouter)

**PB-T** (PolitiFact verdicts):

```bash
python api_calls/api_call_pbt.py \
  --model x-ai/grok-4-fast \
  --runs 20 \
  --prompt prompt_1 \
  --prompt-style chain_of_thought
```

**PB-A Likert** (compass or coordinates questionnaire):

```bash
python api_calls/api_call_pba_likert.py \
  --model x-ai/grok-4-fast \
  --runs 20 \
  --questionnaire compass \
  --prompt-style simple
```

**PB-A free text**:

```bash
python api_calls/api_call_pba_free_text.py \
  --model x-ai/grok-4-fast \
  --runs 20 \
  --questionnaire coordinates
```

Outputs go under `data/api_outputs/{simple|chain_of_thought}/...`. Prompt templates live in `api_calls/prompts/`.

### Step 2 — `build_matrices/`

Pools server JSON + API JSONL into matrices under `data/claim_matrices/`. Requires step 1 outputs (and `data/server_outputs/` if your setup uses them).

```bash
python build_matrices/run_build_matrices.py
```

Or run steps individually:

```bash
python build_matrices/_01_no_persona_pbt.py
python build_matrices/_02_no_persona_pba.py
python build_matrices/_05_no_persona_free_text.py
```

### Step 3 — `evaluation/`

```bash
python evaluation/run_evaluation.py
```

This runs significance tests, metric tables, correlation heatmap, and main figures. Individual scripts under `evaluation/rq1_no_persona/`, `evaluation/rq2_pba_pbt/`, and `evaluation/rq3_llm_size/` can be run separately for debugging.

Optional: `python evaluation/additional_analysis/class_balance_table.py`
