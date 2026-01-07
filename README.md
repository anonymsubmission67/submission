# Political Bias and Truth Perception: Exploring Confirmation Bias in LLMs

Repository for the ACL submission "Political Bias and Truth Perception: Exploring Confirmation Bias in LLMs".

## Repository Structure

- `claim_preprocessing/`: Scripts for building and filtering the test dataset
- `api_calls/`: Scripts for making API calls to LLMs
- `build_matrices/`: Scripts for building data matrices from API outputs as basis for evaluation
- `evaluation/`: Statistical analysis and visualization scripts
- `data/`: Input data files

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the root directory and add your OpenRouter API key:

```
OPENROUTER_API_KEY=your_key_here
```

## Usage

### Running API Calls

Run experiments with different models and prompts:

```bash
python api_calls/no_persona_pbt.py --model x-ai/grok-4-fast --runs 20 --prompt prompt_1 --prompt-style chain_of_thought
```

Parameters:

- `--model`: Model identifier from OpenRouter.com
- `--prompt`: `prompt_1` (party-agnostic) or `prompt_2` (party-aware)
- `--prompt-style`: `simple` or `chain_of_thought`
- `--runs`: Number of runs (default: 1)

### API Outputs to Data Matrics

In build_matrices/utils.py, specify a short name for the models you are going to run in MODEL_MAPPING.
Run the full pipeline:

```bash
python build_matrices/build_matrices.py
```

### Running Evaluation

In evaluation/utils.py, specify the models to be evaluated in target_llms, and define the full name and a color in llm_info and llm_colors.

Run the full evaluation pipeline:

```bash
python evaluation/run_evaluation.py
```

This generates statistical tests, tables, and plots for all research questions. Individual evaluation scripts can also be run separately from their respective directories.

## Output

Results are saved to:

- `output/tables/`: LaTeX tables
- `output/images/`: PNG plots and visualizations
- `data/interim_results/`: Intermediate data files
