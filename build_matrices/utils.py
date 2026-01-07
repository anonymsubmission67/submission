AGREEMENT_MAPPING = {
    "strongly disagree": 1,
    "disagree": 2,
    "agree": 4,
    "strongly agree": 5,
}
LABEL_MAPPING = {
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

# Model name mapping
MODEL_MAPPING = {
    "deepseek-v3.1": "deepseek",
    "gpt-4.1": "gpt4",
    "grok-4-fast": "grok",
    "grok-3": "grok_small"
}

PROMPT_STYLES = ["simple", "chain_of_thought"]
