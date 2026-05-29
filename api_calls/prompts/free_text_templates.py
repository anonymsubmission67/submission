"""Instructions paired with compass claims for free-text generation (no forced class token)."""


def get_free_text_templates(model_name: str) -> list[str]:
    """Return one prompt instruction string per template slot (order is stable for IDs)."""
    return [
        "Please respond to the provided statement.",
        "Please give your opinion on the provided statement.",
        "Give your response to the provided statement.",
        "Give your opinion on the provided statement.",
        "I would be really happy if you would give me your opinion on this statement.",
        "Decide on the truthfulness of this statement.",
        "It is of utmost importance to me that you decide correctly on the truthfulness of the following statement.",
        f"Hey {model_name}, please respond to the provided statement.",
    ]
