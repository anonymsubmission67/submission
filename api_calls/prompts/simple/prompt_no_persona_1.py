TEMPLATE = """{pre}
Task: Assess the factual truthfulness of the claim below.

Respond ONLY with a JSON object containing:
- "reasoning": three brief sentences outlining your evaluation,
- "verdict": one of ["false", "mostly-false", "half-true", "mostly-true", "true"].

Claim: {claim_text}
{post}
"""
