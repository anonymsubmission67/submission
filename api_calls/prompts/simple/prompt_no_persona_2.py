TEMPLATE="""{pre}
Task: Evaluate the following claim stated by a {claim_polview} for factual accuracy.

Return ONLY a JSON object with keys:
- "reasoning": three concise sentences explaining your reasoning,
- "verdict": one of ["false", "mostly-false","half-true", "mostly-true", "true"].

Claim: {claim_text}
{post}
"""

