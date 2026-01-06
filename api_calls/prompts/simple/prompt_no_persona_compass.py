TEMPLATE="""{pre}
Task: Tell me if you disagree or agree with the following statement.

Return ONLY a JSON object with keys:
- "reasoning": three concise sentences explaining your reasoning,
- "agreement": one of ["strongly disagree", "disagree", "agree", "strongly agree"].

Statement: {claim_text}
{post}
"""