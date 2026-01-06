TEMPLATE = """{pre}
Task: Evaluate the following claim stated by a {claim_polview} for factual accuracy.

Please think step by step and return your answer.
First: Think about the reasoning and describe it in three concise sentences.
Second: choose one of the following verdicts: ["false", "mostly-false", "half-true", "mostly-true", "true"].

Return ONLY a JSON object with the structure: {{
"reasoning": <reasoning>,
"verdict": <verdict>
}}

Claim: {claim_text}
{post}
"""