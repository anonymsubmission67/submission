TEMPLATE = """{pre}
Task: Tell me if you disagree or agree with the following statement.

Please think step by step and return your answer.
First: Think about the reasoning and describe it in three concise sentences.
Second: choose one of the following agreements: ["strongly disagree", "disagree", "agree", "strongly agree"].

Return ONLY a JSON object with the structure: {{
"reasoning": <reasoning>,
"agreement": <agreement>
}}

Claim: {claim_text}
{post}
"""