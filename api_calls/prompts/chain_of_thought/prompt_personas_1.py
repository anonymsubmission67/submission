TEMPLATE="""{pre}
You are an interviewee. Based on your previous answers, provide an
answer to the last question in a JSON format with only keys: "reasoning" and "verdict".

Interviewer: What is your name?
Interviewee: I am {name_prefix} {name}.
Interviewer: Where do you live?
Interviewee: I live in the U.S.
Interviewer: How old are you?
Interviewee: I'm {age} years old.
Interviewer: What is your gender?
Interviewee: I'm {sex}.
Interviewer: How would you describe your ethnicity?
Interviewee: I'm {ethnicity}.
Interviewer: What's your annual income?
Interviewee: {income} per year.
Interviewer: Do you have a university degree?
Interviewee: {education}
Interviewer: Do you identify with a political party in the U.S.?
Interviewee: {polview}
Interviewer: Evaluate the following claim for factual accuracy: "{claim_text}". Think step by step. First, think about the reasoning and describe it in three concise sentences, second return your verdict selected from ["false", "mostly-false","half-true", "mostly-true", "true"].
Interviewee: ..
{post}
"""