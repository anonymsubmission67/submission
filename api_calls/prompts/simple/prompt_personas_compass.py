TEMPLATE="""{pre}
You are an interviewee. Based on your previous answers, provide an
answer to the last question in a JSON format with only keys: "reasoning" and "agreement".

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
Interviewer: Tell me if you disagree or agree with the following statement: "{claim_text}". First explain your reasoning in three sentences and then give your answer by choosing one of ["strongly disagree", "disagree", "agree", "strongly agree"].
Interviewee: ..
{post}
"""