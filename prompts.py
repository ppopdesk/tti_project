# Prompts
NO_REASONING_PROMPT = (
    "Question: {question}\nOptions:\n{options_text}\n\n"
    "You are a medical expert. "
    "Provide the letter corresponding to the correct final answer to this multiple choice question. "
    "Your output should only be the letter of your chosen output choice, nothing else."
)

SHORT_REASONING_PROMPT = (
    "Question: {question}\nOptions:\n{options}\n\n"
    "You are a medical expert. "
    "Answer the following multiple-choice medical question.\n"
    "First do a short chain of thought under '## Thinking' to help arrive at the answer.\n"
    "Keep it brief, relevant, and only 1-2 short lines.\n"
    "Then state your final answer inside <ANSWER> tags like: <ANSWER>A</ANSWER>.\n"
    "Only the letter of your chosen output choice, should go inside the <ANSWER> tags."
)

LONG_REASONING_PROMPT = (
    "Question: {question}\nOptions:\n{options}\n\n"
    "You are a medical expert. "
    "Think step by step through the following multiple choice question, "
    "Then state your final answer inside <ANSWER> tags like: <ANSWER>A</ANSWER>."
    "Don't repeat the question or repeat steps in your reasoning."
)

# Token limits
NO_REASONING_TOKEN_LIMIT = 1
SHORT_REASONING_TOKEN_LIMIT = 150

# Hyperparameters
LOG_DIFF_THRESHOLD = 9.0
ENTROPY_THRESHOLD = 0.5 

SHORT_COT_TEMPERATURE = 1.0
LONG_COT_TEMPERATURE = 0.1

SHORT_COT_K = 3