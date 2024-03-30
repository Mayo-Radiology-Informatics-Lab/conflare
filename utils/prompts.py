"""
Contains prompt templates for different tasks.
"""

VANILLA_QA_PROMPT = """
###
Instruction: Answer the question based on your knowledge

### QUESTION:
{question}

 """

VANILLA_QA_PROMPT_MISTRAL = """
### [INST]
Instruction: Answer the question based on your knowledge

### QUESTION:
{question}

[/INST]
 """

