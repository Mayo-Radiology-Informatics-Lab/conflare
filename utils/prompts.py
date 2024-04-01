"""
Contains prompt templates for different tasks.
"""

VANILLA_QA_PROMPT = """
###
Instruction: Answer the question based on your knowledge

### QUESTION:
{question}

 """

#####
# To be used for MISTRAL model
#####

# VANILLA_QA_PROMPT = """
# ### [INST]
# Instruction: Answer the question based on your knowledge

# ### QUESTION:
# {question}

# [/INST]
#  """

# Note to add the specialized chars like [INST], [/INST] if using with a MISTRAL model
RAG_QA_PROMPT = """
###
Instruction: Answer the question based on the provided context. Only rely on
the context to build your answer and do not use your own knowledge:

{context}

### QUESTION:
{question}

"""