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

Q_GENERATION_PROMPT = """
###

-- INSTRUCTIONS --

You are given a CHUNK of text and a specified TOPIC_OF_INTEREST. Your job is to generate a question that has the following properties:

- The question MUST be DIRECTLY RELATED to the TOPIC_OF_INTEREST.
- The answer to the question MUST be clearly stated in the chunk.
- The question MUST not be easy to answer.
- You must not generate questions based on people names, article titles, and reference citations.

YOU MUST ONLY OUTPUT a ```JSON``` object with the key "question", and your generated question as the value for that key. If it is not possible to formulate an eligible question, return {{"question": "NONE"}}

-- EXAMPLES --

Examples of good and poor questions for an example chunk and topic of interest are provided below.

- Example topic of interest: deep learning

- Example chunk: 'Drawing boxes around the lesions, labelling of the lesions in all the slices, and training the models on the patches of detected lesions and manual labels. The time required to perform these manual operations is usually not considered when addressing the real-world application of these models and probably represents one of the major hurdles to widespread clinical adoption. A fully automatic tool running on chest CT images for the differential diagnosis of pneumonias can represent an important step forward for decreasing the variability of interpretation among clinicians and speeding up the diagnostic process. This will unburden medical staff and, in turn, provide better and faster diagnosis for patients, reducing the use of hospital resources. Better allocation of both material and human resources can be essential in a time of crisis, as the COVID-19 pandemic demonstrated with dramatic clarity [9]. To attain this goal, we developed and externally validated a fully automated deep-learning framework with a three-dimensional (3D) CNN, able to classify chest CT scans of patients with COVID-19, influenza/CAP, or no infection without manual intervention. Individual AI-based whole lung and lung abnormalities segmentation models were used to pre-process the CT images to train the 3D CNN model and are an integral part of the workflow to assure that only the patients presenting abnormalities in the lung volume are processed by the model, saving time and computational power. Material and methods'

- Example of a good question:

{{"question": "What aspects of preparing data for training deep learning models on CT scan data might be overlooked at the beginning of a deep learning project?"}}
This is a good question because the user does not need to see the chunk to understand the question, but still, the chunk provides a good example of how some requirements like annotation time could be overlooked at the beginning phase of a deep learning project.

- Example of poor questions:

{{"question": "What is the purpose of the fully automated deep-learning framework developed for chest CT scans in the context of pneumonia diagnosis?"}}
This is a poor question because a deep learning framework developed for chest CT scans in the context of pneumonia can serve many purposes, and it is not clear what framework the question is asking about.
{{"question": "What radiology modality is primarily used for diagnosing COVID 19?"}}
This is a poor question, becuase it is not relevant to the topic of interest (deep learning).
{{"question": "What is a deep learning training loop?}}
This is a poor question, because it is not based on the provided chunk!

-- TOPIC_OF_INTEREST --
{topic}

-- CHUNK --
{context}

"""