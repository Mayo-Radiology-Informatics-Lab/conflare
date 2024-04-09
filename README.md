<img src="https://i.ibb.co/vPyySmT/conflare-pipeline.png" alt="drawing" width="600"/>


# CONFLARE: CONFormal LArge language model REtrieval

This is the official repo for the [CONFLARE paper](https://arxiv.org/abs/2404.04287) and the related python package: `conflare`.

## Installation

```bash
pip install conflare
```

Here are the 3 main tasks this package helps you with:

1. Loading the source documents (+ cleaning and chunking them)
2. Creating (or loading) a Calibration set
3. Retrieval Augmented Generation by applying conformal prediction


## How to use
First, install the `conflare` package using `pip`. Then, use the following example as a starting point to use this package.

Example:

```python
# 1
import os
os.environ['OPENAI_API_KEY'] = 'your openai secret key'
# to use HuggingFace models w/o needing an openai key, look below.

import conflare
from conflare import initialize_pipeline
from conflare.conformal.calibration import create_calibration_records
from conflare.augmented_retrieval.rag import ConformalRetrievalQA

document_dir = './data/documents'
docs, qa_pipeline, vector_db = initialize_pipeline(document_dir)

# 2
calibration_records = create_calibration_records(
    docs,
    qa_pipeline=qa_pipeline,
    vector_db=vector_db,
    size=100,
    topic_of_interest="Deep Learning"
)

# 3
conformal_rag = ConformalRetrievalQA(
    qa_pipeline=qa_pipeline,
    vector_db=vector_db,
    calibration_records=calibration_records,
    error_rate=0.10,
    verbose=True
)

response, retrieved_docs = conformal_rag(
    "How can a transformer model be used in detection of COVID?"
)
print(response)
```
```
>>>
Input Error Rate: 10.00%
Selected cosine distance thereshold: 0.456
Number of retrieved documents: 2

A transformer model can be used in the detection of COVID-19 by analyzing medical images ...
```

If you have run this script once before and saved the calibration records to disk, you can use the following to load the calibration records. We've provided example `.pkl` files of generated questions and calibration recordings in the `./data/calibration_set/` directory of this repo.

```python
from conflare.conformal.calibration import QuestionEvaluation

q_evaluation = QuestionEvaluation.from_pickle(path_to_pickle)
calibration_records = q_evaluation.get_calibration_records()
```

## Arguments

Here are some of the more important arguments that the functions and classes in this package use.
You can also take a look at the definition of `initialize_pipeline` function to see most of them.
Looking at the definition of `initialize_pipeline`, you can see the sequence of the functions called inside it and use them in your own custom way if neccessary.


`model`: the model name used for QA and retreivals. If set to `gpt-*` models, it will use the OpenAI models and an OpenAI API Key will be required. It can also be set to models names on HuggingFace like `mistralai/Mistral-7B-Instruct-v0.1` to use HF models w/o needing a key. 

<br>

`embedding_model`: the model from `sentence-transformers` library to be used to create embeddings for text chunks and user questions.

## Citation

If you use this code in your research, please cite the following paper:

```
@article{conflare,
  title={CONFLARE: CONFormal LArge language model REtrieval},
  author={Pouria Rouzrokh and Shahriar Faghani and Cooper U. Gamble and Moein Shariatnia and Bradley J. Erickson},
  journal={arXiv preprint arXiv:2404.04287},
  year={2024},
  eprint={2404.04287},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```