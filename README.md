<img src="./media/conflare.png" alt="drawing" width="200"/>

# CONFLARE: CONFormal LArge language model REtrieval

This is the repo for the [CONFLARE paper](arxiv.com) giving an easy access to scripts to do RAG w/ Conformal guarantees.
These are 3 main tasks that this repo helps you with:

1. Loading the source documents (+ cleaning and chunking them)
2. Creating (or loading) a Calibration set
3. Retrieval Augmented Generation by applying conformal prediction

Example:

```python
# 1
docs, qa_pipeline, vector_db = initialize_pipeline(path_to_docs="./data/documents")

# 2
calibration_records = create_calibration_records(
    docs,
    size=100,
    topic_of_interest="Deep Learning",
    qa_pipeline=qa_pipeline,
    vector_db=vector_db,
)

# 3
conformal_rag = ConformalRetrievalQA(
    qa_pipeline=qa_pipeline,
    vector_db=vector_db,
    calibration_records=calibration_records,
    error_rate=0.05,
)
QUESTION = "What types of regularization methods have been used in training of the deep models?"
response, retrieved_docs = conformal_rag(QUESTION)
```

If you have run this script once before and saved the calibration records to disk, you can use the following to load the calibration records:

```python
q_evaluation = QuestionEvaluation.from_pickle(path_to_pickle)
calibration_records = q_evaluation.get_calibration_records()
```

![figure1](./media/conflare-pipeline.png)
![figure2](./media/RAG.jpg)
