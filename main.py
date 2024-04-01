from typing import Dict, List, Tuple

import chromadb
from models.openai import OpenAIModelQA
from augmented_retrieval.vector_db import get_vector_db
from augmented_retrieval.embed import TextEmbedding
from augmented_retrieval.rag import ConformalRetrievalQA
from conformal.calibration import QuestionGeneration, QuestionEvaluation, create_calibration_records
from utils.loaders import load_dir, initialize_pipeline
from utils.preprocess import chunk_docs

def initialize_pipeline(
    path_to_docs: str,
) -> Tuple[List[Dict[str, str]], OpenAIModelQA, chromadb.Collection]:
    """
    Loads the basic components necessary for RAG with conformal prediction.

    Args:
        path_to_docs (str): The path to the directory containing the documents to be loaded.

    Returns:
        Tuple[List[Document], OpenAIModelQA, VectorDatabase]: A tuple containing the loaded documents, the QA pipeline, and the vector database.
    """

    docs = load_dir(path_to_docs)
    docs = chunk_docs(docs)
    qa_pipeline = OpenAIModelQA("gpt-4")
    embeddeing_fn = TextEmbedding()
    vector_db = get_vector_db(
        db_path="./data/vector_db", name="conflare", embedding_fn=embeddeing_fn, docs=docs
    )

    return docs, qa_pipeline, vector_db

if __name__ == "__main__":
    docs, qa_pipeline, vector_db = initialize_pipeline(path_to_docs="./data/documents")
    calibration_records = create_calibration_records(
        docs,
        size=100,
        topic_of_interest="Deep Learning",
        qa_pipeline=qa_pipeline,
        vector_db=vector_db,
    )
    conformal_rag = ConformalRetrievalQA(
        qa_pipeline=qa_pipeline,
        vector_db=vector_db,
        calibration_records=calibration_records,
        error_rate=0.05,
    )

    QUESTION = "What types of regularization methods have been used in training of the deep models?"
    response, retrieved_docs = conformal_rag(QUESTION)