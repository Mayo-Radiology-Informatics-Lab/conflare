"""
Main script for RAG with conformal prediction.
"""

from typing import Dict, List, Tuple

import chromadb
from conflare.models.openai import OpenAIModelQA
from conflare.models.hf import HFModelQA
from conflare.augmented_retrieval.vector_db import get_vector_db
from conflare.augmented_retrieval.embed import TextEmbedding
from conflare.augmented_retrieval.rag import ConformalRetrievalQA
from conflare.conformal.calibration import create_calibration_records
from conflare.utils.loaders import load_dir
from conflare.utils.preprocess import chunk_docs
from conflare.utils.prompts import SYSTEM_PROMPT


def initialize_pipeline(
    document_dir: str = "./data/documents",
    db_dir: str = "./data/vector_db",
    db_name: str = "conflare",
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
    model: str = "gpt-4",
    temperature: float = 0.1,
    max_tokens: int = 2000,
    frequency_penalty: float = 1.1,
    quantize: bool = True,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    normalize_embeddings: bool = True,
    db_metadata: dict = {"hnsw:space": "cosine"},
) -> Tuple[List[Dict[str, str]], OpenAIModelQA, chromadb.Collection]:
    """
    Initializes a pipeline for processing documents. 

    Parameters:
        document_dir (str): Directory path where documents are stored. Default is "./data/documents".
        db_dir (str): Directory path where the vector database is stored. Default is "./data/vector_db".
        db_name (str): Name of the vector database. Default is "conflare".
        chunk_size (int): Size of each chunk when processing documents. Default is 1500.
        chunk_overlap (int): Amount of overlap between document chunks. Default is 200.
        model (str): Name of the model to use for question answering. Default is "gpt-4".
        temperature (float): Controls the randomness of the model. Default is 0.1.
        max_tokens (int): Maximum number of tokens to generate. Default is 2000.
        frequency_penalty (float): Adjusts the penalties for repeating tokens. Default is 1.1.
        quantize (bool): Whether to quantize the model, only used for HF models. Default is True.
        embedding_model (str): Name of the embedding model to use. Default is "sentence-transformers/all-MiniLM-L6-v2".
        batch_size (int): Number of samples per batch. Default is 32.
        normalize_embeddings (bool): Whether to normalize embeddings. Default is True.
        db_metadata (dict): Metadata for the vector database. Default is {"hnsw:space": "cosine"}.

    Returns:
        Tuple[List[Dict[str, str]], OpenAIModelQA, chromadb.Collection]: A tuple containing the processed documents, the question answering model pipeline, and the vector database.
    """
    

    docs = load_dir(document_dir)
    docs = chunk_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if "gpt" in model:
        qa_pipeline = OpenAIModelQA(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
        )
    else:
        qa_pipeline = HFModelQA(
            model=model,
            quantize=quantize,
            temperature=temperature,
            max_tokens=max_tokens,
            repetition_penalty=frequency_penalty,
        )
    embeddeing_fn = TextEmbedding(
        model_id=embedding_model,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )
    vector_db = get_vector_db(
        db_path=db_dir,
        db_name=db_name,
        embedding_fn=embeddeing_fn,
        docs=docs,
        db_metadata=db_metadata,
    )

    return docs, qa_pipeline, vector_db


if __name__ == "__main__":
    docs, qa_pipeline, vector_db = initialize_pipeline()
    calibration_records = create_calibration_records(
        docs=docs,
        qa_pipeline=qa_pipeline,
        vector_db=vector_db,
        size=100,
        topic_of_interest="deep learning",
        max_chunk_eval=100,
        save_root="./data/calibration_set",
    )
    conformal_rag = ConformalRetrievalQA(
        qa_pipeline=qa_pipeline,
        vector_db=vector_db,
        calibration_records=calibration_records,
        error_rate=0.05,
        verbose=True,
    )

    response, retrieved_docs = conformal_rag(
        "What types of regularization methods have been used in training of the deep models?"
    )
