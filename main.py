"""
Main script for RAG with conformal prediction.
"""

from types import ModuleType
from typing import Dict, List, Tuple
import os

import chromadb
from src.models.openai import OpenAIModelQA
from src.models.hf import HFModelQA
from src.augmented_retrieval.vector_db import get_vector_db
from src.augmented_retrieval.embed import TextEmbedding
from src.augmented_retrieval.rag import ConformalRetrievalQA
from src.conformal.calibration import QuestionGeneration, QuestionEvaluation, create_calibration_records
from src.utils.loaders import load_dir, initialize_pipeline
from src.utils.preprocess import chunk_docs
from src.utils.prompts import SYSTEM_PROMPT

import src.configs as configs


def initialize_pipeline(
    configs: ModuleType,
) -> Tuple[List[Dict[str, str]], OpenAIModelQA, chromadb.Collection]:
    """
    Loads the basic components necessary for RAG with conformal prediction.

    Args:
        path_to_docs (str): The path to the directory containing the documents to be loaded.

    Returns:
        Tuple[List[Document], OpenAIModelQA, VectorDatabase]: A tuple containing the loaded documents, the QA pipeline, and the vector database.
    """

    docs = load_dir(configs.DOCUMENT_DIR)
    docs = chunk_docs(docs, chunk_size=configs.CHUNK_SIZE, chunk_overlap=configs.CHUNK_OVERLAP)
    if "gpt" in configs.MODEL:
        assert configs.OPENAI_API_KEY.startswith("sk-"), "Put a valid OpenAI API key in /configs/apis.py"
        os.environ["OPENAI_API_KEY"] = configs.OPENAI_API_KEY
        qa_pipeline = OpenAIModelQA(
            model=configs.MODEL,
            system_prompt=SYSTEM_PROMPT,
            temperature=configs.TEMPERATURE,
            max_tokens=configs.MAX_TOKENS,
            frequency_penalty=configs.FREQUENCY_PENALTY,
        )
    else:
        qa_pipeline = HFModelQA(
            model=configs.MODEL,
            quantize=configs.QUANTIZE,
            temperature=configs.TEMPERATURE,
            max_tokens=configs.MAX_TOKENS,
            repetition_penalty=configs.FREQUENCY_PENALTY,
        )
    embeddeing_fn = TextEmbedding(
        model_id=configs.EMBEDDING_MODEL,
        batch_size=configs.BATCH_SIZE,
        normalize_embeddings=configs.NORMALIZE_EMBEDDINGS,
    )
    vector_db = get_vector_db(
        db_path=configs.DB_DIR,
        db_name=configs.DB_NAME,
        embedding_fn=embeddeing_fn,
        docs=docs,
        db_metadata=configs.DB_METADATA,
    )

    return docs, qa_pipeline, vector_db


if __name__ == "__main__":
    docs, qa_pipeline, vector_db = initialize_pipeline(configs)
    calibration_records = create_calibration_records(
        docs=docs,
        qa_pipeline=qa_pipeline,
        vector_db=vector_db,
        size=configs.CALIBRATION_SIZE,
        topic_of_interest=configs.TOPIC,
        max_chunk_eval=configs.MAX_CHUNK_EVAL,
        save_root=configs.SAVE_DIR,
    )
    conformal_rag = ConformalRetrievalQA(
        qa_pipeline=qa_pipeline,
        vector_db=vector_db,
        calibration_records=calibration_records,
        error_rate=configs.ERROR_RATE,
        verbose=configs.VERBOSE,
    )

    response, retrieved_docs = conformal_rag(
        "What types of regularization methods have been used in training of the deep models?"
    )
