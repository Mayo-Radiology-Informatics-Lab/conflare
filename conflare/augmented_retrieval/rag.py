"""
Retrieval Augmented Generation (RAG) utilities.
"""

from typing import Callable, Dict, List
import chromadb
import numpy as np

from conflare.utils.prompts import RAG_QA_PROMPT


class SimpleRetrievalQA:
    "Creates an object to do RAG based QA using an HF or OpenAI model"

    def __init__(self, qa_pipeline: Callable, vector_db: chromadb.Collection, topk: int) -> None:
        """
        Args:
            qa_pipeline (Callable): The QA pipeline to be used; can be HF or OpenAI Callable pipeline.
            vector_db (chromadb.Collection): The collection of vectors for the database.
            topk (int): The number of documents to retrieve.
        """
        self.vector_db = vector_db
        self.qa_pipeline = qa_pipeline
        self.topk = topk

    def retrieve_docs(self, query: str, topk: int) -> List[str]:
        retrieved_docs = self.vector_db.query(query_texts=query, n_results=topk)["documents"][0]
        return retrieved_docs

    def __call__(self, question: str, retuen_retrieved_chunks: bool = True) -> str:
        """
        A function that takes a question, retrieves documents, generates context, creates a prompt,
        and gets a response from a QA pipeline. Returns the response and retrieved documents based on the flag.

        Parameters:
            question (str): The question to be answered.
            topk (int): Number of documents to retrieve (default is 3).
            retuen_retrieved_chunks (bool): Flag to indicate whether to return retrieved documents along with the response (default is True).

        Returns:
            str: The response generated by the QA pipeline.
            list: Retrieved documents based on the flag.
        """
        retrieved_docs = self.retrieve_docs(query=question, topk=self.topk)
        context = "\n\n".join(retrieved_docs)
        prompt = RAG_QA_PROMPT.format(context=context, question=question)
        response = self.qa_pipeline(prompt)
        if retuen_retrieved_chunks:
            return response, retrieved_docs
        else:
            return response


class ConformalRetrievalQA(SimpleRetrievalQA):
    "Creates an object to do RAG based QA using an HF or OpenAI model with conformal filtering"
    def __init__(
        self,
        qa_pipeline: Callable,
        vector_db: chromadb.Collection,
        calibration_records: List[Dict[str, str]],
        error_rate: float = 0.05,
        verbose: bool = False,
    ) -> None:
        self.calibration_records = calibration_records
        self.error_rate = error_rate
        self.verbose = verbose
        super().__init__(qa_pipeline, vector_db, topk=vector_db.count())

    def retrieve_docs(self, query: str, topk: int) -> List[str]:
        retrieved_docs = self.vector_db.query(query_texts=query, n_results=topk)
        return self.conformal_filter(retrieved_docs)

    def conformal_filter(self, retrieved_docs: Dict[str, list]):
        """
        Filters the retrieved documents based on a calibration records.

        Args:
            retrieved_docs (Dict[str, list]): A dictionary containing the retrieved documents.
                The dictionary should have two keys: "documents" and "distances".
                The value corresponding to the "documents" key is a list of document texts.
                The value corresponding to the "distances" key is a list of cosine distances.

        Returns:
            List[str]: A list of relevant chunks from the retrieved documents.
                The list contains the document texts that have a cosine distance less than the threshold.
        """
        cosine_distances = [record["cosine_distance"] for record in self.calibration_records]
        threshold = np.percentile(cosine_distances, 100 - self.error_rate * 100)
        doc_texts = retrieved_docs["documents"][0]
        doc_distances = retrieved_docs["distances"][0]
        relevant_docs = [
            doc for doc, distance in zip(doc_texts, doc_distances) if distance < threshold
        ]
        if self.verbose:
            print(f"Input Error Rate: {self.error_rate * 100:.2f}%")
            print(f"Selected cosine distance thereshold: {threshold:.3f}")
            print(f"Number of retrieved documents: {len(relevant_docs)}")

        return relevant_docs
