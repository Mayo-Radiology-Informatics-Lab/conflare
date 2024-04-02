"""
Functions for building a calibration set to be used for conformal prediction.
"""

import os
import pickle
from typing import Callable, Dict, List
import random
import chromadb
from tqdm import tqdm

from utils.prompts import Q_GENERATION_PROMPT, Q_EVAL_PROMPT


class QuestionGeneration:
    "A class for generating questions based on chunks and saving / loading them to / from a pickle file."

    def __init__(
        self,
        docs: List[Dict[str, str]] | None = None,
        qa_pipeline: Callable | None = None,
        num_questions: int | None = 100,
        topic_of_interest: str | None = None,
        q_generation_prompt: str = Q_GENERATION_PROMPT,
        path_to_pickle: str | None = None,
    ) -> None:
        """
        Args:
            docs (List[Dict[str, str]] | None, optional): A list of dictionaries representing documents. Defaults to None.
            qa_pipeline (Callable | None, optional): A Callable object representing the question-answering pipeline. Defaults to None.
            num_questions (int | None, optional): The number of questions to generate. Defaults to 100.
            topic_of_interest (str | None, optional): The topic of interest for the questions. Defaults to None.
            q_generation_prompt (str, optional): The prompt for generating questions. Defaults to Q_GENERATION_PROMPT.
            path_to_pickle (str | None, optional): The path to a pickle file. Defaults to None.
        """

        self.docs = docs
        self.qa_pipeline = qa_pipeline
        self.num_questions = num_questions
        self.topic_of_interest = topic_of_interest
        self.q_generation_prompt = q_generation_prompt
        self.path_to_pickle = path_to_pickle
        self.generated_questions = []
        self.error_logs = []

    def generate_questions(self, save_to_disk: bool = False) -> List[List]:
        """
        Generate questions based on the content of the documents (chunks) using the qa_pipeline.

        Parameters:
            save_to_disk (bool): Whether to save the generated questions to disk. Default is True.

        Returns:
            None
        """
        random.shuffle(self.docs)
        generation_pbar = tqdm(total=self.num_questions, desc="Generation Progress")
        for i, doc in enumerate(self.docs):
            chunk = doc["chunk"]
            prompt = self.q_generation_prompt.format(topic=self.topic_of_interest, context=chunk)
            response = self.qa_pipeline(prompt)

            try:
                g_question = eval(response)["question"]
            except:
                self.error_logs.append([i, doc, response])
                continue

            if g_question.lower() != "none":
                self.generated_questions.append([i, doc, g_question])
                generation_pbar.update(1)

            if len(self.generated_questions) == self.num_questions:
                break

        if save_to_disk:
            self.save()
        
        return self.generated_questions

    def save(self):
        """
        Saves the generated questions and related information to a pickle file.

        This function creates a dictionary called `generated_questions_records` that contains the following keys:
        - 'model': The model used for generating the questions.
        - 'topic_of_interest': The topic of interest for generating the questions.
        - 'generated_questions': The list of generated questions.

        The `generated_questions_records` dictionary is then serialized and saved to a pickle file specified by `self.path_to_pickle`.

        Returns:
            None
        """
        generated_questions_records = {
            "Model": self.qa_pipeline.model,
            "topic_of_interest": self.topic_of_interest,
            "generated_qs": self.generated_questions,
        }

        with open(self.path_to_pickle, "wb") as f:
            pickle.dump(generated_questions_records, f)
        print("Saved to disk!")

    @classmethod
    def from_pickle(cls, path_to_pickle: str = None):
        """
        Load data from a pickle file and return a class instance with the relevant values.

        Parameters:
            path_to_pickle (str): The path to the pickle file to load.

        Raises:
            FileNotFoundError: If the specified file is not found.
        """
        if os.path.exists(path_to_pickle):
            with open(path_to_pickle, "rb") as f:
                generated_questions_records = pickle.load(f)

            instance = cls(
                topic_of_interest=generated_questions_records["topic_of_interest"],
                num_questions=len(generated_questions_records["generated_qs"]),
            )
            instance.generated_questions = generated_questions_records["generated_qs"]
            return instance

        else:
            raise FileNotFoundError(f"File {path_to_pickle} not found.")

    def __getitem__(self, idx: int):
        if len(self.generated_questions) != 0:
            return self.generated_questions[idx][-1]
        else:
            raise IndexError(
                "There are no generated questions. Please generate some questions first using the generate_questions() method or load the data from the previously saved pickle file using the load() method."
            )

    def __len__(self):
        return len(self.generated_questions)


class QuestionEvaluation:
    def __init__(
        self,
        questions: List[list] | None = None,
        vector_db: chromadb.Collection | None = None,
        qa_pipeline: Callable | None = None,
        eval_prompt: str = Q_EVAL_PROMPT,
        max_chunk_eval: int = 100,
        path_to_pickle: str | None = None,
    ) -> None:
        """
        Args:
            questions (List[list]): A list of questions to be used for initialization.
            vector_db (chromadb.Collection): The vector database for the QA pipeline.
            qa_pipeline (Callable): The QA pipeline for evaluating the questions against chunks.
            eval_prompt (str, optional): The evaluation prompt. Defaults to Q_EVAL_PROMPT.
            max_chunk_eval (int, optional): The maximum number of retrieved chunks to evaluate. Defaults to 100.
            path_to_pickle (str, optional): The path to the pickle file to load or save.

        Returns:
            None
        """

        self.questions = questions
        self.vector_db = vector_db
        self.qa_pipeline = qa_pipeline
        self.eval_prompt = eval_prompt
        self.max_chunk_eval = max_chunk_eval
        self.path_to_pickle = path_to_pickle
        self.calibration_records = []

    def evaluate(self, save_to_disk: bool = False) -> List[Dict[str, str]]:
        for i, question_list in tqdm(enumerate(self.questions)):
            _, doc, question = question_list
            source_chunk = doc["chunk"]
            retrieved_chunks = self.vector_db.query(
                query_texts=question,
                n_results=self.vector_db.count(),
            )
            sorted_chunks = retrieved_chunks["documents"][0]
            sorted_scores = retrieved_chunks["distances"][0]
            source_chunk_index = sorted_chunks.index(source_chunk)

            relevant_chunk_found = False
            max_chunk_eval = min(self.max_chunk_eval, len(sorted_chunks))

            for j, (chunk_text, chunk_score) in enumerate(
                zip(sorted_chunks[:max_chunk_eval], sorted_scores[:max_chunk_eval])
            ):

                if chunk_text.lower() == source_chunk.lower():
                    relevant_chunk_found = True
                    break

                else:
                    response = self.qa_pipeline(
                        self.eval_prompt.format(question=question, context=chunk_text)
                    )
                    try:
                        decision = eval(response)["decision"]
                    except:
                        continue

                    if decision.lower() == "yes":
                        relevant_chunk_found = True
                        break

            if relevant_chunk_found:
                record = {
                    "generated_q": question_list,
                    "source_chunk_text": source_chunk,
                    "source_chunk_rank": source_chunk_index,
                    "relevant_chunk_text": chunk_text,
                    "relevant_chunk_rank": j,
                    "cosine_distance": chunk_score,
                }
                self.calibration_records.append(record)

        if save_to_disk:
            self.save()
        
        return self.calibration_records

    def save(self):
        with open(self.path_to_pickle, "wb") as f:
            pickle.dump(self.calibration_records, f)
        print("Saved to disk!")

    @classmethod
    def from_pickle(cls, path_to_pickle: str = None):
        if os.path.exists(path_to_pickle):
            with open(path_to_pickle, "rb") as f:
                calibration_records = pickle.load(f)

            instance = cls(questions=[record["generated_q"] for record in calibration_records])
            instance.calibration_records = calibration_records
            return instance

        else:
            raise FileNotFoundError(f"File {path_to_pickle} not found.")

    def get_calibration_records(self) -> List[Dict[str, str]]:
        return self.calibration_records

    def __getitem__(self, idx: int):
        if len(self.calibration_records) != 0:
            return self.calibration_records[idx]
        else:
            raise IndexError(
                "There are no calibration records. Please evaluate some questions first using the evaluate() method or load the data from the previously saved pickle file using the load() method."
            )

    def __len__(self):
        return len(self.calibration_records)


def create_calibration_records(
    docs: List[Dict[str, str]],
    size: int,
    topic_of_interest: str,
    qa_pipeline: Callable,
    vector_db: chromadb.Collection,
    save_root: str = "./data/calibration_set",
    save_to_disk: bool = False,
) -> List[Dict[str, str]]:
    """
    Create calibration records for a given set of documents.

    Args:
        docs (List[Dict[str, str]]): A list of documents to create calibration records for.
        size (int): The size of the final calibration set (calibration records).
        topic_of_interest (str): The topic of interest for down stream RAG QA.
        qa_pipeline (Callable): The question-answering pipeline to use for generating questions.
        vector_db (chromadb.Collection): The vector db to use for evaluating questions.
        save_root (str): The root directory where the generated questions and calibration records will be saved.
        save_to_disk (bool, optional): Whether to save the generated questions and calibration records to disk. Defaults to False.

    Returns:
        calibration_records: The generated calibration records.
    """
    q_generation = QuestionGeneration(
        docs=docs,
        qa_pipeline=qa_pipeline,
        num_questions=size,
        topic_of_interest=topic_of_interest,
        path_to_pickle=os.path.join(save_root, "Generated_Questions.pkl"),
    )
    generated_questions = q_generation.generate_questions(save_to_disk=save_to_disk)
    q_evaluation = QuestionEvaluation(
        questions=generated_questions,
        qa_pipeline=qa_pipeline,
        vector_db=vector_db,
        path_to_pickle=os.path.join(save_root, "Calibration_Records.pkl"),
    )
    calibration_records = q_evaluation.evaluate_questions(save_to_disk=save_to_disk)

    return calibration_records
