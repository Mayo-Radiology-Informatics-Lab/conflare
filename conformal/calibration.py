"""
Functions for building a calibration set to be used for conformal prediction.
"""

import os
import pickle
from typing import Dict, List
import random

from tqdm import tqdm

from utils.prompts import Q_GENERATION_PROMPT


class QuestionGeneration:
    "A class for generating questions based on chunks and saving / loading them to / from a pickle file."

    def __init__(
        self,
        docs: List[Dict[str, str]] | None = None,
        qa_pipeline: callable | None = None,
        num_questions: int | None = 100,
        topic_of_interest: str | None = None,
        q_generation_prompt: str = Q_GENERATION_PROMPT,
        path_to_pickle: str | None = None,
    ) -> None:
        """
        Args:
            docs (List[Dict[str, str]] | None, optional): A list of dictionaries representing documents. Defaults to None.
            qa_pipeline (callable | None, optional): A callable object representing the question-answering pipeline. Defaults to None.
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

    def generate_questions(self, save_to_disk: bool = True):
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
            chunk = doc["content"]
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

            if len(self.generate_questions) == self.num_questions:
                break

        if save_to_disk:
            self.save()

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
            "model": self.qa_pipeline.model,
            "topic_of_interest": self.topic_of_interest,
            "generated_questions": self.generated_questions,
        }

        with open(self.path_to_pickle, "wb") as f:
            pickle.dump(generated_questions_records, f)
        print("Saved to disk!")

    def load(self, path_to_pickle: str = None):
        """
        Load data from a pickle file and assign relevant values to class attributes.

        Parameters:
            path_to_pickle (str): The path to the pickle file to load.

        Raises:
            FileNotFoundError: If the specified file is not found.
        """
        if os.path.exists(path_to_pickle):
            with open(path_to_pickle, "rb") as f:
                generated_questions_records = pickle.load(f)

            self.topic_of_interest = generated_questions_records["topic_of_interest"]
            self.generated_questions = generated_questions_records["generated_questions"]
            self.num_questions = len(self.generate_questions)
        else:
            raise FileNotFoundError(f"File {path_to_pickle} not found.")

    def __getitem__(self, idx: int):
        if len(self.generated_questions) != 0:
            return self.generated_questions[idx]
        else:
            raise IndexError(
                "There are no generated questions. Please generate some questions first using the generate_questions() method or load the data from the previously saved pickle file using the load() method."
            )
