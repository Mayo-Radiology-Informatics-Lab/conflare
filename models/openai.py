"""
Utility functions for interacting with OpenAI API.
"""

import os
import openai


def setup_openai_api(api_key: str | None = None) -> None:
    """
    A function to set up the OpenAI API by providing an API key,
    either through parameter or environment variable.

    Parameters:
    api_key (str | None): The API key to authenticate with OpenAI API. Defaults to None.

    Returns:
    None
    """

    if api_key is not None:
        os.environ["OPENAI_API_KEY"] = api_key
        openai.api_key = api_key
    elif os.getenv("OPENAI_API_KEY") is not None:
        openai.api_key = os.getenv("OPENAI_API_KEY")
    else:
        raise ValueError(
            "No OpenAI API key provided. Please either add it as env variable or put it in /configs/apis.py file"
        )


class OpenAIModelQA:
    "Creates an object for QA using an OpenAI model"

    def __init__(
        self,
        model: str = "gpt-4",
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2000,
        frequency_penalty: float = 1.1,
        api_key: str | None = None,
    ) -> None:
        """
        Initializes the class with the specified model, system prompt, temperature, max tokens, frequency penalty, and API key.
        
        Args:
            model (str): The GPT model to use for the generation.
            system_prompt (str): The prompt to provide to the GPT model.
            temperature (float): The temperature parameter for generation.
            max_tokens (int): The maximum number of tokens to generate.
            frequency_penalty (float): The frequency penalty parameter for generation.
            api_key (str | None): The API key for accessing the OpenAI API, or None if not provided.
        """
                
        setup_openai_api(api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        self.system_prompt = system_prompt
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def __call__(self, prompt: str) -> str:
        """
        A method that generates a response based on the given prompt.

        Parameters:
            prompt (str): The prompt for which a response needs to be generated.

        Returns:
            str: The response generated based on the prompt.
        """

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            frequency_penalty=self.frequency_penalty,
        )

        return response.choices[0].message.content.strip()


# def qa_with_openai_models(
#     prompt: str,
#     model: str = "gpt-4",
#     system_prompt: str = "",
#     temperature: float = 0.1,
#     max_tokens: int = 2000,
#     frequency_penalty: float = 1.1,
# ) -> str:
#     """
#     Basic function for QA wih OpenAI models.

#     Args:
#         prompt (str): The prompt for the question-answering task.
#         model (str, optional): The name of the OpenAI model to use. Defaults to "gpt-4".
#         system_prompt (str, optional): The system prompt to include in the conversation. Defaults to "".
#         temperature (float, optional): The temperature parameter for sampling responses. Defaults to 0.1.
#         max_tokens (int, optional): The maximum number of tokens in the response. Defaults to 2000.
#         frequency_penalty (float, optional): The frequency penalty parameter for sampling responses. Defaults to 1.1.

#     Returns:
#         str: The generated response.
#     """

#     messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
#     client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#     response = client.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=temperature,
#         max_tokens=max_tokens,
#         frequency_penalty=frequency_penalty,
#     )

#     return response.choices[0].message.content.strip()
