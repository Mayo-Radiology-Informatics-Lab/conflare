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

def qa_with_openai_models(
    prompt: str,
    model: str = "gpt-4",
    system_prompt: str = "",
    temperature: float = 0.1,
    max_tokens: int = 2000,
    frequency_penalty: float = 1.1,
) -> str:
    """
    Basic function for QA wih OpenAI models.

    Args:
        prompt (str): The prompt for the question-answering task.
        model (str, optional): The name of the OpenAI model to use. Defaults to "gpt-4".
        system_prompt (str, optional): The system prompt to include in the conversation. Defaults to "".
        temperature (float, optional): The temperature parameter for sampling responses. Defaults to 0.1.
        max_tokens (int, optional): The maximum number of tokens in the response. Defaults to 2000.
        frequency_penalty (float, optional): The frequency penalty parameter for sampling responses. Defaults to 1.1.

    Returns:
        str: The generated response.
    """

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ]
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty
    )

    return response.choices[0].message.content.strip()