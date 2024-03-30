"""
Utility functions for loading Hugging Face models and interacting with them.
"""

from typing import Tuple
from torch import cuda, bfloat16
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
)


def load_hf_model_tokenizer(
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.1", quantize: bool = True
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model and tokenizer for language modeling.

    Args:
        model_id (str): The ID of the model to be loaded. Defaults to "mistralai/Mistral-7B-Instruct-v0.1".
        quantize (bool): Whether to quantize the model. Defaults to True.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: The loaded model and its corresponding tokenizer.
    """
    device = f"cuda: {cuda.current_device()}" if cuda.is_available() else "cpu"
    bnb_config = None
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16,
        )
    model_config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=model_config,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
    )
    model.eval()
    print(f"Model loaded on {device} successfully.")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer


def qa_with_hf_models(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    temperature: float = 0.1,
    max_tokens: int = 2000,
    repetition_penalty: float = 1.1,
) -> str:
    """
    Generates a response using a Hugging Face model for question answering.

    Args:
        prompt (str): The prompt for the question answering.
        model (AutoModelForCausalLM): The Hugging Face model for question answering.
        tokenizer (AutoTokenizer): The Hugging Face tokenizer for question answering.
        temperature (float, optional): The temperature for sampling. Defaults to 0.1.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 2000.
        repetition_penalty (float, optional): The repetition penalty to apply. Defaults to 1.1.

    Returns:
        str: The generated response.
    """

    generator = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        return_full_text=False,
        temperature=temperature,
        max_new_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
    )
    generator.tokenizer.pad_token_id = generator.model.config.eos_token_id
    response = generator(prompt, pad_token_id=tokenizer.eos_token_id)

    return response[0]["generated_text"].strip()
