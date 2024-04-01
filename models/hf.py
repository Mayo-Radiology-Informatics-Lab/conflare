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


class HFModelQA:
    "Creates an object for QA using an HF model"
    def __init__(
        self,
        model: str = "mistralai/Mistral-7B-Instruct-v0.1",
        quantize: bool = True,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        repetition_penalty: float = 1.1,
    ) -> None:
        """
        Initialize the model with the specified model ID or use the default one if not provided.
        
        Parameters:
            model_id (str): The ID of the model to be used.
            quantize (bool): A flag to quantize the model.
            temperature (float): The temperature for sampling the outputs.
            max_tokens (int): The maximum number of tokens to be generated.
            repetition_penalty (float): The repetition penalty to be applied.
        
        Returns:
            None
        """
        self.model, self.tokenizer = self.load_hf_model_tokenizer(model, quantize)
        self.generator = transformers.pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            return_full_text=False,
            temperature=temperature,
            max_new_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
        )
        self.generator.tokenizer.pad_token_id = self.generator.model.config.eos_token_id

    def load_hf_model_tokenizer(
        self, model_id: str, quantize: bool
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load a model and tokenizer for language modeling.

        Args:
            model_id (str): The ID of the model to be loaded.
            quantize (bool): Whether to quantize the model.

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

    def __call__(self, prompt: str) -> str:
        """
        Call the generator with the given prompt and return the generated text.

        Parameters:
            prompt (str): The prompt to be passed to the generator.

        Returns:
            str: The generated text stripped of leading and trailing whitespace.
        """
        response = self.generator(prompt, pad_token_id=self.tokenizer.eos_token_id)

        return response[0]["generated_text"].strip()


# def qa_with_hf_models(
#     prompt: str,
#     model: AutoModelForCausalLM,
#     tokenizer: AutoTokenizer,
#     temperature: float = 0.1,
#     max_tokens: int = 2000,
#     repetition_penalty: float = 1.1,
# ) -> str:
#     """
#     Generates a response using a Hugging Face model for question answering.

#     Args:
#         prompt (str): The prompt for the question answering.
#         model (AutoModelForCausalLM): The Hugging Face model for question answering.
#         tokenizer (AutoTokenizer): The Hugging Face tokenizer for question answering.
#         temperature (float, optional): The temperature for sampling. Defaults to 0.1.
#         max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 2000.
#         repetition_penalty (float, optional): The repetition penalty to apply. Defaults to 1.1.

#     Returns:
#         str: The generated response.
#     """

#     generator = transformers.pipeline(
#         model=model,
#         tokenizer=tokenizer,
#         task="text-generation",
#         return_full_text=False,
#         temperature=temperature,
#         max_new_tokens=max_tokens,
#         repetition_penalty=repetition_penalty,
#     )
#     generator.tokenizer.pad_token_id = generator.model.config.eos_token_id
#     response = generator(prompt, pad_token_id=tokenizer.eos_token_id)

#     return response[0]["generated_text"].strip()
