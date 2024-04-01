"""
Pre-processing and text cleaning utilities.
"""

from typing import Dict, List


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    A function that chunks a text into smaller pieces with a specified size and overlap.

    Parameters:
    - text (str): The input text to be chunked.
    - chunk_size (int): The size of each chunk.
    - chunk_overlap (int): The amount of overlap between consecutive chunks.

    Returns:
    - List[str]: A list of chunked text segments.
    """

    chunked_docs = []
    i = 0
    while i < len(text):
        end_index = min(i + chunk_size, len(text))
        chunk = text[i: end_index]
        chunked_docs.append("..." + chunk + "...")
        i += chunk_size - chunk_overlap
        if i + chunk_size - chunk_overlap > len(text):
            break

    return chunked_docs


def chunk_docs(
    docs: List[Dict[str, str]], chunk_size: int = 1500, chunk_overlap: int = 200
) -> List[Dict[str, str]]:
    """
    Generates chunks of text from a list of documents.

    Args:
        docs (List[Dict[str, str]]): A list of documents, where each document is represented as a dictionary with "content" and other optional fields.
        chunk_size (int, optional): The maximum size of each chunk. Defaults to 1500.
        chunk_overlap (int, optional): The number of characters by which each chunk should overlap. Defaults to 200.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, where each dictionary represents a chunk of text. 
        Each dictionary contains the chunk's text in "chunk", the index of the chunk in "chunk_index" within the document, 
        and any other fields from the original document that are not "content".
    """
    
    chunked_docs = []
    for doc in docs:
        chunks = chunk_text(doc["content"], chunk_size, chunk_overlap)
        for i, chunk in enumerate(chunks):
            chunked_docs.append(
                {"chunk": chunk, "chunk_index": i, **{k: v for k, v in doc.items() if k != "content"}}
            )

    return chunked_docs

def remove_illegal_chars(text: str) -> str:
    """
    Removes illegal characters from the given text.

    Args:
        text (str): The text from which illegal characters need to be removed.

    Returns:
        str: The text with illegal characters removed.
    """
    illegal_chars = [
        "\x00",
        "\x01",
        "\x02",
        "\x03",
        "\x04",
        "\x05",
        "\x06",
        "\x07",
        "\x08",
        "\x0b",
        "\x0c",
        "\x0e",
        "\x0f",
        "\x10",
        "\x11",
        "\x12",
        "\x13",
        "\x14",
        "\x15",
        "\x16",
        "\x17",
        "\x18",
        "\x19",
        "\x1a",
        "\x1b",
        "\x1c",
        "\x1d",
        "\x1e",
        "\x1f",
    ]
    for char in illegal_chars:
        text = text.replace(char, "")

    return text
