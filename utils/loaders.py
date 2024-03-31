"""
Document loaders for different formats.
"""

from typing import Dict, List
from glob import glob
from pypdf import PdfReader
from tqdm import tqdm

from utils.preprocess import remove_illegal_chars

def pdf_loader(pdf_path: str) -> List[Dict[str, str]]:
    """
    Load a PDF file and extract its contents into a list of dictionaries.

    Parameters:
        pdf_path (str): The path to the PDF file.

    Returns:
        List[Dict[str, str]]: list of dicts, where each dictionary contains the source PDF path,
        the page number, and the extracted text content of each page in the PDF file.
    """
    pdf_docs = []
    reader = PdfReader(pdf_path)
    for i, page in enumerate(reader.pages):
        content = remove_illegal_chars(page.extract_text())
        pdf_docs.append({"source": pdf_path, "page": i, "content": content})

    return pdf_docs


def load_dir(path: str, extension: str = "pdf") -> List[Dict[str, str]]:
    """
    Loads all files with a specified extension from a given directory and returns
    a list of dictionaries representing the documents.

    Parameters:
        path (str): The path to the directory containing the files.
        extension (str, optional): The file extension to filter the files by. Defaults to "pdf".

    Returns:
        List[Dict[str, str]]: A list of dictionaries representing the documents.
        Each dictionary contains the document path and its content.

    Note:
        The function currently supports loading PDF files. Other formats are not implemented.
    """
    doc_paths = glob(f"{path}/*.{extension}")
    print(f"Found {len(doc_paths)} {extension} files in this dir.")
    docs = []
    if extension == "pdf":
        loader = pdf_loader
    else:
        # TODO: Implement other formats
        raise NotImplementedError(f"Extension {extension} not supported")

    for doc_path in tqdm(doc_paths):
        docs.extend(loader(doc_path))

    return docs
