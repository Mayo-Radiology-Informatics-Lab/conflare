"""
Utility functions for building a vector db. Currently Supporting ChromaDB
"""

from typing import Dict, List
import chromadb


def get_vector_db(
    db_path: str,
    db_name: str,
    embedding_fn: callable,
    docs: List[Dict[str, str]] = [],
    db_metadata: Dict[str, str] = {'hnsw:space':'cosine'},
) -> chromadb.Collection:
    """
    A function to load or create a vector database using the provided parameters.
    If a vector database already exists, it will be returned and the docs parameter will be ignored.
    
    Args:
        db_path (str): The path to the vector database.
        db_name (str): The name of the vector database.
        embedding_fn (callable): The function used for embedding.
        docs (List[Dict[str, str]], optional): List of documents with content, source, page, and index. Defaults to [].
        db_metadata (Dict[str, str], optional): Metadata for the database, default includes 'hnsw:space' as 'cosine'.
    
    Returns:
        chromadb.Collection: The vector database collection after loading or creation.
    """
    
    chroma_client = chromadb.PersistentClient(path=db_path)
    vector_db = chroma_client.get_or_create_collection(
        name=db_name,
        embedding_function=embedding_fn,
        metadata=db_metadata
    )
    if vector_db.count() > 0:
        print("Loading from disk... the given docs will be ignored.")
        return vector_db
    
    vector_db.add(
        documents=[doc["content"] for doc in docs],
        metadatas=[
            {
                "type": "article_chunk",
                "source": doc["source"],
                "page": doc["page"],
                "chunk_index": doc["index"]
            }
            for doc in docs
        ],
        ids=[str(i) for i in range(len(docs))]
    )

    return vector_db
    



if __name__ == "__main__":
    client = chromadb.PersistentClient(path="../data/vector_db")
    collection = client.get_or_create_collection(name="example")
    print(collection.count())
