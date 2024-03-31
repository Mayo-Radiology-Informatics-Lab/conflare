from utils.loaders import load_dir
from utils.preprocess import chunk_docs

if __name__ == "__main__":
    docs = load_dir("./data/documents")
    docs = chunk_docs(docs)
    print(len(docs))
    # index = 10
    # doc = {k: v[:50] if k == 'content' else v for k, v in docs[index].items()}
    chunk_count = {}
    for doc in docs:
        if doc['source'] in chunk_count:
            chunk_count[doc['source']] += 1
        else:
            chunk_count[doc['source']] = 1

    for k, v in chunk_count.items():
        print(k[:20], v)