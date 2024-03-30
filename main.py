from utils.loaders import load_dir

if __name__ == "__main__":
    docs = load_dir("./data/documents")
    print(len(docs))
    print(docs[0])
