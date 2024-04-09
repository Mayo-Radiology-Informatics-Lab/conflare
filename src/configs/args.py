# DATA
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
CALIBRATION_SIZE = 100
TOPIC = "Deep Learning"
MAX_CHUNK_EVAL = 100

# MODEL
MODEL = "gpt-4" # remember to modify the prompts in utils/prompts.py accordingly with the type of model you use.
QUANTIZE = True # used only for HF models
TEMPERATURE = 0.1
MAX_TOKENS = 2000
FREQUENCY_PENALTY = 1.1

# EMBEDDING
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32
NORMALIZE_EMBEDDINGS = True

# VECTOR DB
DB_NAME = "conflare"
DB_METADATA = {'hnsw:space':'cosine'}

# RAG
ERROR_RATE = 0.05
VERBOSE = True