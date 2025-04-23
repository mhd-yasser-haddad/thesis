# reference_parsing/config.py
from pathlib import Path

# Define the project root relative to this file
PROJECT_ROOT = Path(__file__).resolve().parent

# Commonly used directories relative to the project root:
# DATA_DIR = PROJECT_ROOT / "data"
HELPER_FILES_DIR = PROJECT_ROOT / "helper_files"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings_resources"

DEFAULT_EMBEDDING_DIM = 300
DICT_FILE = HELPER_FILES_DIR / "dict.txt.gz"

# BPEm Variables
BPEM_MODEL_FILE_PATH = EMBEDDINGS_DIR / "BytePairEmbeddingsFiles" / "multi.wiki.bpe.vs100000.model"
BPME_EMBEDDING_FILE_PATH = EMBEDDINGS_DIR / "BytePairEmbeddingsFiles" / "multi.wiki.bpe.vs100000.d300.w2v.txt"

LABEL2ID = {
    'B-AUTHOR': 0, 'I-AUTHOR': 1, 'B-YEAR': 2, 'I-YEAR': 3, 
    'B-TITLE': 4, 'I-TITLE': 5, 'B-CONTAINER-TITLE': 6, 'I-CONTAINER-TITLE': 7,
    'B-VOLUME': 8, 'I-VOLUME': 9, 'B-ISSUE': 10, 'I-ISSUE': 11, 
    'B-PAGE': 12, 'I-PAGE': 13, 'B-ISBN': 14, 'I-ISBN': 15,
    'B-ISSN': 16, 'I-ISSN': 17, 'B-PUBLISHER': 18, 'I-PUBLISHER': 19,
    'B-DOI': 20, 'I-DOI': 21, 'B-URL': 22, 'I-URL': 23, 
    'O': 24, 'B-PUNC': 25
}
