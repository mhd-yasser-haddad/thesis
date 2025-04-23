import numpy as np

from flair.data import Sentence
from flair.embeddings import BytePairEmbeddings as FlairBytePairEmbeddings
from reference_parsing.embeddings.BaseReferenceEmbedding import BaseReferenceEmbedding
from reference_parsing.config import BPEM_MODEL_FILE_PATH, BPME_EMBEDDING_FILE_PATH

class BytePairReferenceEmbedding(BaseReferenceEmbedding):
    def __init__(self, language="multi", dim=300, model_file_path=BPEM_MODEL_FILE_PATH, embedding_file_path=BPME_EMBEDDING_FILE_PATH):
        self.emb_model = FlairBytePairEmbeddings(language=language, dim=dim,
                                            model_file_path=model_file_path,
                                            embedding_file_path=embedding_file_path)

    def get_token_embedding(self, token: str) -> np.array:
        if not token.strip():
            return np.zeros(600)
        # Wrap the token in a Sentence (Flair requires a Sentence object)
        sent = Sentence(token, use_tokenizer=False)
        # Compute the embedding for this sentence (which consists of a single token)
        self.emb_model.embed([sent])    
        return sent[0].embedding.cpu().numpy()
    
    def get_reference_embedding(self, tokens: list) -> np.array:
        """
        return a 2D array of token embeddings with shape (n_tokens, dim).
        """
        embeddings = [self.get_token_embedding(token) for token in tokens]
        return np.vstack(embeddings)
