# reference_parsing/embeddings/distilbert_reference_embedding.py

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from reference_parsing.embeddings.BaseReferenceEmbedding import BaseReferenceEmbedding

class DistilBertReferenceEmbedding(BaseReferenceEmbedding):
    """
    Token-level, encoder-only embedding extractor using
    `distilbert-base-multilingual-cased` from HuggingFace.
    """

    def __init__(self,
                 model_name: str = "distilbert-base-multilingual-cased",
                 device: str = None):
        # pick CPU if no CUDA, else GPU
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # load tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model     = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # hidden size (Dimensionality of each token vector)
        self.dim = self.model.config.hidden_size

    def get_token_embedding(self, token: str) -> np.ndarray:
        """
        Returns a single vector for `token` by:
          1) tokenizing → subword IDs
          2) passing through DistilBERT encoder
          3) averaging its subword embeddings
        """
        if not token.strip():
            return np.zeros(self.dim, dtype=np.float32)

        # tokenize w/o special tokens so length == # subwords
        inputs = self.tokenizer(token,
                                return_tensors="pt",
                                add_special_tokens=False)
        inputs = {k: v.to(self.device) for k,v in inputs.items()}

        with torch.no_grad():
            out = self.model(**inputs)

        # out.last_hidden_state shape: (1, subwords, dim) → (subwords, dim)
        hidden = out.last_hidden_state.squeeze(0)
        # average subword vectors → single token vector
        vec = hidden.mean(dim=0)
        return vec.cpu().numpy()

    def get_reference_embedding(self, tokens: list) -> np.ndarray:
        """
        Given a list of original tokens, returns a (n_tokens × dim) array.
        """
        embs = [self.get_token_embedding(t) for t in tokens]
        return np.vstack(embs)
