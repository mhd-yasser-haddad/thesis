import torch
import numpy as np
from flair.data import Sentence
from transformers import AutoTokenizer, AutoModel
from reference_parsing.embeddings.BaseReferenceEmbedding import BaseReferenceEmbedding

class BERTReferenceEmbedding(BaseReferenceEmbedding):
    def __init__(self,
                 model_name: str = "Linq-AI-Research/Linq-Embed-Mistral",
                 device: str = None):
        # pick CPU if no CUDA, else GPU
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # load tokenizer + model locally (already cloned)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       trust_remote_code=True,
                                                       use_fast=True)
        self.model     = AutoModel.from_pretrained(model_name,
                                                   trust_remote_code=True,
                                                   low_cpu_mem_usage=True,
                                                   device_map="auto",
                                                   torch_dtype=torch.float16)
        # don't call .to() when device_map is used
        self.model.eval()
        self.dim = self.model.config.hidden_size

    def get_token_embedding(self, token: str) -> np.ndarray:
        if not token.strip():
            return np.zeros(self.dim, dtype=np.float32)

        # tokenize (no special tokens so length = number of subwords)
        inputs = self.tokenizer(token,
                                return_tensors="pt",
                                add_special_tokens=False)
        # dispatch inputs to wherever the model is sharded
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        # shape (1, subwords, dim) -> (subwords, dim)
        last_hidden = outputs.last_hidden_state.squeeze(0)
        # average to a single vector
        vec = last_hidden.mean(dim=0)
        return vec.cpu().numpy()

    def get_reference_embedding(self, tokens: list) -> np.ndarray:
        embs = [self.get_token_embedding(t) for t in tokens]
        # stack into (n_tokens, dim)
        return np.vstack(embs)
