# reference_parsing/embeddings/base_reference_embedding.py
from abc import ABC, abstractmethod

class BaseReferenceEmbedding(ABC):
    """
    An abstract base class for extracting token embeddings and reference-level embeddings.
    """

    @abstractmethod
    def get_token_embedding(self, token: str):
        """
        Returns the embedding for a single token.
        For numeric embeddings, this might be a 1-D NumPy array.
        For dictionary-based features, it might be a dictionary.
        """
        pass

    @abstractmethod
    def get_reference_embedding(self, tokens: list):
        """
        Given a list of tokens, returns a single reference embedding.
        This method must be implemented by each subclass.
        """
        pass
