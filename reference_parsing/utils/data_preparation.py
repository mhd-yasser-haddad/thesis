# reference_parsing/utils/data_preparation.py
import torch
import numpy as np
from reference_parsing.utils.embedding_utils import combine_embedding_lists

def prepare_crf_data(embedding_obj1, dataset, embedding_obj2=None, prefix1="emb1", prefix2="emb2"):
    """
    Prepares data for CRFSuite by obtaining token-level embeddings from one or two embedding objects.
    
    Parameters:
      embedding_obj1: The first embedding object. It must have a method:
                      get_reference_embedding(tokens: list) -> list
                      that returns a list of token embeddings (e.g., one per token), where each embedding is
                      a dictionary (or a numeric vector convertible to a dictionary) for that token.
      dataset: A dict-like object with keys "tokens" and "labels". 
               - dataset["tokens"] is a list of sentences, with each sentence being a list of token strings.
               - dataset["labels"] is a list of label sequences.
      embedding_obj2: (Optional) A second embedding object with the same interface. If provided, the function
                      will merge its token embeddings with those from embedding_obj1.
      prefix1: A string prefix for the features coming from embedding_obj1.
      prefix2: A string prefix for the features coming from embedding_obj2.
                      
    Returns:
      X: A list of sentences, where each sentence is a list of merged feature dictionaries for each token.
         If only embedding_obj1 is provided, then each token's features come solely from that object.
      y: A list of label sequences (one per sentence).
    """
    X = []
    y = []
    
    # Loop over each sentence in the dataset
    for tokens, labels in zip(dataset["tokens"], dataset["labels"]):
        # Get token embeddings from the first embedding object.
        emb1 = embedding_obj1.get_reference_embedding(tokens)
        
        # If a second embedding object was provided, get its token embeddings and merge them.
        if embedding_obj2 is not None:
            emb2 = embedding_obj2.get_reference_embedding(tokens)
            merged = combine_embedding_lists(emb1, emb2, prefix1=prefix1, prefix2=prefix2)
            X.append(merged)
        else:
            # Otherwise, use the embeddings from the first object directly.
            X.append(emb1)
        
        y.append(labels)
    return X, y

def prepare_bilstm_crf_data(embedding_obj1, embedding_obj2, dataset):
    """
    For each sentence in the dataset, obtains two separate token-level embeddings:
      - From embedding_obj1 (e.g., a BPE-based embedding object)
      - From embedding_obj2 (e.g., a hand-feature embedding object)
    and returns:
      X_emb1: a list of sentences, each sentence a list of numeric token vectors from embedding_obj1.
      X_emb2: a list of sentences, each sentence a list of numeric token vectors from embedding_obj2.
      Y: a list of label sequences (each label is still a string).
    
    Parameters:
      embedding_obj1: An object with get_reference_embedding(tokens: list).
      embedding_obj2: An object with get_reference_embedding(tokens: list).
      dataset: A dict with keys "tokens" and "labels" (lists of sentences).
      
    Returns:
      X_bpe, X_hand, Y.
    """
    X_emb1 = []
    X_emb2 = []
    Y = []
    for tokens, labels in zip(dataset["tokens"], dataset["labels"]):
        # Each of these is expected to return a list of embeddings, one per token.
        emb1 = embedding_obj1.get_reference_embedding(tokens)
        emb2 = embedding_obj2.get_reference_embedding(tokens)
        X_emb1.append(emb1)
        X_emb2.append(emb2)
        Y.append(labels)
    return X_emb1, X_emb2, Y

def collate_bilstm_crf(batch, label2id, hand_feature_vocab):
    """
    Collate function for the BiLSTM-CRF model.

    Each item in batch is a tuple: (bpe_embeddings, hand_embeddings, labels),
    where:
      - bpe_embeddings is a list of numpy arrays (one per token) from the BPEmb branch.
      - hand_embeddings is a list of hand-feature outputs for each token.
        In this revised approach, hand_embeddings for a token is a list of static strings.
      - labels is a list of label strings (one per token).

    This function pads all sequences to the maximum length in the batch and returns:
      - bpe_tensor: (batch, max_seq_len, bpe_dim) as float32 tensor.
      - hand_tensor: (batch, max_seq_len) as LongTensor containing indices from hand_feature_vocab.
      - tags_tensor: (batch, max_seq_len) as LongTensor containing label indices.
      - mask_tensor: (batch, max_seq_len) as BoolTensor for valid tokens.
    """
    bpe_seqs = [item[0] for item in batch]
    hand_seqs = [item[1] for item in batch]
    label_seqs = [item[2] for item in batch]

    batch_size = len(bpe_seqs)
    max_len = max(len(seq) for seq in bpe_seqs)

    # Determine bpe_dim from the first token
    bpe_dim = bpe_seqs[0][0].shape[0]

    bpe_tensor_list = []
    hand_tensor_list = []
    tags_tensor_list = []
    mask_list = []

    for bpe_seq, hand_seq, labels in zip(bpe_seqs, hand_seqs, label_seqs):
        seq_len = len(bpe_seq)
        padded_bpe = np.zeros((max_len, bpe_dim), dtype=np.float32)
        # For hand features, we'll create a vector of indices, padded with 0.
        padded_hand = np.full((max_len,), fill_value=hand_feature_vocab["<UNK>"], dtype=np.int64)
        padded_tags = np.full((max_len,), fill_value=-1, dtype=np.int64)
        mask = np.zeros((max_len,), dtype=bool)

        for i in range(seq_len):
            # BPE embeddings.
            padded_bpe[i] = bpe_seq[i]
            
            # Process hand embeddings: if hand_seq[i] is a list of strings,
            # join them to form a composite string.
            token_hand_list = hand_seq[i]  # e.g., ["caps", "Lu", "Ll", "T", "initial", "none", "F", "F", "first", "none", "none", "none", "F"]
            if isinstance(token_hand_list, (list, tuple)):
                composite = "_".join(token_hand_list)
            else:
                composite = str(token_hand_list)
            index = hand_feature_vocab.get(composite, hand_feature_vocab["<UNK>"])
            padded_hand[i] = index
            
            # Convert labels.
            padded_tags[i] = label2id[labels[i]]
            mask[i] = True
        
        bpe_tensor_list.append(torch.tensor(padded_bpe))
        hand_tensor_list.append(torch.tensor(padded_hand))
        tags_tensor_list.append(torch.tensor(padded_tags))
        mask_list.append(torch.tensor(mask))
    
    bpe_tensor = torch.stack(bpe_tensor_list, dim=0)
    hand_tensor = torch.stack(hand_tensor_list, dim=0)
    tags_tensor = torch.stack(tags_tensor_list, dim=0)
    mask_tensor = torch.stack(mask_list, dim=0)
    
    return bpe_tensor, hand_tensor, tags_tensor, mask_tensor



# def collate_bilstm_crf(batch, label2id):
#     """
#     Collate function for the BiLSTM-CRF model.

#     Each item in batch is a tuple: (bpe_embeddings, hand_embeddings, labels),
#     where:
#       - bpe_embeddings is a list of numpy arrays (one per token) from the first embedding object.
#       - hand_embeddings is a list of numpy arrays (one per token) from the second embedding object.
#       - labels is a list of label strings (one per token).
    
#     This function pads all sequences to the maximum length in the batch and returns:
#       - bpe_tensor: (batch_size, max_seq_len, bpe_dim)
#       - hand_tensor: (batch_size, max_seq_len, hand_dim)
#       - tags_tensor: (batch_size, max_seq_len) as LongTensor (numeric labels)
#       - mask_tensor: (batch_size, max_seq_len) as BoolTensor (True for valid tokens)
#     """
#     bpe_seqs = [item[0] for item in batch]
#     hand_seqs = [item[1] for item in batch]
#     label_seqs = [item[2] for item in batch]
    
#     batch_size = len(bpe_seqs)
#     max_len = max(len(seq) for seq in bpe_seqs)
    
#     # Determine dimensions from the first token of the first sentence
#     # (Assume at least one sentence is non-empty)
#     bpe_dim = bpe_seqs[0][0].shape[0]
#     first_hand = hand_seqs[0][0]
#     if hasattr(first_hand, "shape"):
#         hand_dim = first_hand.shape[0]
#     elif isinstance(first_hand, (list, tuple)):
#         hand_dim = len(first_hand)
#     else:
#         hand_dim = 1  # If it's a scalar
#     # hand_dim = hand_seqs[0][0].shape[0]
    
#     bpe_tensor_list = []
#     hand_tensor_list = []
#     tags_tensor_list = []
#     mask_list = []
    
#     for bpe_seq, hand_seq, labels in zip(bpe_seqs, hand_seqs, label_seqs):
#         seq_len = len(bpe_seq)
#         padded_bpe = np.zeros((max_len, bpe_dim), dtype=np.float32)
#         padded_hand = np.zeros((max_len, hand_dim), dtype=np.float32)
#         padded_tags = np.full((max_len,), fill_value=-1, dtype=np.int64)
#         mask = np.zeros((max_len,), dtype=np.bool_)
        
#         # For each token in the sentence, assign values
#         for i in range(seq_len):
#             padded_bpe[i] = bpe_seq[i]
#             padded_hand[i] = hand_seq[i]
#             padded_tags[i] = label2id[labels[i]]
#             mask[i] = True
        
#         bpe_tensor_list.append(torch.tensor(padded_bpe))
#         hand_tensor_list.append(torch.tensor(padded_hand))
#         tags_tensor_list.append(torch.tensor(padded_tags))
#         mask_list.append(torch.tensor(mask))
    
#     bpe_tensor = torch.stack(bpe_tensor_list, dim=0)
#     hand_tensor = torch.stack(hand_tensor_list, dim=0)
#     tags_tensor = torch.stack(tags_tensor_list, dim=0)
#     mask_tensor = torch.stack(mask_list, dim=0)
    
#     return bpe_tensor, hand_tensor, tags_tensor, mask_tensor


