import numpy as np

def normalize_embedding_with_prefix(emb, prefix="default"):
    """
    Normalizes an embedding (which could be a numpy array, list/tuple, dict, or scalar)
    into a dictionary where the keys are prefixed by the given string.
    
    The features will be added in the natural iteration order:
      - If emb is a numpy array, they will be added in order from index 0 to len-1.
      - If emb is a list or tuple, the order of indices is preserved.
      - If emb is already a dict, it is iterated in its insertion order.
    """
    if isinstance(emb, np.ndarray):
        # Convert array to list
        emb_list = emb.tolist()
        return {f"{prefix}_emb_{i}": float(emb_list[i]) for i in range(len(emb_list))}
    elif isinstance(emb, (list, tuple)):
        # Iterate by index order; convert numeric elements appropriately.
        return {f"{prefix}_emb_{i}": float(emb[i]) if isinstance(emb[i], (int, float, np.number)) 
                else str(emb[i])
                for i in range(len(emb))}
    elif isinstance(emb, dict):
        # Use the dictionary's insertion order.
        return {f"{prefix}_{key}": emb[key] for key in emb}
    else:
        # For a scalar or other type, wrap it in a dict.
        return {f"{prefix}_emb": emb}

def merge_token_embeddings(emb1, emb2, prefix1="emb1", prefix2="emb2"):
    """
    Merges two token embeddings into a single dictionary.
    
    Both emb1 and emb2 may be numeric vectors (e.g., a numpy array or a list)
    or dictionaries. This function converts each into a dictionary with features
    ordered as provided, with keys prefixed using the given prefixes.
    
    The merged output will contain all features from emb1 (in order)
    followed by all features from emb2 (in order).
    """
    norm1 = normalize_embedding_with_prefix(emb1, prefix1)
    norm2 = normalize_embedding_with_prefix(emb2, prefix2)
    merged = {}
    # Insert norm1's features first.
    for key, value in norm1.items():
        merged[key] = value
    # Then insert norm2's features.
    for key, value in norm2.items():
        merged[key] = value
    return merged

def combine_embedding_lists(list1, list2, prefix1="emb1", prefix2="emb2"):
    """
    Given two lists of token embeddings (one embedding per token) of equal length,
    merge them token-by-token using merge_token_embeddings.
    
    Returns:
      A list of dictionaries where the i-th dictionary is the merged output for the i-th token.
    """
    if len(list1) != len(list2):
        raise ValueError("Embedding lists must have the same length.")
    
    return [merge_token_embeddings(e1, e2, prefix1, prefix2) for e1, e2 in zip(list1, list2)]
