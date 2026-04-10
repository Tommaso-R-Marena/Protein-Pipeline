import numpy as np

def extract_features(sequence: str) -> np.ndarray:
    """
    Extracts 118 physicochemical features at 7 length scales 
    (7, 11, 15, 21, 31, 51, 101).
    Total features per residue combined with ESM PCA will be 406.
    
    Order promoters: W, Y, F, I, L, V, C, N
    Disorder promoters: R, K, E, S, P, Q, A, G
    """
    N = len(sequence)
    # Stub: return mock features of expected dims
    return np.zeros((N, 406))
