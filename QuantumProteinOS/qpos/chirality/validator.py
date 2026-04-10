import numpy as np

def signed_chiral_volume(r_center: np.ndarray, r_j: np.ndarray, r_k: np.ndarray, r_l: np.ndarray) -> float:
    """
    Computes signed chiral volume.
    V = (r_center - r_j) · ((r_center - r_k) × (r_center - r_l))
    L-amino acid: V < 0
    D-amino acid: V > 0
    """
    a = r_center - r_j
    b = r_center - r_k
    c = r_center - r_l
    return float(np.dot(a, np.cross(b, c)))

def validate_chirality(r_ca: np.ndarray, r_n: np.ndarray, r_c: np.ndarray, r_cb: np.ndarray) -> bool:
    """
    Returns True if the coordinates form a correct L-amino acid layout (V < 0).
    """
    v = signed_chiral_volume(r_ca, r_n, r_c, r_cb)
    return v < 0
