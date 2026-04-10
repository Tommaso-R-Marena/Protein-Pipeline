import numpy as np

def mirror_coordinates(coords: np.ndarray) -> np.ndarray:
    """
    Reflect through y-z plane: x → -x. Exactly 0.0 Å error on L↔D flip.
    """
    mirrored = coords.copy()
    if mirrored.ndim == 2:
        mirrored[:, 0] *= -1
    elif mirrored.ndim == 1:
        mirrored[0] *= -1
    return mirrored
