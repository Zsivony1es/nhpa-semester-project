import numpy as np


class Helpers:
    def normalize(vector: np.ndarray) -> np.ndarray:
        norm = vector.sum()
        if norm == 0:
            return vector
        return vector / norm
