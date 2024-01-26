from typing import Union

import numpy as np
import scipy


class Helpers:
    @staticmethod
    def normalize(vector: np.ndarray) -> np.ndarray:
        norm = vector.sum()
        if norm == 0:
            return vector
        return vector / norm

    @staticmethod
    def random_onezero(prob: float) -> bool:
        return np.random.choice([True, False], p=[prob, 1 - prob])

