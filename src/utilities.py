import numpy as np


class Utilities:
    @staticmethod
    def normalize(vector: np.ndarray) -> np.ndarray:
        norm = vector.sum()
        if norm == 0:
            return vector
        return vector / norm

    @staticmethod
    def random_onezero(prob: float) -> bool:
        return np.random.choice([True, False], p=[prob, 1 - prob])

    @staticmethod
    def map_to_type(num: int):
        if num % 2 == 0:
            return "uniform"
        else:
            return "optimal"