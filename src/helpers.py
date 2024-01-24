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

    @staticmethod
    def generate_matrices(shape_a: Union[tuple[int, int], np.ndarray],
                          shape_b: Union[tuple[int, int], np.ndarray],
                          matrix_type: str = "dense") -> tuple[np.ndarray, np.ndarray]:
        if matrix_type == "dense":
            a = np.random.rand(shape_a[0], shape_a[1])
            b = np.random.rand(shape_b[0], shape_b[1])
        elif matrix_type == "sparse":
            a = scipy.sparse.random(shape_a[0], shape_a[1], density=0.01).toarray()
            b = scipy.sparse.random(shape_b[0], shape_b[1], density=0.01).toarray()
        else:
            raise ValueError("Invalid matrix type!")

        return a, b
