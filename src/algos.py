import numpy as np

from helpers import Helpers


class Algorithms:
    def __init__(self):
        pass

    @staticmethod
    def select(column: np.ndarray) -> tuple[int, float]:
        f"""
        The SELECT algorithm as described in the paper: https://doi.org/10.1137/S0097539704442684
        """
        value = 0.0
        index = 0

        d = 0
        length = column.shape[0]
        for i in range(length):
            val = np.abs(column[i])
            d += val
            if d == 0:
                continue
            prob = val / d
            if Helpers.random_onezero(prob):
                index = i
                value = val

        return index, value

    @staticmethod
    def basic_matrix_mult(A: np.ndarray,
                          B: np.ndarray,
                          c: int,
                          prob: np.ndarray) -> np.ndarray:
        f"""
        The Basic Matrix Multiplication algorithm as described in the paper: 
        https://doi.org/10.1137/S0097539704442684
        """
        assert A.shape[1] == B.shape[0], f"The dimensions of A {A.shape} and B {B.shape} don't match!"
        assert 1 <= c <= A.shape[1], f"The c value must be between 1 and {A.shape[1]}!"
        assert np.isclose(prob.sum(), 1), f"The probabilities should add up to 1, but it is {prob.sum()}!"

        # Initialize C and R
        C = np.empty((A.shape[0], c))
        R = np.empty((c, B.shape[1]))

        # Calculate C and R
        for t in range(c):
            i_t = np.random.choice(np.arange(A.shape[1]), p=prob, replace=True)
            C[:, t] = A[:, i_t] / np.sqrt(c * prob[i_t])
            R[t, :] = B[i_t, :] / np.sqrt(c * prob[i_t])

        return C @ R

    @staticmethod
    def elementwise_mult(A: np.ndarray,
                         B: np.ndarray,
                         prob_A: np.ndarray,
                         prob_B: np.ndarray) -> np.ndarray:
        f"""
        The Elementwise Matrix Multiplication algorithm as described in the paper: 
        https://doi.org/10.1137/S0097539704442684
        """
        assert A.shape[1] == B.shape[0], f"The dimensions of A ({A.shape}) and B ({B.shape}) don't match!"
        assert A.shape == prob_A.shape, f"The dimensions of A ({A.shape}) and B ({prob_A.shape}) don't match!"
        assert B.shape == prob_B.shape, f"The dimensions of A ({B.shape}) and B ({prob_B.shape}) don't match!"

        S = np.zeros((A.shape[0], A.shape[1]))
        R = np.zeros((B.shape[0], B.shape[1]))

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if Helpers.random_onezero(prob_A[i, j]):
                    S[i, j] = A[i, j] / prob_A[i, j]

        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                if Helpers.random_onezero(prob_B[i, j]):
                    R[i, j] = B[i, j] / prob_B[i, j]

        return S @ R
