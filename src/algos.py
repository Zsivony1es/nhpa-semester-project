import numpy as np
import math

from helpers import Helpers


class Algorithms:
    def __init__(self):
        pass

    def get_optimal_probdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        f"""
        Compute the optimal probability distribution for the Basic Matrix Multiplication algorithm
        """
        total_AiBi = 0
        prob = np.empty(A.shape[1])
        for i in range(A.shape[1]):
            prob[i] = np.linalg.norm(Algorithms.select(A[:, i])) * np.linalg.norm(Algorithms.select(B[i, :]))
            total_AiBi += prob[i]
        return prob / total_AiBi

    def calculate_prob_bound(A: np.ndarray, B: np.ndarray, c: int, delta: float, type: str = "opt", beta: float = 1.0) -> float:
        """
        :param A: The matrix A
        :param B: The matrix B
        :param c: The number of columns/rows sampled from the matrices
        :param delta: (1 - delta) is the probability that the error is lower than the calculated bound
        :param type: Which formula to use for the bound calculation.
        Either 'opt' for nearly optimal probabilities or 'nonopt' for the rest.
        :param beta: How similar the probabilities are to the optimal probability distribution
        :return: The calculated bound value
        Calculates the bound for ||AB - CR||_F with a given probability
        """
        if type == "opt":
            eta = 1 + np.sqrt((8 / beta) * np.log(1 / delta))
            whp_bound = eta / (beta * np.sqrt(c)) * np.linalg.norm(A, ord='fro') * np.linalg.norm(B, ord='fro')
            return whp_bound
        else:
            pass

    def select(column: np.ndarray) -> (int, float):
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

        return (index, value)

    def opt_select(column: np.ndarray) -> (int, float):
        f"""
        An optimized implementation of the SELECT algorithm as described in the paper: 
        https://doi.org/10.1137/S0097539704442684
        """
        cumulative_sum = np.cumsum(column)
        total_sum = cumulative_sum[-1]

        random_value = np.random.uniform(0, total_sum)
        index = np.searchsorted(cumulative_sum, random_value)

        return index, column[index]

    def basic_matrix_mult(A: np.ndarray, B: np.ndarray, c: int, prob: np.ndarray) -> np.ndarray:
        f"""
        The Basic Matrix Multiplication algorithm as described in the paper: 
        https://doi.org/10.1137/S0097539704442684
        """
        assert A.shape[1] == B.shape[0], f"The dimensions of A ({A.shape}) and B ({B.shape}) don't match!"
        assert 1 <= c <= A.shape[1], f"The c value must be between 1 and {A.shape[1]}!"
        assert np.isclose(prob.sum(), 1), "The probabilities should add up to 1!"

        # Initialize C and R
        C = np.empty((A.shape[0], c))
        R = np.empty((c, B.shape[1]))

        # Calculate C and R
        for t in range(c):
            i_t = np.random.choice(np.arange(A.shape[1]), p=prob, replace=True)
            C[:, t] = A[:, i_t] / math.sqrt(c * prob[i_t])
            R[t, :] = B[i_t, :] / math.sqrt(c * prob[i_t])

        return C @ R

    def elementwise_mult(A: np.ndarray, B: np.ndarray, prob_A: np.ndarray, prob_B: np.ndarray) -> np.ndarray:
        f"""
        The Elementwise Matrix Multiplication algorithm as described in the paper: 
        https://doi.org/10.1137/S0097539704442684
        """
        assert A.shape[1] == B.shape[0], f"The dimensions of A ({A.shape}) and B ({B.shape}) don't match!"
        assert A.shape == prob_A.shape, f"The dimensions of A ({A.shape}) and B ({prob_A.shape}) don't match!"
        assert B.shape == prob_B.shape, f"The dimensions of A ({B.shape}) and B ({prob_B.shape}) don't match!"
        assert np.isclose(prob_A.sum(), 1) and np.isclose(prob_B.sum(), 1), \
            "The entries in the probability matrices should add up to 1!"

        S = np.zeros((A.shape[0], A.shape[1]))
        R = np.zeros((B.shape[0], B.shape[1]))

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if Helpers.random_onezero(prob_A[i, j]):
                    S[i, j] = A[i, j] / prob_A[i, j]

        return S @ R
