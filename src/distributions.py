import numpy as np
from algos import Algorithms

class Distributions:
    f"""
    The class contains all methods related to calculating probability distributions for the randomized
    matrix multiplication algorithms.
    """

    @staticmethod
    def get_semiopt_probdist(M: np.ndarray,
                             sampling="col") -> tuple[np.ndarray, float]:
        f"""
        :param A: The matrix used to compute the probability distribution
        :param sampling: Sets whether to sample the matrix by column or row. Options: 'col', 'row', default: col
        Compute the a semi-optimal probability distribution for the Basic Matrix Multiplication algorithm 
        using only one of the matrices. (Found in the second/third row of Table 1 in the paper)
        """
        M_frobnorm_squared = np.linalg.norm(M) ** 2
        if sampling == "col":
            prob = np.empty(M.shape[1])
            for i in range(M.shape[1]):
                prob[i] = (Algorithms.select(M[:, i])[1] ** 2) / M_frobnorm_squared
        elif sampling == "row":
            prob = np.empty(M.shape[0])
            for i in range(M.shape[0]):
                prob[i] = (Algorithms.select(M[i, :])[1] ** 2) / M_frobnorm_squared
        else:
            raise Exception("Argument sampling must be either 'col' or 'row'!")

        # I have to adjust the values by calculating beta, so it returns a valid probability distribution
        beta = (1 / prob.sum())
        prob = beta * prob
        return prob, beta

    @staticmethod
    def get_optimal_probdist(A: np.ndarray,
                             B: np.ndarray) -> np.ndarray:
        f"""
        Compute the optimal probability distribution for the Basic Matrix Multiplication algorithm
        """
        total_AiBi = 0
        prob = np.empty(A.shape[1])
        for i in range(A.shape[1]):
            prob[i] = np.linalg.norm(Algorithms.select(A[:, i])) * np.linalg.norm(Algorithms.select(B[i, :]))
            total_AiBi += prob[i]
        return prob / total_AiBi

    @staticmethod
    def get_opt_prodist_elementwise(A: np.ndarray,
                                    B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns {p_ij} and {q_ij}, which are probability distributions to be used in the element-wise
        matrix multiplication algorithm. This assumes that we are multiplying square matrices, as per the paper.
        """
        shape = A.shape
        dim = shape[0]
        p = np.empty(shape)
        q = np.empty(shape)
        A_norm = np.linalg.norm(A)
        B_norm = np.linalg.norm(B)
        l = (A_norm ** 2) / (A.max() ** 2)
        k = (B_norm ** 2) / (B.max() ** 2)

        for i in range(dim):
            for j in range(dim):
                if np.abs(A[i, j]) > (A_norm * np.log(2 * dim) ** 3) / np.sqrt(2 * dim * l):
                    p[i, j] = min(1,
                                  (l * A[i, j] ** 2) / (A_norm ** 2) )
                else:
                    p[i, j] = min(1,
                                  (np.sqrt(l) * np.abs(A[i, j]) * np.log(2 * dim) ** 3) / (np.sqrt(2 * dim) * A_norm))

                if np.abs(B[i, j]) > (B_norm * np.log(2 * dim) ** 3) / np.sqrt(2 * dim * k):
                    q[i, j] = min(1,
                                  (k * B[i, j] ** 2) / (B_norm ** 2) )
                else:
                    q[i, j] = min(1,
                                  (np.sqrt(k) * np.abs(B[i, j]) * np.log(2 * dim) ** 3) / (np.sqrt(2 * dim) * B_norm))

        return p, q

