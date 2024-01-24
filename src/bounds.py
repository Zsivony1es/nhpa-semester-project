import numpy as np


class Bounds:
    f"""
    Calculate probability bounds for the accuracy of the matrix multiplication algoritmhs.
    """

    @staticmethod
    def calc_prob_bound_m(A: np.ndarray,
                          B: np.ndarray,
                          using_mx='A') -> float:
        assert A.shape[1] == B.shape[0], "Number of columns in A must be equal to the number of rows in B"

        num_alphas = A.shape[1]
        max_ratios = np.zeros(num_alphas)

        for alpha in range(num_alphas):
            A_alpha = A[:, alpha].reshape(-1, 1)
            B_alpha = B[alpha, :].reshape(1, -1)
            if using_mx == 'A':
                ratio = np.linalg.norm(B_alpha) / np.linalg.norm(A_alpha)
                max_ratios[alpha] = ratio
            elif using_mx == 'B':
                ratio = np.linalg.norm(A_alpha) / np.linalg.norm(B_alpha)
                max_ratios[alpha] = ratio
            else:
                raise ValueError("using_mx must be 'A' or 'B'!")

        return np.max(max_ratios)

    @staticmethod
    def calculate_prob_bound(A: np.ndarray,
                             B: np.ndarray,
                             c: int,
                             delta: float,
                             type: str = "opt",
                             beta: float = 1.0) -> float:
        """
        :param A: The matrix A
        :param B: The matrix B
        :param c: The number of columns/rows sampled from the matrices
        :param delta: (1 - delta) is the probability that the error is lower than the calculated bound
        :param type: Which formula to use for the bound calculation.
        Options: 'opt' (default), 'nearopt', 'nonopt'
        :param beta: How similar the probabilities are to the optimal probability distribution
        :return: The calculated bound value
        Calculates the bound for ||AB - CR||_F with a given probability
        """
        if type == "opt":
            eta = 1 + np.sqrt((8 / beta) * np.log(1 / delta))
            whp_bound = eta / (beta * np.sqrt(c)) * np.linalg.norm(A, ord='fro') * np.linalg.norm(B, ord='fro')
            return whp_bound
        elif type == "nearopt":
            M = Bounds.calc_prob_bound_m(A, B)
            eta = 1 + ( (np.linalg.norm(A, ord='fro') / np.linalg.norm(B, ord='fro'))
                   * M * np.sqrt((8 / beta) * np.log(1 / delta)))
        elif type == "nonopt":
            pass
        else:
            raise Exception("Argument 'type' must be either 'opt', 'nearopt' or 'nonopt'!")
