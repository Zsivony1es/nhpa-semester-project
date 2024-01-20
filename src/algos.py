import numpy as np
import math


class Algorithms:
    def __init__(self):
        pass

    def select(column: np.ndarray) -> (int, float):

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
            if np.random.choice([True, False], p=[prob, 1 - prob]):
                index = i
                value = val

        return (index, value)

    def opt_select(column: np.ndarray) -> (int, float):
        cumulative_sum = np.cumsum(column)
        total_sum = cumulative_sum[-1]

        random_value = np.random.uniform(0, total_sum)
        index = np.searchsorted(cumulative_sum, random_value)

        return index, column[index]

    def basic_matrix_mult(A: np.ndarray, B: np.ndarray, c: int, prob: np.ndarray) -> np.ndarray:
        assert A.shape[1] == B.shape[0], f"The dimensions of A ({A.shape}) and B ({B.shape}) don't match!"
        assert 1 <= c <= A.shape[1], f"The c value must be between 1 and {A.shape[1]}!"
        assert prob.shape[0] == A.shape[1], \
            f"'prob' must contain {A.shape[1]} entries, but it contains {prob.shape[0]}!"
        assert np.isclose(prob.sum(), 1), f"The sum of the probabilities must be 1, but it is {prob.sum()}!"

        C = np.empty((A.shape[0], c))
        R = np.empty((c, B.shape[1]))

        for t in range(c):
            i_t = np.random.choice(np.arange(A.shape[1]), prob, replace=True)
            C[:, t] = A[:, i_t] / math.sqrt(c * prob[i_t])
            R[t, :] = B[i_t, :] / math.sqrt(c * prob[i_t])

        return C @ R
