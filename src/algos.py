import numpy as np


class Algorithms:
    def __init__(self):
        pass

    def select(column: np.ndarray) -> (int, float):

        value = 0.0
        index = 0

        d = 0
        length = column.shape[0]
        for i in range(length):
            d += column[i]
            if d == 0:
                continue
            prob = column[i] / d
            if np.random.choice([True, False], p=[prob, 1-prob]):
                index = i
                value = column[i]
        
        return (index, value)

    def opt_select(column: np.ndarray) -> (int, float):
        cumulative_sum = np.cumsum(column)
        total_sum = cumulative_sum[-1]

        random_value = np.random.uniform(0, total_sum)
        index = np.searchsorted(cumulative_sum, random_value)

        return index, column[index]



