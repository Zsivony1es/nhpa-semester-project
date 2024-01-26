from typing import Union

import numpy as np
import scipy


class MatrixGenerator:
    def __init__(self):
        self._shape = None
        self._matrix_type = None
        self._entries_type = None
        self._density = None

    def set_shape(self,
                  shape: Union[tuple[int, int], np.ndarray]) -> None:
        self._shape = shape

    def set_matrix_type(self,
                        matrix_type: str,
                        density: float = 0.01) -> None:
        f"""
        Decides what kind of matrix to generate
        :param matrix_type: Options:
            'dense' - Fills every entry in the matrix
            'sparse' - Generates a sparse matrix with density :density:
        :param density: Set the density if sparse matrices are to be genereated. (Default: 0.01)
        """
        assert matrix_type in ['dense', 'sparse']
        self._matrix_type = matrix_type
        self._density = density

    def set_entries_type(self,
                         entries_type: str):
        f"""
        Sets the rule based on which the value of the entries are to be generated.
        :param entries_type: Options: 
            'frac' - generates uniformly distributed values on the interval [0,1)
            'int' - generates integer only values from 0 - 9
            'float' - generates floats from the interval [1,11)
        """
        assert entries_type in ['frac', 'int', 'float']
        self._entries_type = entries_type

    def generate(self) -> np.ndarray:
        if self._matrix_type == "dense":
            m = np.random.rand(self._shape[0], self._shape[1])
        elif self._matrix_type == "sparse":
            m = scipy.sparse.random(self._shape[0], self._shape[1], density=self._density).toarray()
        else:
            raise ValueError("Invalid matrix type!")

        if self._entries_type == "frac":
            pass
        elif self._entries_type == "int":
            m = np.floor(m * 10).astype(int)
        elif self._entries_type == "float":
            m = m * 10 + 1
        else:
            raise ValueError("Invalid entries type!")

        return m
