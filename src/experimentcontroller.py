import logging

import numpy as np
from parso.normalizer import Normalizer

from matrix_gen import MatrixGenerator
from algos import Algorithms
from bounds import Bounds
from distributions import Distributions


class ExperimentController:

    def __init__(self,
                 matrix_type: str = "dense",
                 entries_type: str = "frac",
                 prob_dist_type: str = "opt",
                 delta: float = 0.05,
                 normalize_error: bool = False):
        f"""
        :param matrix_type: The type of matrix to be generated. Options are 'dense' and 'sparse'. (Default: 'dense')
        :param entries_type: Options: 
            'frac' - generates uniformly distributed values on the interval [0,1)
            'int' - generates integer only values from 0 - 9
            'float' - generates floats from the interval [1,11)
        :param prob_dist_type: Used probability distribution type. Options are 'opt', 'nearopt', 'nonopt', 'uniform'.
          (Default: 'opt')
        :param delta: The probability that the calculated accuracy metric will be beyond the calculated bound
        :param normalize_error: If set to True, then the returned values will be divided by ||AB||_F
        """
        self._matrix_type = matrix_type
        self._entries_type = entries_type
        self._prob_dist_type = prob_dist_type
        self._delta = delta
        self._normalize_error = normalize_error

    def set_matrix_type(self,
                        matrix_type: str) -> None:
        self._matrix_type = matrix_type

    def set_entries_type(self,
                         entries_type: str) -> None:
        self._entries_type = entries_type

    def set_prob_dist_type(self,
                           prob_dist_type: str) -> None:
        self._prob_dist_type = prob_dist_type

    def set_delta(self,
                  delta: float) -> None:
        self._delta = delta

    def normalize_error(self,
                        normalize: bool = True) -> None:
        self._normalize_error = normalize

    def const_dim_changing_c(self,
                             shape_a: tuple[int, int],
                             shape_b: tuple[int, int],
                             list_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f"""
        Generates two random matrices with dimensions :shape_a: and :shape_b: and runs the Basic Matrix Multiplication
         multiple times for each value in list_c standing for the number of columns sampled during selection.
         :param shape_a: shape of the A matrix
         :param shape_b: shape of the B matrix
         :param list_c: A NumPy array containing the number of columns, which should be sampled in a specific iteration
         :return: Two NumPy arrays containing the ||AB - CR||_F errors and the calculating upper bounds for these 
          values respectively.
        """
        errors = []
        bounds = []
        matrixgen = MatrixGenerator()
        matrixgen.set_matrix_type(self._matrix_type)
        matrixgen.set_entries_type(self._entries_type)

        for c in list_c:
            print(f"Calculating for c = {c}...")

            matrixgen.set_shape(shape_a)
            A = matrixgen.generate()
            matrixgen.set_shape(shape_b)
            B = matrixgen.generate()

            if self._prob_dist_type == "opt":
                prob = Distributions.get_opt_probdist_bmm(A, B)
            elif self._prob_dist_type == "nearopt":
                pass
            elif self._prob_dist_type == "nonopt":
                pass
            elif self._prob_dist_type == "uniform":
                prob = Distributions.get_uniform_probdist_bmm(shape_a[1])
            else:
                raise ValueError(f"Invalid probability distribution type: {self._prob_dist_type}! " +
                                 "Must be 'opt', 'nearopt', 'nonopt' or 'uniform'!")

            res = Algorithms.basic_matrix_mult(A, B, c=c, prob=prob)
            whp_bound = Bounds.calculate_prob_bound(A=A, B=B, c=c, delta=self._delta)
            AtimesB = A @ B
            AtimesB_norm: float = np.linalg.norm(AtimesB)
            unnormalized_error: float = np.linalg.norm(AtimesB - res, ord='fro')

            if not self._normalize_error:
                errors.append(unnormalized_error)
                bounds.append(whp_bound)
            else:
                errors.append(unnormalized_error / AtimesB_norm)
                bounds.append(whp_bound / AtimesB_norm)

        return np.array(errors), np.array(bounds)

    def changing_dim_const_c(self,
                             a_dims: np.ndarray,
                             b_dims: np.ndarray,
                             c: int) -> tuple[np.ndarray, np.ndarray]:
        f"""
        :param a_dims: A 2D NumPy array, with 2 columns and k rows. Each row represents the dimensions of the
         matrix to be generated. Has to match with row count of b_dims.
        :param b_dims: A 2D NumPy array, with 2 columns and k rows. Each row represents the dimensions of the
         matrix to be generated. Has to match with row count of a_dims.
        :param c: The number of columns to be sampled from a matrix
        :return: Two NumPy arrays containing the ||AB - CR||_F errors and the calculating upper bounds for these 
          values respectively.
        """
        assert a_dims.shape == b_dims.shape, "a_dims and b_dims don't have the same shape!"
        for i in range(a_dims.shape[0]):
            assert a_dims[i, 1] == b_dims[i, 0], (f"The dimensions of the matrices don't match!"
                                                  f"A: ({a_dims[i, 0]} x {a_dims[i, 1]}) "
                                                  f"B: ({b_dims[i, 0]} x {b_dims[i, 1]})")
        matrixgen = MatrixGenerator()
        matrixgen.set_matrix_type(self._matrix_type)
        matrixgen.set_entries_type(self._entries_type)

        errors = []
        bounds = []
        for i in range(a_dims.shape[0]):
            print(f"Calculating for A: ({a_dims[i, 0]} x {a_dims[i, 1]})   B: ({b_dims[i, 0]} x {b_dims[i, 1]})")

            matrixgen.set_shape(a_dims[i])
            A = matrixgen.generate()
            matrixgen.set_shape(b_dims[i])
            B = matrixgen.generate()

            if self._prob_dist_type == "opt":
                prob = Distributions.get_opt_probdist_bmm(A, B)
            elif self._prob_dist_type == "nearopt":
                raise NotImplementedError("Near-Opt not implemented")
            elif self._prob_dist_type == "nonopt":
                raise NotImplementedError("Non-Opt not implemented")
            elif self._prob_dist_type == "uniform":
                n = int(a_dims[i, 1])
                logging.debug(f"n = {n}")
                prob = Distributions.get_uniform_probdist_bmm(n)
            else:
                raise ValueError(f"Invalid probability distribution type: {self._prob_dist_type}! " +
                                 "Must be 'opt', 'nearopt', 'nonopt' or 'uniform'!")

            res = Algorithms.basic_matrix_mult(A, B, c=c, prob=prob)
            whp_bound = Bounds.calculate_prob_bound(A=A, B=B, c=c, delta=self._delta)
            AtimesB = A @ B
            AtimesB_norm: float = np.linalg.norm(AtimesB)
            unnormalized_error: float = np.linalg.norm(AtimesB - res, ord='fro')

            if not self._normalize_error:
                errors.append(unnormalized_error)
                bounds.append(whp_bound)
            else:
                errors.append(unnormalized_error / AtimesB_norm)
                bounds.append(whp_bound / AtimesB_norm)

        return np.array(errors), np.array(bounds)
