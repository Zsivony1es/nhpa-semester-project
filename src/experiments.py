import logging

import numpy as np
from helpers import Helpers
from matrix_gen import MatrixGenerator
from algos import Algorithms
from bounds import Bounds
from distributions import Distributions


class Experiments:

    @staticmethod
    def const_dim_changing_c(shape_a: tuple[int, int],
                             shape_b: tuple[int, int],
                             list_c: np.ndarray,
                             matrix_type: str = "dense",
                             entries_type: str = "frac",
                             prob_dist_type: str = "opt",
                             delta: float = 0.05,
                             normalize_error: bool = False) -> tuple[np.ndarray, np.ndarray]:
        f"""
        Generates two random matrices with dimensions :shape_a: and :shape_b: and runs the Basic Matrix Multiplication
         multiple times for each value in list_c standing for the number of columns sampled during selection.
         :param shape_a: shape of the A matrix
         :param shape_b: shape of the B matrix
         :param list_c: A NumPy array containing the number of columns, which should be sampled in a specific iteration
         :param matrix_type: The type of matrix to be generated. Options are 'dense' and 'sparse'. (Default: 'dense')
         :param entries_type: Options: 
            'frac' - generates uniformly distributed values on the interval [0,1)
            'int' - generates integer only values from 0 - 9
            'float' - generates floats from the interval [1,11)
         :param prob_dist_type: Used probability distribution type. Options are 'opt', 'nearopt', 'nonopt', 'uniform'.
          (Default: 'opt')
         :param delta: The probability that the calculated accuracy metric will be beyond the calculated bound
         :param normalize_error: If set to True, then the returned values will be divided by ||AB||_F
         :return: Two NumPy arrays containing the ||AB - CR||_F errors and the calculating upper bounds for these 
          values respectively.
        """
        errors = []
        bounds = []
        matrixgen = MatrixGenerator()
        matrixgen.set_matrix_type(matrix_type)
        matrixgen.set_entries_type(entries_type)

        for c in list_c:
            print(f"Calculating for c = {c}...")

            matrixgen.set_shape(shape_a)
            A = matrixgen.generate()
            matrixgen.set_shape(shape_b)
            B = matrixgen.generate()

            if prob_dist_type == "opt":
                prob = Distributions.get_opt_probdist_bmm(A, B)
            elif prob_dist_type == "nearopt":
                pass
            elif prob_dist_type == "nonopt":
                pass
            elif prob_dist_type == "uniform":
                prob = Distributions.get_uniform_probdist_bmm(shape_a[1])
            else:
                raise ValueError("Invalid probability distribution type! " +
                                 "Must be 'opt', 'nearopt', 'nonopt' or 'uniform'!")

            res = Algorithms.basic_matrix_mult(A, B, c=c, prob=prob)
            whp_bound = Bounds.calculate_prob_bound(A=A, B=B, c=c, delta=delta)
            AtimesB = A @ B
            AtimesB_norm: float = np.linalg.norm(AtimesB)
            unnormalized_error: float = np.linalg.norm(AtimesB - res, ord='fro')

            if not normalize_error:
                errors.append(unnormalized_error)
                bounds.append(whp_bound)
            else:
                errors.append(unnormalized_error / AtimesB_norm)
                bounds.append(whp_bound / AtimesB_norm)

        return np.array(errors), np.array(bounds)

    @staticmethod
    def changing_dim_const_c(a_dims: np.ndarray,
                             b_dims: np.ndarray,
                             c: int,
                             matrix_type: str = "dense",
                             entries_type: str = "frac",
                             prob_dist_type: str = "opt",
                             delta: float = 0.05,
                             normalize_error: bool = False) -> tuple[np.ndarray, np.ndarray]:
        f"""
        :param a_dims: A 2D NumPy array, with 2 columns and k rows. Each row represents the dimensions of the
         matrix to be generated. Has to match with row count of b_dims.
        :param b_dims: A 2D NumPy array, with 2 columns and k rows. Each row represents the dimensions of the
         matrix to be generated. Has to match with row count of a_dims.
        :param c: The number of columns to be sampled from a matrix
        :param matrix_type: The type of matrix to be generated. Options are 'dense' and 'sparse'. (Default: 'dense')
        :param entries_type: Options: 
            'frac' - generates uniformly distributed values on the interval [0,1)
            'int' - generates integer only values from 0 - 9
            'float' - generates floats from the interval [1,11)
        :param prob_dist_type: Used probability distribution type. Options are 'opt', 'nearopt', 'nonopt', 'uniform'.
          (Default: 'opt')
        :param delta: The probability that the calculated accuracy metric will be beyond the calculated bound
        :param normalize_error: If set to True, then the returned values will be divided by ||AB||_F
        :return: Two NumPy arrays containing the ||AB - CR||_F errors and the calculating upper bounds for these 
          values respectively.
        """
        assert a_dims.shape == b_dims.shape, "a_dims and b_dims don't have the same shape!"
        for i in range(a_dims.shape[0]):
            assert a_dims[i, 1] == b_dims[i, 0], (f"The dimensions of the matrices don't match!"
                                                  f"A: ({a_dims[i, 0]} x {a_dims[i, 1]}) "
                                                  f"B: ({b_dims[i, 0]} x {b_dims[i, 1]})")
        matrixgen = MatrixGenerator()
        matrixgen.set_matrix_type(matrix_type)
        matrixgen.set_entries_type(entries_type)

        errors = []
        bounds = []
        for i in range(a_dims.shape[0]):
            print(f"Calculating for A: ({a_dims[i, 0]} x {a_dims[i, 1]})   B: ({b_dims[i, 0]} x {b_dims[i, 1]})")

            matrixgen.set_shape(a_dims[i])
            A = matrixgen.generate()
            matrixgen.set_shape(b_dims[i])
            B = matrixgen.generate()

            if prob_dist_type == "opt":
                prob = Distributions.get_opt_probdist_bmm(A, B)
            elif prob_dist_type == "nearopt":
                pass
            elif prob_dist_type == "nonopt":
                pass
            elif prob_dist_type == "uniform":
                n = int(a_dims[i, 1])
                logging.debug(f"n = {n}")
                prob = Distributions.get_uniform_probdist_bmm(n)
            else:
                raise ValueError("Invalid probability distribution type! " +
                                 "Must be 'opt', 'nearopt', 'nonopt' or 'uniform'!")

            res = Algorithms.basic_matrix_mult(A, B, c=c, prob=prob)
            whp_bound = Bounds.calculate_prob_bound(A=A, B=B, c=c, delta=delta)
            AtimesB = A @ B
            AtimesB_norm: float = np.linalg.norm(AtimesB)
            unnormalized_error: float = np.linalg.norm(AtimesB - res, ord='fro')

            if not normalize_error:
                errors.append(unnormalized_error)
                bounds.append(whp_bound)
            else:
                errors.append(unnormalized_error / AtimesB_norm)
                bounds.append(whp_bound / AtimesB_norm)

        return np.array(errors), np.array(bounds)
