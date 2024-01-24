import numpy as np
from helpers import Helpers
from algos import Algorithms
from bounds import Bounds
from distributions import Distributions


class Experiments:

    @staticmethod
    def const_dim_changing_c(shape_a: tuple[int, int],
                             shape_b: tuple[int, int],
                             list_c: np.ndarray,
                             matrix_type: str = "dense",
                             prob_dist_type: str = "opt",
                             delta: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        f"""
        Generates two random matrices with dimensions :shape_a: and :shape_b: and runs the Basic Matrix Multiplication
         multiple times for each value in list_c standing for the number of columns sampled during selection.
         :param shape_a: shape of the A matrix
         :param shape_b: shape of the B matrix
         :param list_c: A NumPy array containing the number of columns, which should be sampled in a specific iteration
         :param matrix_type: The type of matrix to be generated. Options are 'dense' and 'sparse'. (Default: 'dense')
         :param prob_dist_type: Used probability distribution type. Options are 'opt', 'nearopt', 'nonopt', 'uniform'.
          (Default: 'opt')
         :param delta: The probability that the calculated accuracy metric will be beyond the calculated bound
         :return: Two NumPy arrays containing the ||AB - CR||_F errors and the calculating upper bounds for these 
          values respectively.
        """
        errors = []
        bounds = []

        for c in list_c:
            print(f"Calculating for c = {c}...")
            A, B = Helpers.generate_matrices(shape_a=shape_a,
                                             shape_b=shape_b,
                                             matrix_type=matrix_type)

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

            errors.append(np.linalg.norm(A @ B - res, ord='fro'))
            bounds.append(whp_bound)

        return np.array(errors), np.array(bounds)

    @staticmethod
    def changing_dim_const_c(a_dims: np.ndarray,
                             b_dims: np.ndarray,
                             c: int,
                             matrix_type: str = "dense",
                             prob_dist_type: str = "opt",
                             delta: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        f"""
        :param a_dims: A 2D NumPy array, with 2 columns and k rows. Each row represents the dimensions of the
         matrix to be generated. Has to match with row count of b_dims.
        :param b_dims: A 2D NumPy array, with 2 columns and k rows. Each row represents the dimensions of the
         matrix to be generated. Has to match with row count of a_dims.
        :param c: The number of columns to be sampled from a matrix
        :param matrix_type: The type of matrix to be generated. Options are 'dense' and 'sparse'. (Default: 'dense')
        :param prob_dist_type: Used probability distribution type. Options are 'opt', 'nearopt', 'nonopt', 'uniform'.
          (Default: 'opt')
        :param delta: The probability that the calculated accuracy metric will be beyond the calculated bound
        :return: Two NumPy arrays containing the ||AB - CR||_F errors and the calculating upper bounds for these 
          values respectively.
        """
        assert a_dims.shape == b_dims.shape, "a_dims and b_dims don't have the same shape!"
        for i in range(a_dims.shape[0]):
            assert a_dims[i, 1] == b_dims[i, 0], (f"The dimensions of the matrices don't match!"
                                                  f"A: ({a_dims[i, 0]} x {a_dims[i, 1]}) "
                                                  f"B: ({b_dims[i, 0]} x {b_dims[i, 1]})")

        errors = []
        bounds = []
        for i in range(a_dims.shape[0]):
            print(f"Calculating for A: ({a_dims[i, 0]} x {a_dims[i, 1]})   B: ({b_dims[i, 0]} x {b_dims[i, 1]})")
            c = 50

            A, B = Helpers.generate_matrices(shape_a=a_dims[i],
                                             shape_b=b_dims[i],
                                             matrix_type=matrix_type)

            if prob_dist_type == "opt":
                prob = Distributions.get_opt_probdist_bmm(A, B)
            elif prob_dist_type == "nearopt":
                pass
            elif prob_dist_type == "nonopt":
                pass
            elif prob_dist_type == "uniform":
                prob = Distributions.get_uniform_probdist_bmm(a_dims[i, 1])
            else:
                raise ValueError("Invalid probability distribution type! " +
                                 "Must be 'opt', 'nearopt', 'nonopt' or 'uniform'!")

            res = Algorithms.basic_matrix_mult(A, B, c=c, prob=prob)

            delta = 0.05
            whp_bound = Bounds.calculate_prob_bound(A=A, B=B, c=c, delta=delta)

            errors.append(np.linalg.norm(A @ B - res, ord='fro'))
            bounds.append(whp_bound)

        return np.array(errors), np.array(bounds)
