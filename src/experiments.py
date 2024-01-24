import numpy as np
from helpers import Helpers
from algos import Algorithms
from bounds import Bounds
from distributions import Distributions


class Experiments:
    def __init__(self):
        pass

    def constant_dimensions_changing_c(self,
                                       shape_a: tuple[int, int],
                                       shape_b: tuple[int, int],
                                       list_c: np.ndarray,
                                       matrix_type: str = "dense",
                                       prob_dist_type: str = "opt") -> tuple[np.ndarray, np.ndarray]:
        f"""
        Generates two random matrices with dimensions :shape_a: and :shape_b: and runs the Basic Matrix Multiplication
         multiple times for each value in list_c standing for the number of columns sampled during selection.
         :param shape_a: shape of the A matrix
         :param shape_b: shape of the B matrix
         :param list_c: A NumPy array containing the number of columns, which should be sampled in a specific iteration
         :param matrix_type: The type of matrix to be generated. Options are 'dense' and 'sparse'. (Default: 'dense')
         :param prob_dist_type: Used probability distribution type. Options are 'opt', 'nearopt', 'nonopt', 'uniform'.
          (Default: 'opt')
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

            prob = Distributions.get_opt_probdist_bmm(A, B)
            res = Algorithms.basic_matrix_mult(A, B, c=c, prob=prob)

            delta = 0.05
            whp_bound = Bounds.calculate_prob_bound(A=A, B=B, c=c, delta=delta)

            errors.append(np.linalg.norm(A @ B - res, ord='fro'))
            bounds.append(whp_bound)

        return np.array(errors), np.array(bounds)
