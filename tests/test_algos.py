import pytest
import numpy as np

from algos import Algorithms


def test_select():
    assert (5, 7) == Algorithms.select(np.asarray([0,0,0,0,0,7]))
    assert (2, 5) == Algorithms.select(np.asarray([0,0,5,0,0,0]))

def test_opt_select():
    assert (5, 7) == Algorithms.opt_select(np.asarray([0,0,0,0,0,7]))
    assert (2, 5) == Algorithms.opt_select(np.asarray([0,0,5,0,0,0]))

def test_basic_matrix_mult():
    pass
