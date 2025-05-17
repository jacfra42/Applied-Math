"""Unit testing file svd_image_compression.py"""


import svd_image_compression as sv
import numpy as np
import numpy.linalg as nla
import pytest   

def test_compact_svd(): #problem 1
        """Unit test for the algorithm to compute the compact SVD of a matrix"""
        m = 7 #change m and n as you see fit
        n= 6
        A = np.random.randint(1,10,(m,n)).astype(float)
        U, sigma, V =  sv.compact_svd(A) 

        assert np.allclose(U@np.diag(sigma)@V, A) is True, "Incorrect truncated SVD"
        assert np.allclose(U.T @ U, np.identity(n)) is True, "U is not orthonormal"
        assert np.allclose(V.T @ V, np.identity(n)) is True, "V is not orthonormal"
        assert nla.matrix_rank(A) == len(sigma), "Number of nonzero singular values is not equal to rank of A"
        
def test_svd_approx(): #problem 3
    """Unit test for approximating the rank S SVD approximation of a matrix A"""
    A = np.random.random((3,3))
    U,sig,Vh = sv.svd_approx(A,2)
    assert nla.matrix_rank(A) == 2, "rank is correct"
    with pytest.raises(ValueError) as excinfo:
        sv.svd_approx(A,8)
    assert excinfo.value.args[0] == "s is greater than rank(A)"