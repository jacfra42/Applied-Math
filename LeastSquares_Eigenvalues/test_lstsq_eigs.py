"""Unit testing file for Least Squares and Computing Eigenvalues problem 6"""


import lstsq_eigs
import pytest
import numpy as np
from scipy import linalg as la

def test_qr_algorithm():
    """
    Write at least one unit test for problem 6, the qr algorithm function.
    """
    A = np.random.random((3,3))
    B = A+A.T  #Create a random symmetric matrix
    assert la.eig(B) == lstsq_eigs.qr_algorithm(B), "Incorrect eigenvalues"

def test_power_method():
    #Sets up test cases
    A = np.array([[1,1],[1,1]])
    B = np.array([[1,0,0],[0,2,0],[0,0,3]])
    C = np.array([[2,2],[1,3]])
    
    Aval, Avec = lstsq_eigs.power_method(A)
    Bval, Bvec = lstsq_eigs.power_method(B)
    Cval, Cvec = lstsq_eigs.power_method(C)
    
    #Checks if it finds the appropriate eigenvalue
    assert abs(Aval - 2) < 1e-5, "Incorrect eigenvalue"
    assert abs(Bval - 3) < 1e-5, "Incorrect eigenvalue"
    assert abs(Cval - 4) < 1e-5, "Incorrect eigenvalue"
    
    #Checks if it finds an eigenvector that works
    assert np.linalg.norm(A @ Avec - Aval * Avec) < 1e-3, "Incorrect vector"
    assert np.linalg.norm(B @ Bvec - Bval * Bvec) < 1e-3, "Incorrect vector"
    assert np.linalg.norm(C @ Cvec - Cval * Cvec) < 1e-3, "Incorrect vector"