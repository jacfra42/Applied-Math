# qr_decomposition.py
"""Volume 1: The QR Decomposition.
<Name>
<Class>
<Date>
"""
import numpy as np
from scipy import linalg as la

# Problem 1
def qr_gram_schmidt(A):
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """
    # Find the shape of A and initialize Q and R
    m,n = A.shape 
    Q = np.copy(A)
    R = np.zeros((n,n))
    # Per Algorithm 1, normalize the ith column of Q, then orthogonalize the i+1 column of Q
    for i in range(n):
        R[i,i] = la.norm(Q[:,i])
        Q[:,i] = Q[:,i]/R[i,i]
        for j in range(i+1,n):
            R[i,j] = Q[:,j].T @ Q[:,i]
            Q[:,j] = Q[:,j]-R[i,j]*Q[:,i]

    return Q,R


# Problem 2
def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the determinant of A.
    """
    Q,R = la.qr(A, mode="economic")      # Find the QR decomposition
    det_A = np.abs(np.prod(np.diag(R)))  # The absolute value of det(A) is equal to the absolute value of the 
                                         # product of the diagonal entries of R by (8.1)
    return det_A


# Problem 3
def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """
    Q,R = la.qr(A, mode="economic")      # Find the QR decomposition
    y = Q.T @ b                          # Calculate y=Q^T*b
    x = np.copy(y)                       
    # Use back substitution to solve Rx=y for x
    for k in range(x.shape[0]-1,-1,-1):
        x[k] = 1/R[k,k]*(y[k]-np.sum(R[k,k+1:]*x[k+1:])) 
    return x 


# Problem 4
def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    # Find the shape of A and initialize R and Q
    m,n = A.shape
    R = np.copy(A)
    Q = np.eye(m)
    # Follow the loop given in Algorithm 2 to find Q^T and R
    for k in range(n):
        u = np.copy(R[k:,k])                            
        sign = lambda x: 1 if x >= 0 else -1
        u[0] = u[0] + sign(u[0])*la.norm(u)
        u = u/(la.norm(u))
        R[k:,k:] = R[k:,k:]-2*np.outer(u, u.T @ R[k:,k:])
        Q[k:,:]  = Q[k:,:]-2*np.outer(u, u.T @ Q[k:,:])

    return Q.T, R


# Problem 5
def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """
    # Find the shape of A and initialize H and Q
    m,n = A.shape
    H = np.copy(A)
    Q = np.eye(m)
    # Follow Algorithm 3 to find H and Q^T
    for k in range(n-2):
        u = np.copy(H[k+1:,k])
        sign = lambda x: 1 if x >= 0 else -1
        u[0] = u[0] + sign(u[0])*la.norm(u)
        u = u/(la.norm(u))
        H[k+1:,k:] = H[k+1:,k:] - 2*np.outer(u, u.T @ H[k+1:,k:])
        H[:,k+1:]  = H[:,k+1:]  - 2*np.outer(H[:,k+1:] @ u, u.T)
        Q[k+1:,:]  = Q[k+1:,:]  - 2*np.outer(u, u.T @ Q[k+1:,:])

    return H, Q.T

if __name__ == "__main__":
    """A = np.random.random((8,8))
    b = np.random.random(8)
    Q,R = la.qr(A)
    print(A.shape, Q.shape, R.shape)
    Q_2,R_2 = qr_householder(A)
    print(Q_2.shape, R_2.shape)
    print(np.allclose(Q_2 @ R_2, A))
    print(la.det(A))
    print(abs_det(A))
    H,Q = hessenberg(A)
    print(np.allclose(np.triu(H,-1),H))
    print(np.allclose(Q @ H @ Q.T, A))
    print(la.det(A))
    x = solve(A,b)
    print(np.allclose(A@x,b))"""


    