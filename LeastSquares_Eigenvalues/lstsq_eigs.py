# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
Jacob Francis
Vol 1 Lab
31 Oct 2023
"""

# (Optional) Import functions from your QR Decomposition lab.
# import sys
# sys.path.insert(1, "../QR_Decomposition")
# from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg

import xxlimited
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la
import cmath


# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    Q, R = la.qr(A, mode="economic")  # Find Q and R with the reduced QR decomposition
    x = la.solve_triangular(R, Q.T@b) # Solve for x 

    return x

# Problem 2
def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    # Load tha data
    data = np.load("housing.npy")
    m = data.shape[0]
    # Create A and b matrices
    A = np.column_stack((data[:,0], np.ones(m)))
    b = data[:,1]
    # Find x
    x = least_squares(A, b)
    dom = np.arange(17)
    # Plot the data points and the least squares line
    plt.plot(data[:,0],data[:,1], 'ko', label="Data Points")
    plt.plot(dom, x[0]*dom + x[1] , 'r-', label="Least Squares Line")
    plt.xlabel("Year")
    plt.ylabel("Price Index")
    plt.legend()
    plt.show()



# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    # Load tha data
    data = np.load("housing.npy")
    # Create b matrix
    b = data[:,1]
    # Find x for degrees 3,6,9,12
    x_3 = la.lstsq(np.vander(data[:,0],4), b)[0]
    f_3 = np.poly1d(x_3)
    x_6 = la.lstsq(np.vander(data[:,0],7), b)[0]
    f_6 = np.poly1d(x_6)
    x_9 = la.lstsq(np.vander(data[:,0],10), b)[0]
    f_9 = np.poly1d(x_9)
    x_12 = la.lstsq(np.vander(data[:,0],13), b)[0]
    f_12 = np.poly1d(x_12)

    # Plot each least squares line and plot it in a subplot with the data points
    dom = np.linspace(0,16,33)
    ax1 = plt.subplot(221)
    ax1.plot(dom, data[:,1], 'ko', label="Data Points")
    ax1.plot(dom, f_3(dom), 'r-', label="deg 3")
    plt.legend()
    ax2 = plt.subplot(222)
    ax2.plot(dom, data[:,1], 'ko', label="Data Points")
    ax2.plot(dom, f_6(dom), 'r-', label="deg 6")
    plt.legend()
    ax3 = plt.subplot(223)
    ax3.plot(dom, data[:,1], 'ko', label="Data Points")
    ax3.plot(dom, f_9(dom), 'r-', label="deg 9")
    plt.legend()
    ax4 = plt.subplot(224)
    ax4.plot(dom, data[:,1], 'ko', label="Data Points")
    ax4.plot(dom, f_12(dom), 'r-', label="deg 12")
    plt.legend()

    plt.show()

def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")

# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    x,y = np.load("ellipse.npy").T             # Load data points into x and y lists
    A   = np.column_stack((x**2,x,x*y,y,y**2)) # Initialize A of the least squares equation
    B   = np.ones_like(x)                      # Initialize B of the least squares equation
    a,b,c,d,e = la.lstsq(A,B)[0]               # Find the solutions 
    plot_ellipse(a, b, c, d, e)                # Plot the ellipse
    plt.plot(x,y,'ko')                         # Plot the data points
    plt.show()


# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    # Follow algorithm 1
    m,n = A.shape
    x = np.random.random(n)
    x = x/la.norm(x)

    for k in range(N):  #Loop through the maximum number of iterations given
        x_k = A@x
        x_k = x_k/la.norm(x_k)
        if la.norm(x_k-x) < tol: # Break if less than the tolerance given
            break
        x = x_k
    
    return x.T@A@x, x




# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    # Follow Algorithm 2
    m,n = A.shape
    S = la.hessenberg(A) # Make S the Hessenberg form of A

    for k in range(N):
        Q,R = la.qr(S)
        S = R@Q
    eigs = []
    i=0
    while i<n:
        S_i = S[i:i+2,i:i+2]
        if S_i.shape == (1,1): # If S_i is 1x1
            eigs.append(S_i[0][0])  # Add S_i as an eigenvalue
        elif S_i.shape == (2,2): # If S_i is 2x2
            a,b,c,d = S_i[0][0], S_i[0][1],S_i[1][0],S_i[1][1] # Use quadratic formula to find eigenvalues
            B = a+d
            C = a*d - b+c

            eigs.append((B + (cmath.sqrt((B**2)-4*C)))/2)
            eigs.append((B - (cmath.sqrt((B**2)-4*C)))/2)

            i = i+1
        i = i+1

    return eigs


if __name__ == "__main__":
    # A = np.random.random((2,2))
    # B = A+A.T
    # print(la.eig(B))
    # print("My equation:")
    # print(qr_algorithm(B))
    polynomial_fit()
    

