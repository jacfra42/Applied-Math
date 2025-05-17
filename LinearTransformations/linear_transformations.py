# linear_transformations.py
"""Volume 1: Linear Transformations.
<Name>
<Class>
<Date>
"""

from random import random
import numpy as np
from matplotlib import pyplot as plt
import time

data = np.load("horse.npy") # Extract horse array

# Problem 1
def stretch(A, a, b):
    """Scale the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    return (np.array([[a,0],[0,b]]))@A  # Stretch an array A

def shear(A, a, b):
    """Slant the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    return (np.array([[1,a],[b,1]]))@A  # Slant an array A

def reflect(A, a, b):
    """Reflect the points in A about the line that passes through the origin
    and the point (a,b).

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): x-coordinate of a point on the reflecting line.
        b (float): y-coordinate of the same point on the reflecting line.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    x = (1/(a*a+b*b))*np.array([[a*a-b*b,2*a*b],[2*a*b,b*b-a*a]])  # Reflect an array A 
    return x@A

def rotate(A, theta):
    """Rotate the points in A about the origin by theta radians.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        theta (float): The rotation angle in radians.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    R = np.array([[np.cos(theta),-(np.sin(theta))],[np.sin(theta),np.cos(theta)]])  # Rotate an array A by theta radians
    return R@A


# Problem 2
def solar_system(T, x_e, x_m, omega_e, omega_m):
    """Plot the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (x_e,0) and the initial
    position of the moon is (x_m,0).

    Parameters:
        T (float): The final time.
        x_e (float): The earth's initial x coordinate.
        x_m (float): The moon's initial x coordinate.
        omega_e (float): The earth's angular velocity.
        omega_m (float): The moon's angular velocity.
    """
    # Initialize an array for the earth's and moon's initial position
    pe_0 = np.array([x_e,0])
    pm_0 = np.array([x_m-x_e,0])
    # Initialize lists for the positions of the earth and moon
    p_e = []
    p_m = []
    # Fill the position arrays with 1000 values from 0 to time T
    for t in np.linspace(0,T, 1000):
        p_et = rotate(pe_0,t*omega_e)
        p_mt = rotate(pm_0, t*omega_m) 
        p_e.append(p_et)
        p_m.append(p_mt)
    # Turn the position lists into arrays
    p_e = np.array(p_e)
    p_m = np.array(p_m) + p_e

    plt.plot(p_e[:,0],p_e[:,1], 'b', label = "Earth")       # Plot the earth positions
    plt.plot(p_m[:,0],p_m[:,1], 'orange', label = "Moon")   # Plot the moon positions
    plt.axis("equal")                                       # Make the plot look nice (:
    plt.gca().set_aspect("equal")
    plt.legend(loc="lower right")
    plt.show()
    




def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in range(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in range(n)] for i in range(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p) ]
                                    for i in range(m) ]

# Problem 3
def prob3():
    """Use time.time(), timeit.timeit(), or %timeit to time
    matrix_vector_product() and matrix-matrix-mult() with increasingly large
    inputs. Generate the inputs A, x, and B with random_matrix() and
    random_vector() (so each input will be nxn or nx1).
    Only time the multiplication functions, not the generating functions.

    Report your findings in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times. Choose a domain for n so
    that your figure accurately describes the growth, but avoid values of n
    that lead to execution times of more than 1 minute.
    """
    # Initialize the domain and sets for the time values
    dom = 2**np.arange(1,9) 
    mvtimes = []
    mmtimes = []
    # Time the matrix_vector_product for all n in the domian
    # Save the times to a set
    for n in dom:
        A = random_matrix(n)
        x = random_vector(n)
        start = time.time()
        matrix_vector_product(A,x)
        mvtimes.append(time.time()-start)
    # Time the matrix_matrix_product for all n in the domian
    # Save the times to a set
    for k in dom:
        A = random_matrix(k)
        B = random_matrix(k)
        start = time.time()
        matrix_matrix_product(A,B)
        mmtimes.append(time.time()-start)
    # Plot the matrix_vector_product values in the left subplot
    ax1 = plt.subplot(121)
    ax1.plot(dom, mvtimes, 'b.-')
    plt.axis([0, 250,0, .007])
    plt.title("Matrix-Vector Multiplication")
    plt.xlabel("n")
    plt.ylabel("Seconds")
    # Plot the matrix_matrix_product values in the right subplot
    ax2 = plt.subplot(122)
    ax2.plot(dom, mmtimes, 'orange')
    plt.axis([0,250,0,2.5])
    plt.title("Matrix-Matrix Multiplication")
    plt.xlabel("n")

    plt.show()


# Problem 4
def prob4():
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot().

    Report your findings in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """
    # Initialize the domain and sets for the time values
    dom = 2**np.arange(1,10)
    mvtimes = []
    mmtimes = []
    Vtimes = []
    Mtimes = []
    # Time the matrix_vector_product for all n in the domian
    # Save the times to a set
    for n in dom:
        A = random_matrix(n)
        x = random_vector(n)
        start = time.time()
        matrix_vector_product(A,x)
        mvtimes.append(time.time()-start)
    # Time the matrix_matrix_product for all n in the domian
    # Save the times to a set
    for k in dom:
        A = random_matrix(k)
        B = random_matrix(k)
        start = time.time()
        matrix_matrix_product(A,B)
        mmtimes.append(time.time()-start)
    # Time the numpy matrix-vector product for all n in the domian
    # Save the times to a set
    for j in dom:
        A = np.array(random_matrix(j))
        x = np.array(random_vector(j))
        start=time.time()
        A@x
        Vtimes.append(time.time()-start)
    # Time the numpy matrix-matrix product for all n in the domian
    # Save the times to a set
    for m in dom:
        A = np.array(random_matrix(m))
        B = np.array(random_matrix(m))
        start=time.time()
        A@B
        Mtimes.append(time.time()-start)
    # Plot the values on the left subplot
    ax1 = plt.subplot(121)
    ax1.plot(dom, mvtimes, 'b', label = 'matrix-vector')
    ax1.plot(dom, mmtimes, 'orange', label = 'matrix-matrix')
    ax1.plot(dom, Vtimes, 'r', label = 'numpy MxV')
    ax1.plot(dom, Mtimes, 'k', label = 'numpy MxM')
    plt.axis([0, 350,0, 0.05])
    plt.xlabel("n")
    plt.ylabel("Seconds")
    plt.legend(loc="upper right")

    # Plot the values, with a logarithmic scale, in the right subplot
    ax2 = plt.subplot(122)
    ax2.loglog(dom, mmtimes, 'orange', base=2, label='matrix-matrix')
    ax2.loglog(dom, mvtimes, 'b', base=2, label="matrix-vector")
    ax2.loglog(dom, Vtimes, 'r' , base=2, label='numpy MxV')
    ax2.loglog(dom, Mtimes, 'k', base=2, label='numpy MxM')
    plt.legend(loc="upper right")
    
    plt.show()

if __name__ == "__main__":
    prob4()