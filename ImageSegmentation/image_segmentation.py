# image_segmentation.py
"""Volume 1: Image Segmentation.
Jacob Francis
Vol 1 Lab
"""

import numpy as np
from scipy import linalg as la
from scipy import sparse
from imageio.v2 import imread
from matplotlib import pyplot as plt

# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    D = np.diag(np.sum(A, axis=0)) # Create the degree matrix
    L = D-A # Create the Laplacian matrix

    return L 


# Problem 2
def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    L = laplacian(A) 
    E = np.real(la.eigvals(L)) # Set of eigenvalues
    c = 0 # Number of connected components

    for lam in E:      # Find number of 0 eigenvalues
        if lam <= tol:
            c += 1

    E.sort()   # Put E in order of least to greatest
    a_c = E[1] # The algebraic connectivity is the second smallest eigenvalue

    return c, a_c


# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(int), R[mask]


# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        self.image = imread(filename)  # Read file as an image
        self.scaled = self.image / 255 # Scale the image
        if self.scaled.ndim == 3:      # If image is color compute is brightness matrix
            brightness = self.scaled.mean(axis=2)
        else: # Image is gray
            brightness = self.scaled
        self.brightness = np.ravel(brightness) # Store brightness as a 1-D array attribute

    # Problem 3
    def show_original(self):
        """Display the original image."""
        if self.scaled.ndim == 3:  # Show the original color image
            plt.imshow(self.image)
            plt.axis("off")
        else:
            plt.imshow(self.image, cmap="gray") # Show the original gray image
            plt.axis("off")
        plt.show()

    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        m,n = self.image.shape[:2]
        A = sparse.lil_matrix((m*n,m*n)) # Initialize A as an mn x mn sparse matrix
        D = np.zeros(m*n)  # Initialize D as a length mn 1-D array

        for i in range(m*n):
            J, d = get_neighbors(i,r,m,n)
            if len(J) > 0:
                B = abs(self.brightness[i]-self.brightness[J])
                w = np.exp(-(B/sigma_B2)-(d/sigma_X2))
                
        A = sparse.csc_matrix(A) # Convert A to a csc_matrix for faster computations

        return A,D

    # Problem 5
    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        L = sparse.csgraph.laplacian(A)   # Compute the Laplacian
        D_nhalf = sparse.diags(D**(-1/2)) # < These both seem wrong ^
        DLD = D_nhalf@L@D_nhalf
        eig_vec = sparse.linalg.eigsh(DLD, which='SM', k=2)
        eig_vec.reshape('m x n')


    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        raise NotImplementedError("Problem 6 Incomplete")


if __name__ == '__main__':
    dg = ImageSegmenter("dream.png")
    dg.show_original()
#     ImageSegmenter("dream.png").segment()
#     ImageSegmenter("monument_gray.png").segment()
#     ImageSegmenter("monument.png").segment()
    # A = [[0,1,0,0,1,1],[1,0,1,0,1,0],[0,1,0,1,0,0],[0,0,1,0,1,1],[1,1,0,1,0,0],[1,0,0,1,0,0]]
    # print(connectivity(A))
    

    