"""Volume 1: The SVD and Image Compression."""

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from imageio.v2 import imread

# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    lam, V = la.eig(A.conj().T@A) # Calculate e-values and e-vectors
    sigma = np.sqrt(lam)
    ind = np.flip(np.argsort(sigma)) # Find the order from greatest to least
    sigma = sigma[ind]   # Fancy indexing   ([V[i] for i in ind]: this is list comprehension)
    V = V[:,ind] # Fancy index the columns of V according to ind
    r = 0
    for i in sigma: # Count how many non-zero singular values there are
        if i > tol:
            r += 1
    s_1 = sigma[:r] # Keep only the positive values from sigma and V
    V_1 = V[:,:r]
    U_1 = A@V_1/s_1

    return U_1, s_1, V_1.conj().T




# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    theta = np.linspace(0,2*np.pi,200) # Create 200 theta values
    x = np.cos(theta)
    y = np.sin(theta)
    S = np.stack((x,y),axis=0) # Initialize S as a 2x200 matrix with points on the unit circle
    E = np.array([[1,0,0],[0,0,1]]) # Initialize E
    u,s,vh = la.svd(A) # Find the SVD of A
    s = np.diag(s) # Turn singular values into a matrix

    ax1 = plt.subplot(221)  # Plot S and E
    ax1.plot(S[0],S[1],'b')
    ax1.plot(E[0],E[1],'r-')
    plt.axis('equal')
    ax2 = plt.subplot(222)  # Plot vhS and vhE
    ax2.plot((vh@S)[0],(vh@S)[1],'b')
    ax2.plot((vh@E)[0],(vh@E)[1],'r-')
    plt.axis('equal')
    ax3 = plt.subplot(223)  # Plot svhS and svhE
    ax3.plot((s@vh@S)[0],(s@vh@S)[1],'b')
    ax3.plot((s@vh@E)[0],(s@vh@E)[1],'r-')
    plt.axis('equal')
    ax4 = plt.subplot(224)  # Plot usvhS and usvhE
    ax4.plot((u@s@vh@S)[0],(u@s@vh@S)[1],'b')
    ax4.plot((u@s@vh@E)[0],(u@s@vh@E)[1],'r-')
    plt.axis('equal')

    plt.show()

# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    if s > np.linalg.matrix_rank(A):
        raise ValueError("s is greater than rank(A)") # Raise error if s is greater than the rank of A
    
    U,sig,Vh = la.svd(A, full_matrices = False) # Compute the compact SVD of A
    U_1  = U[:,:s]         # Take only the mxs portion of U
    sig_1  = np.diag(sig[:s])  # Take the first s values and make it a diag matrix
    Vh_1 = Vh[:s,:]        # Take only the sxn values of Vh
    T = U_1@sig_1@Vh_1      # Create the truncated SVD matrix
    T_size = U_1.size + Vh_1.size + sig[:s].size # Find the number of entries (with sigmas stored as a 1-D array)
    
    return T, T_size


# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    U,sig,Vh = la.svd(A, full_matrices = False) # Compute the compact SVD 

    if sig[np.argmin(sig)] > err:
        raise ValueError("Chosen error is not greater than the smallest sigma value.")
    
    sigs = np.where(sig < err) # Create an array of the indexes of all sigma values less than err
    
    s = sigs[0][0] # Use the greatest value less than err as the s value
    
    U_1  = U[:,:s]         # Take only the mxs portion of U
    sig_1  = np.diag(sig[:s])  # Take the first s values and make it a diag matrix
    Vh_1 = Vh[:s,:]        # Take only the sxn values of Vh
    T = U_1@sig_1@Vh_1      # Create the truncated SVD matrix
    T_size = U_1.size + Vh_1.size + sig[:s].size # Find the number of entries (with sigmas stored as a 1-D array)
    
    return T, T_size


# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    image = imread(filename)/255  # Send values to (0,1)
    # Gray Images
    if len(image.shape) == 2:
        M, M_size = svd_approx(image, s) # Compute the best rank-s approximation and its number of entries
        # Plot the original image and the approximation
        ax1 = plt.subplot(121)
        ax1.imshow(image, cmap="gray")
        plt.axis("off")
        ax2 = plt.subplot(122)
        ax2.imshow(M, cmap="gray")
        plt.axis("off")
    # Color images
    else:
        R = image[:,:,0] # Initialize red matrix
        G = image[:,:,1] # Initialize green matrix
        B = image[:,:,2] # Initilaize blue matrix
        # Find the approximations for each color
        Rs, R_size = svd_approx(R,s)
        Gs, G_size = svd_approx(G,s)
        Bs, B_size = svd_approx(B,s)
        M = np.dstack((Rs,Gs,Bs)) # Put the colors back into one array
        M = np.clip(M,0,1) # Clip entries that are less than 0 or greater than 1
        M_size = R_size+G_size+B_size # Compute the number of entries in the approximation
        # Plot the original image and the approximation
        ax1 = plt.subplot(121)
        ax1.imshow(image)
        plt.axis("off")
        ax2 = plt.subplot(122)
        ax2.imshow(M)
        plt.axis("off")
    diff = image.size-M_size # Find the number of entries saved
    plt.suptitle(f"Saved {diff} entries")
    plt.show()


if __name__ == '__main__':
    A = np.random.random((5,5))
    # U,s,Vh = la.svd(A, full_matrices=False)
    # Ux,sx,Vhx = compact_svd(A)
    # print(U,Ux)
    # B = np.array([[3,1],[1,3]])
    print(lowest_rank_approx(A,1))

    