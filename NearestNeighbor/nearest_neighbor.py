# nearest_neighbor.py
"""Volume 2: Nearest Neighbor Search.
Jacob Francis
Vol 2 Lab
2 Nov 2023
"""

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from scipy import stats


# Problem 1
def exhaustive_search(X, z):
    """Solve the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    d_min = np.min(np.linalg.norm(X-z, axis=1))        # Find the minimum euclidean norm
    x_min = X[np.argmin(np.linalg.norm(X-z, axis=1))]  # Find the vector that corresponds to the d_min

    return x_min, d_min

# Problem 2: Write a KDTNode class.
class KDTNode:
    """Node class for K-D Trees.

    Attributes:
        left (KDTNode): a reference to this node's left child.
        right (KDTNode): a reference to this node's right child.
        value ((k,) ndarray): a coordinate in k-dimensional space.
        pivot (int): the dimension of the value to make comparisons on.
    """
    def __init__(self, x):
        if isinstance(x, np.ndarray):
            self.left = None
            self.right = None
            self.value = x
            self.pivot = None
        else:
            raise TypeError("x is not of type np.ndarray") # Raise error if x is not an array

# Problems 3 and 4
class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    # Problem 3
    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
            ValueError: if data is already in the tree
        """
        def find_parent(current):
            # Raise error for duplicates
            if (new.value == current.value).all():
                raise ValueError("node already in tree")
            # New node belongs in left subtree
            if new.value[current.pivot] < current.value[current.pivot]:
                # Attach to parent if no left child exists
                if current.left == None:
                    current.left = new
                    new.pivot = current.pivot + 1
                    if new.pivot == self.k:
                        new.pivot = 0
                # If there is a left child, check that node
                else:
                    find_parent(current.left)
            # New node belongs in right subtree
            if new.value[current.pivot] >= current.value[current.pivot]:
                # Attach to parent if no right child exists
                if current.right == None:
                    current.right = new
                    new.pivot = current.pivot + 1
                    if new.pivot == self.k:
                        new.pivot = 0
                # If there is a right child, check that node
                else:
                    find_parent(current.right)
        # Insert as root if the tree is empty
        if self.root == None:
            new = KDTNode(data)
            new.pivot = 0
            self.root = new
            self.k = len(data)
        # Raise error if data is not of the appropriate length
        elif len(data) != self.k:
            raise ValueError("data is not in R^k")
        # Insert a new node to a non-empty tree
        else:
            new = KDTNode(data)
            find_parent(self.root)

            

    # Problem 4
    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        # Follow Algorithm 1
        def KD_search(current, nearest, d):
            # Base Case
            if current is None:
                return nearest, d
            
            x = current.value
            i = current.pivot
            # Check if 'current' is closer than 'nearest'
            if la.norm(x-z) < d:
                nearest = current
                d = la.norm(x-z)
            # Search to the left
            if z[i] < x[i]:
                nearest, d = KD_search(current.left, nearest, d)
                # Serach right if needed
                if z[i]+d >= x[i]:
                    nearest, d = KD_search(current.right, nearest, d)
            # Search to the right
            else:
                nearest, d = KD_search(current.right, nearest, d)
                # Search left if needed
                if z[i]-d <= x[i]:
                    nearest, d = KD_search(current.left, nearest, d)

            return nearest, d
        # Start at the root
        node, d = KD_search(self.root, self.root, la.norm(self.root.value-z))

        return node.value, d


    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)


# Problem 5: Write a KNeighborsClassifier class.
class KNeighborsClassifier:
    """A k-nearest neighbors classifier that uses SciPy's KDTree to solve
    the nearest neighbor problem efficiently.
    """
    # Accept an integer and save it as n_neighbors attribute
    def __init__(self, n_neighbors):
        if isinstance(n_neighbors, int): 
            self.k = n_neighbors
        else:
            raise TypeError("n_neighbor is not an integer") # Raise error if not an integer
    # Accept a training set X and training labels y
    # and save them as tree and labels attributes respectively
    def fit(self, X, y):
        self.tree = KDTree(X)
        self.labels = y
    # Query the tree to find the most common labels
    def predict(self, z):
        distances, indices = self.tree.query(z, k=self.k)
        return stats.mode(self.labels[indices])[0][0]

# Problem 6
def prob6(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """
    # Import the data
    data    = np.load("mnist_subset.npz")
    X_train = data["X_train"].astype(np.float64)   # Training data
    y_train = data["y_train"]                    # Training labels
    X_test  = data["X_test"].astype(np.float64)    # Test data
    y_test  = data["y_test"]                     # Test labels
    # Load a classifier and fit the data
    kn = KNeighborsClassifier(n_neighbors) 
    kn.fit(X_train,y_train)
    # Find the accuracy
    count = 0
    for i in range(len(y_test)):
        labels = kn.predict(X_test[i])
        if labels==y_test[i]:
            count += 1

    return count/len(y_test)

if __name__=="__main__":
    print(prob6(4))

