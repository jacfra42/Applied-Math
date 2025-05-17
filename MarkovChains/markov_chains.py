# markov_chains.py
"""Volume 2: Markov Chains.
<Name>
<Class>
<Date>
"""

import numpy as np
from numpy import linalg as la


class MarkovChain:
    """A Markov chain with finitely many states.

    Attributes:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.
        dict (dict): matches label to index
    """
    # Problem 1
    def __init__(self, A, states=None):
        """Check that A is column stochastic and construct a dictionary
        mapping a state's label to its index (the row / column of A that the
        state corresponds to). Save the transition matrix, the list of state
        labels, and the label-to-index dictionary as attributes.

        Parameters:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.

        Raises:
            ValueError: if A is not square or is not column stochastic.

        Example:
            >>> MarkovChain(np.array([[.5, .8], [.5, .2]], states=["A", "B"])
        corresponds to the Markov Chain with transition matrix
                                   from A  from B
                            to A [   .5      .8   ]
                            to B [   .5      .2   ]
        and the label-to-index dictionary is {"A":0, "B":1}.
        """
        if np.allclose(A.sum(axis=0), np.ones(A.shape[1])): # Make sure the matrix is column stochastic
            self.A = A # Save A as an attribute
            if states == None: 
                self.states = [*range(len(A[0]))] # Make the states 0,..,n-1
                self.dict = {self.states[i]: range(len(self.states))[i] for i in range(len(self.states))} # Save dictionary as an attribute
            else:
                self.states = states # Save states given as an attribute
                self.dict = {states[i]: range(len(states))[i] for i in range(len(states))} # Save dictionary as an attribute
        else:
            raise ValueError("Matrix is not column stochastic") # Raise error if A is not column stochastic

    # Problem 2
    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.

        Parameters:
            state (str): the label for the current state.

        Returns:
            (str): the label of the state to transitioned to.
        """
        n = self.dict[state] # Find the corresponding column
        probs = self.A[:,n]  # Set probs as the array that contains the probabilities of moving to a state
        new_state = np.argmax(np.random.multinomial(1, probs)) # Find the index of the new state, 
        
        return self.states[new_state]                         # determined by a draw from a categorical distributuion 

    # Problem 3
    def walk(self, start, N):
        """Starting at the specified state, use the transition() method to
        transition from state to state N-1 times, recording the state label at
        each step.

        Parameters:
            start (str): The starting state label.

        Returns:
            (list(str)): A list of N state labels, including start.
        """
        list_states = [start] # Initialize list of states
        cur_state = start     # Set current state as start
        
        for i in range(N-1): # Change state N-1 times
            new_state = self.transition(cur_state) # Find new state
            list_states.append(new_state)          # Add new state to list of states
            cur_state = new_state                  # Change current state
        
        return list_states

    # Problem 3
    def path(self, start, stop):
        """Beginning at the start state, transition from state to state until
        arriving at the stop state, recording the state label at each step.

        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.

        Returns:
            (list(str)): A list of state labels from start to stop.
        """
        list_states = [start] # Initialize list of states
        cur_state = start     # Set current state as start
        
        while stop not in list_states: # Stop when 'stop' state is reached
            new_state = self.transition(cur_state) # Find new state
            list_states.append(new_state)          # Add new state to list of states
            cur_state = new_state                  # Change current state
        
        return list_states
    # Problem 4
    def steady_state(self, tol=1e-12, maxiter=40):
        """Compute the steady state of the transition matrix A.

        Parameters:
            tol (float): The convergence tolerance.
            maxiter (int): The maximum number of iterations to compute.

        Returns:
            ((n,) ndarray): The steady state distribution vector of A.

        Raises:
            ValueError: if there is no convergence within maxiter iterations.
        """
        x_0 = np.ones(len(self.A[0]))/len(self.A[0]) # Generate an initial state distribution vector
        x_k = x_0
        x = self.A@x_0 # Calculate the next x
        k = 1
        while la.norm(x_k-x) >= tol: # Continue if ||x_k-x|| is greater than the tolerance
            x_k = x         # Set the next iteration of values
            x = self.A@x_k
            k += 1 # Count number of iterations
            if k > maxiter: # Raise error if we iterate too many times
                raise ValueError("A^k does not converge")
        
        return x

class SentenceGenerator(MarkovChain):
    """A Markov-based simulator for natural language.

    Attributes:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.
        dict (dict): matches label to index
    """
    # Problem 5
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        with open(filename, 'r') as file: # Read the file line by line
            contents = file.readlines()
        set_words = [] # Words
        list_sents = [] # Sentences
        for line in contents:    # Create a list of words and sentences
            sents = line.split()
            list_sents.append(sents)
            for word in sents:
                set_words.append(word)
        set_words = set(set_words)
        self.states = ["$tart"]+list(set_words)+["$top"] # Add '$tart' and '$top'
        n = len(self.states)
        self.A = np.zeros((n,n)) # Initialize the attribute A
        self.dict = {self.states[i]: range(len(self.states))[i] for i in range(len(self.states))} # Save dictionary as an attribute

        for s in list_sents:         # Fill the matrix A
            s = ["$tart"]+s+["$top"]
            for i in range(len(s)):
                if i+1 < len(s):
                    self.A[self.dict[s[i+1]],self.dict[s[i]]] += 1
        self.A[n-1,n-1] = 1 # Make '$top'only go to itself
        col_sums = self.A.sum(axis=0) # Normalize the columns
        self.A = self.A / col_sums

    # Problem 6
    def babble(self):
        """Create a random sentence using MarkovChain.path().

        Returns:
            (str): A sentence generated with the transition matrix, not
                including the labels for the $tart and $top states.

        Example:
            >>> yoda = SentenceGenerator("yoda.txt")
            >>> print(yoda.babble())
            The dark side of loss is a path as one with you.
        """
        s = self.path("$tart","$top") # Use path to create a sentnce (list of words)
        s.remove("$tart")
        s.remove("$top")
        return ' '.join(str(i) for i in s) # Return it as a string
    

if __name__ == '__main__':
    J = MarkovChain(np.array([[.5, .8], [.5, .2]]), states=["A", "B"])
    F = MarkovChain(np.array([[.5,.3,.1,0],[.3,.3,.3,.3],[.2,.3,.4,.5],[0,.1,.2,.2]]), states = ["hot","mild","cold","freezing"])
    A = np.array([[.7,.6],[.3,.4]])
    M = SentenceGenerator("yoda.txt")
    print(M.babble())
    