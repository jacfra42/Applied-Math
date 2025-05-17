# breadth_first_search.py
"""Volume 2: Breadth-First Search.
Jacob Francis
Vol 2 Lab
2 Nov 2023
"""


# Problems 1-3
from collections import deque
import networkx as nx

class Graph:
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a set of
    the corresponding node's neighbors.

    Attributes:
        d (dict): the adjacency dictionary of the graph.
    """
    def __init__(self, adjacency={}):
        """Store the adjacency dictionary as a class attribute"""
        self.d = dict(adjacency)

    def __str__(self):
        """String representation: a view of the adjacency dictionary."""
        return str(self.d)

    # Problem 1
    def add_node(self, n):
        """Add n to the graph (with no initial edges) if it is not already
        present.

        Parameters:
            n: the label for the new node.
        """
        # If n is in adjacency don't update it, if not add it 
        # as a key with an empty set as its value
        if n not in self.d.keys():
            self.d.update({n:set()})

    # Problem 1
    def add_edge(self, u, v):
        """Add an edge between node u and node v. Also add u and v to the graph
        if they are not already present.

        Parameters:
            u: a node label.
            v: a node label.
        """
        # Add nodes
        self.add_node(u)
        self.add_node(v)
        # Access the key and add an edge to its value set
        self.d[u].add(v)
        self.d[v].add(u)

    # Problem 1
    def remove_node(self, n):
        """Remove n from the graph, including all edges adjacent to it.

        Parameters:
            n: the label for the node to remove.

        Raises:
            KeyError: if n is not in the graph.
        """
        # Remove n
        self.d.pop(n)
        # For every key remove n from their value sets
        for key in self.d:
            self.d[key].discard(n) # Discard will not raise error if n is not in the set

    # Problem 1
    def remove_edge(self, u, v):
        """Remove the edge between nodes u and v.

        Parameters:
            u: a node label.
            v: a node label.

        Raises:
            KeyError: if u or v are not in the graph, or if there is no
                edge between u and v.
        """
        # Remove v from the value set of u, and u from v (Will raise appropriate errors as asked)
        self.d[u].remove(v)
        self.d[v].remove(u) 

    # Problem 2
    def traverse(self, source):
        """Traverse the graph with a breadth-first search until all nodes
        have been visited. Return the list of nodes in the order that they
        were visited.

        Parameters:
            source: the node to start the search at.

        Returns:
            (list): the nodes in order of visitation.

        Raises:
            KeyError: if the source node is not in the graph.
        """
        V = []        # List of nodes visited
        Q = deque([]) # Queue of nodes to be visited
        M = set()     # Set of nodes that have been, or that are marked to be, visited

        # Raise error if source is not in graph
        if source not in self.d.keys():
            raise KeyError("source is not in the graph")
        # Add source to Q and M
        Q.append(source)
        M.add(source)
        # While Q is not empty, perform a BFS
        while len(Q) != 0: 
            current = Q.popleft()
            V.append(current)
            for i in self.d[current]:
                if i not in M:
                    Q.append(i)
                    M.add(i)

        return V

    # Problem 3
    def shortest_path(self, source, target):
        """Begin a BFS at the source node and proceed until the target is
        found. Return a list containing the nodes in the shortest path from
        the source to the target, including endoints.

        Parameters:
            source: the node to start the search at.
            target: the node to search for.

        Returns:
            A list of nodes along the shortest path from source to target,
                including the endpoints.

        Raises:
            KeyError: if the source or target nodes are not in the graph.
        """
        D = dict({})  # Create dictionary, Key: visited node, Value: visiting node
        Q = deque([]) # Queue of nodes to be visited
        M = set()     # Set of nodes that have been, or that are marked to be, visited

        # Raise error if source is not in graph
        if source not in self.d.keys():
            raise KeyError("source is not in the graph")
        # Raise error if target is not in graph
        if target not in self.d.keys():
            raise KeyError("target is not in the graph")

        # Add source to Q and M
        Q.append(source)
        M.add(source)
        # Until target is found, do a BFS
        while target not in M: 
            current = Q.popleft() 
            for i in self.d[current]:
                if i not in M:
                    Q.append(i)
                    M.add(i)
                    D[i] = current
        path = [target]
        node = D[target]
        while node is not source:
            path.append(node)
            node = D[node]
        if source != target:
            path.append(source)   # Fix it up

        s_path = path[::-1]

        return s_path


# Problems 4-6
class MovieGraph:
    """Class for solving the Kevin Bacon problem with movie data from IMDb."""

    # Problem 4
    def __init__(self, filename="movie_data.txt"):
        """Initialize a set for movie titles, a set for actor names, and an
        empty NetworkX Graph, and store them as attributes. Read the speficied
        file line by line, adding the title to the set of movies and the cast
        members to the set of actors. Add an edge to the graph between the
        movie and each cast member.

        Each line of the file represents one movie: the title is listed first,
        then the cast members, with entries separated by a '/' character.
        For example, the line for 'The Dark Knight (2008)' starts with

        The Dark Knight (2008)/Christian Bale/Heath Ledger/Aaron Eckhart/...

        Any '/' characters in movie titles have been replaced with the
        vertical pipe character | (for example, Frost|Nixon (2008)).
        """
        self.movies = set()
        self.actors = set()
        self.nx_graph = nx.Graph()

        with open(filename, 'r') as file: # Get file
            file_contents = file.read()
        movies_list = file_contents.split('\n') # Make a list where each element is a movie with its actors
    # For each movie make the title and actors individual elememts
        list_names = []
        for movie in movies_list: 
            names = movie.split('/') 
            list_names.append(names)
    # For each movie pop the title and add it to a list of all the titles
        titles = []
        m=0
        for m in list_names:
            title = m.pop(0)
            titles.append(title)
        self.movies = set(titles) # Initialize movie title set as an attribute
        
        set_actors = set()
        for names in list_names:
            set_actors.update(names)
        self.actors = set_actors # Initialize actors set as an attribute
        
        for name in list_names:
            self.nx_graph.add_edge(names[0],name)



    # Problem 5
    def path_to_actor(self, source, target):
        """Compute the shortest path from source to target and the degrees of
        separation between source and target.

        Returns:
            (list): a shortest path from source to target, including endpoints and movies.
            (int): the number of steps from source to target, excluding movies.
        """
        raise NotImplementedError("Problem 5 Incomplete")

    # Problem 6
    def average_number(self, target):
        """Calculate the shortest path lengths of every actor to the target
        (not including movies). Plot the distribution of path lengths and
        return the average path length.

        Returns:
            (float): the average path length from actor to target.
        """
        raise NotImplementedError("Problem 6 Incomplete")

if __name__ == '__main__':
    G = MovieGraph()
    print(list(G.actors)[0:7])