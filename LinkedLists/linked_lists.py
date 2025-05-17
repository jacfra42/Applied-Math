# linked_lists.py
"""Volume 2: Linked Lists.
Jacob Francis
Vol 2 Lab
5 Oct 2023
"""


# Problem 1
class Node:
    """A basic node class for storing data if it is int, float, or str."""
    def __init__(self, data):
        """Store the data in the value attribute.
                
        Raises:
            TypeError: if data is not of type int, float, or str.
        """
        if type(data) in [int, float, str]:
            self.value = data
        else:
            raise TypeError("data is not int, float, or str.")


class LinkedListNode(Node):
    """A node class for doubly linked lists. Inherits from the Node class.
    Contains references to the next and previous nodes in the linked list.
    """
    def __init__(self, data):
        """Store the data in the value attribute and initialize
        attributes for the next and previous nodes in the list.
        """
        Node.__init__(self, data)       # Use inheritance to set self.value.
        self.next = None                # Reference to the next node.
        self.prev = None                # Reference to the previous node.


# Problems 2-5
class LinkedList:
    """Doubly linked list data structure class.

    Attributes:
        head (LinkedListNode): the first node in the list.
        tail (LinkedListNode): the last node in the list.
    """
    def __init__(self):
        """Initialize the head and tail attributes by setting
        them to None, since the list is empty initially.
        """
        # Create head, tail, and length attributes
        self.head = None
        self.tail = None
        self.length = 0

    def append(self, data):
        """Append a new node containing the data to the end of the list."""
        # Create a new node to store the input data.
        new_node = LinkedListNode(data)
        if self.head is None:
            # If the list is empty, assign the head and tail attributes to
            # new_node, since it becomes the first and last node in the list.
            self.head = new_node
            self.tail = new_node
            self.length = 1
        else:
            # If the list is not empty, place new_node after the tail.
            self.tail.next = new_node               # tail --> new_node
            new_node.prev = self.tail               # tail <-- new_node
            # Now the last node in the list is new_node, so reassign the tail.
            self.tail = new_node
            self.length += 1

    # Problem 2
    def find(self, data):
        """Return the first node in the list containing the data.

        Raises:
            ValueError: if the list does not contain the data.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.find('b')
            >>> node.value
            'b'
            >>> l.find('f')
            ValueError: <message>
        """
        if self.length == 0:
            raise ValueError("list does not contain the data")
        
        if self.head.value == data:   # If data is the head return head
            return self.head
        
        node = self.head.next         #Start after the head, iterate through the linked list until you find the data.
        while node != None:           # If data not found raise an error.
            if node.value == data:
                return node
            else:
                node = node.next

        raise ValueError("list does not contain the data")

    # Problem 2
    def get(self, i):
        """Return the i-th node in the list.

        Raises:
            IndexError: if i is negative or greater than or equal to the
                current number of nodes.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.get(3)
            >>> node.value
            'd'
            >>> l.get(5)
            IndexError: <message>
        """
        if (i < 0) or (i >= self.length+1):                      # If index is negative or too big, raise an error
            raise IndexError("index must be in the list")
        else:                                                  # Start at the head, move to the ith node and return the node
            node = self.head
            while i != 0:
                node = node.next
                i -= 1
        return node

    # Problem 3
    def __len__(self):
        """Return the number of nodes in the list.

        Examples:
            >>> l = LinkedList()
            >>> for i in (1, 3, 5):
            ...     l.append(i)
            ...
            >>> len(l)
            3
            >>> l.append(7)
            >>> len(l)
            4
        """
        # Since we have a length attribute all we do is return the length (since we count each insertion or removal)
        return self.length

    # Problem 3
    def __str__(self):
        """String representation: the same as a standard Python list.

        Examples:
            >>> l1 = LinkedList()       |   >>> l2 = LinkedList()
            >>> for i in [1,3,5]:       |   >>> for i in ['a','b',"c"]:
            ...     l1.append(i)        |   ...     l2.append(i)
            ...                         |   ...
            >>> print(l1)               |   >>> print(l2)
            [1, 3, 5]                   |   ['a', 'b', 'c']
        """
        node = self.head
        # Add each node value to a list and return the string representation
        list = []
        while node != None:
            list.append(node.value)
            node = node.next

        return repr(list)

            

    # Problem 4
    def remove(self, data):
        """Remove the first node in the list containing the data.

        Raises:
            ValueError: if the list is empty or does not contain the data.

        Examples:
            >>> print(l1)               |   >>> print(l2)
            ['a', 'e', 'i', 'o', 'u']   |   [2, 4, 6, 8]
            >>> l1.remove('i')          |   >>> l2.remove(10)
            >>> l1.remove('a')          |   ValueError: <message>
            >>> l1.remove('u')          |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.remove(10)
            ['e', 'o']                  |   ValueError: <message>
        """
        target = self.find(data)    # Find target
        if target == self.head:     # If target is the head, remove it and update head
            next = self.head.next
            self.head.next = None
            self.head = next
            self.length -= 1
        elif target == self.tail:           # If target is tail, remove it and update tail
                self.tail = self.tail.prev
                self.tail.next = None
                self.length -= 1
        else:                               # Iterate through list, when target is found adjust links for 
            node = self.head                # previous and next terms. Delete links for the target.
            while node != target:
                node = node.next
            if node == target:
                p = node.prev
                p.next = node.next
                n = node.next
                n.prev = p
                node.next = None
                node.prev = None
                self.length -= 1
            
                


    # Problem 5
    def insert(self, index, data):
        """Insert a node containing data into the list immediately before the
        node at the index-th location.

        Raises:
            IndexError: if index is negative or strictly greater than the
                current number of nodes.

        Examples:
            >>> print(l1)               |   >>> len(l2)
            ['b']                       |   5
            >>> l1.insert(0, 'a')       |   >>> l2.insert(6, 'z')
            >>> print(l1)               |   IndexError: <message>
            ['a', 'b']                  |
            >>> l1.insert(2, 'd')       |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.insert(1, 'a')
            ['a', 'b', 'd']             |   IndexError: <message>
            >>> l1.insert(2, 'c')       |
            >>> print(l1)               |
            ['a', 'b', 'c', 'd']        |
        """
        if index < 0 or index > self.length:           # raise error if index is out of range
            raise IndexError("Index is out of range")
        
        new_node = LinkedListNode(data)
                                                      # Inserting at the head
        if self.get(index) == self.head:
            new_node.prev = None
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
            self.length += 1
        elif index == self.length:           # Inserting at the tail, just use append()
            self.append(data)
        else:                               # Inserting anywhere else in the linked list
            node = self.head
            while node.next != None:
                node = node.next
            if self.get(index) == node:
                new_node.prev = node.prev
                new_node.next = node
                p = node.prev
                p.next = new_node
                node.prev = new_node
                self.length += 1


# Problem 6: Deque class.
class Deque:
    def __init__(self):
        LinkedList.__init__(self)  
    # Above: Inherit LinkedList class
    # Below: Add pop() attribute
    def pop(self):
        node = self.tail
        self.tail = self.tail.prev
        self.tail.next = None
        self.length -= 1
        return node.value
    # Add popleft() attribute
    def popleft(self):
        node = self.head
        next = self.head.next
        self.head.next = None
        self.head = next
        self.length -= 1
        return node
    # Use insert() from LinkedList class to appendleft()
    def appendleft(self, data):
        LinkedList.insert(self,0,data)
    # Remove remove() and insert() attributes
    def remove(*args, **kwargs):
        raise NotImplementedError("Use pop() or popleft() for removal")

    def insert(*args, **kwargs):
        raise NotImplementedError("Use appendleft() or append() for inserting")


# Problem 7
def prob7(infile, outfile):
    """Reverse the contents of a file by line and write the results to
    another file.

    Parameters:
        infile (str): the file to read from.
        outfile (str): the file to write to.
    """
    raise NotImplementedError("Problem 7 Incomplete")
