# object_oriented.py
"""Python Essentials: Object Oriented Programming.
Jacob Francis
V1 Lab
12/11/2023

"""


from operator import truediv



class Backpack:
    """A Backpack object class. Has a name, color, max_size, and a list of contents.

    Attributes:
        name (str): the name of the backpack's owner.
        color (str): the color of the backpack.
        max_size (int): the number of items allowed in the backpack.
        contents (list): the contents of the backpack.

    """

    # Problem 1: Modify __init__() and put(), and write dump().
    def __init__(self, name, color, max_size=5):
        """Set the name, color, and max_size and initialize an empty list of contents.

        Parameters:
            name (str): the name of the backpack's owner.
            color (str): the color of the backpack.
            max_size (int): the number of items allowed in the backpack.
        """
        self.name = name
        self.color = color
        self.max_size = max_size
        self.contents = []

    def put(self, item):
        """Add an item to the backpack's list of contents or let them know the list of contents is full when it is full"""
        if len(self.contents) < self.max_size:
            self.contents.append(item)
        else:
            print("No Room!")

    def take(self, item):
        """Remove an item from the backpack's list of contents."""
        self.contents.remove(item)

    def dump(self):
        """Remove all items from the backpack's list of contents"""
        self.contents = []

    # Magic Methods -----------------------------------------------------------

    # Problem 3: Write __eq__() and __str__().
    def __add__(self, other):
        """Add the number of contents of each Backpack."""
        return len(self.contents) + len(other.contents)

    def __lt__(self, other):
        """Compare two backpacks. If 'self' has fewer contents
        than 'other', return True. Otherwise, return False.
        """
        return len(self.contents) < len(other.contents)
    
    def __eq__(self, other):
        """Compare two backpacks. If 'self' and 'other' have 
        the same name, color, and number of contents, return true.
        Otherwise, return false.
        """
        return self.name == other.name and self.color == other.color and len(self.contents) == len(other.contents)
    
    def __str__(self):
         """Print the name, color, number of contents, max size, and list of contents of the Backpack object."""
         return "Owner:\t\t" + self.name + "\n" + "Color:\t\t" + self.color + '\nSize:\t\t' + str(len(self.contents)) + '\nMax Size:\t' + str(self.max_size) + '\nContents:\t' + str(self.contents)
        
        


# An example of inheritance. You are not required to modify this class.
class Knapsack(Backpack):
    """A Knapsack object class. Inherits from the Backpack class.
    A knapsack is smaller than a backpack and can be tied closed.

    Attributes:
        name (str): the name of the knapsack's owner.
        color (str): the color of the knapsack.
        max_size (int): the maximum number of items that can fit inside.
        contents (list): the contents of the backpack.
        closed (bool): whether or not the knapsack is tied shut.
    """
    def __init__(self, name, color):
        """Use the Backpack constructor to initialize the name, color,
        and max_size attributes. A knapsack only holds 3 item by default.

        Parameters:
            name (str): the name of the knapsack's owner.
            color (str): the color of the knapsack.
            max_size (int): the maximum number of items that can fit inside.
        """
        Backpack.__init__(self, name, color, max_size=3)
        self.closed = True

    def put(self, item):
        """If the knapsack is untied, use the Backpack.put() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.put(self, item)

    def take(self, item):
        """If the knapsack is untied, use the Backpack.take() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.take(self, item)

    def weight(self):
        """Calculate the weight of the knapsack by counting the length of the
        string representations of each item in the contents list.
        """
        return sum(len(str(item)) for item in self.contents)


# Problem 2: Write a 'Jetpack' class that inherits from the 'Backpack' class.

class Jetpack(Backpack):
    """A Jetpack object class. Inherits from the Backpack class.
    A Jetpack is smaller than a backpack and has a fuel tank.

    Attributes:
        name (str): the name of the Jetpack's owner.
        color (str): the color of the Jetpack.
        max_size (int): the maximum number of items that can fit inside.
        contents (list): the contents of the backpack.
        fuel (int): amount of fuel in the fuel tank.
    """
    def __init__(self, name, color, max_size=2, fuel=10):
        """Use the Backpack constructor to initialize the name, color,
        and max_size attributes. A Jetpack only holds 2 item and 10 units of fuel by default.

        Parameters:
            name (str): the name of the Jetpack's owner.
            color (str): the color of the Jetpack.
            max_size (int): the maximum number of items that can fit inside.
            fuel (int): amount of fuel in the fuel tank.

        """

        Backpack.__init__(self, name, color, max_size)
        self.fuel = fuel
    
    def fly(self, burn):
        """"Accepts an amount of fuel to be burned and decrements the appropriate amount,
            or if not enough fuel available prints "Not enough fuel!"
        """
        if burn <= self.fuel:
            self.fuel = self.fuel - burn
        else:
            print("Not enough fuel!")

    def dump(self):
        """Empties the contents of the jetpack and the fuel tank.
        """
        self.contents = []
        self.fuel = 0
    

# Problem 4: Write a 'ComplexNumber' class.

class ComplexNumber:
    """ Createa ComplexNumber object class. Has a real and imaginary part.
    
    Attributes: 
        real (int): the real part of a complex number
        imag (int): the imaginary part of a complex number
        """

    # Initialize real and imag
    def __init__(self, real, imag):
        
        self.real = real
        self.imag = imag
        
    #Accepts the real and imaginary parts of a complex number and returns the cojugate.
    def conjugate(self):
        i = -1*self.imag
        conj = ComplexNumber(self.real, i)
        return conj
    
    # Initializes print() of the ComplexNumber object to return the complex number as formatted
    def __str__(self):
        if self.imag >= 0:
            return "(" + str(self.real) + '+' + str(self.imag) + 'j)'
        else:
            return "(" + str(self.real) + str(self.imag) + 'j)'

    # Returns the absolute value of the complex number
    def __abs__(self):
        return  (self.real**2 + self.imag**2)**(1/2)
    
    # Returns True if two complex numbers are the same and false if not
    def __eq__(self, other):
        if self.real == other.real and self.imag == other.imag:
            return True
        else: 
            return False 

    # Changes '+' to add two complex numbers 
    def __add__(self, other):
        return ComplexNumber(self.real + other.real, self.imag + other.imag)
    
    # Changes '-' to subtract two complex numbers
    def __sub__(self, other):
        return ComplexNumber(self.real - other.real, self.imag - other.imag)
    
    # Changes '*' to multiply two complex numbers
    def __mul__(self, other):
        return ComplexNumber(self.real*other.real-self.imag*other.imag, self.real*other.imag+self.imag*other.real)
    
    # Changes '/' to divide two complex numbers
    def __truediv__(self, other):
        real_Part = ((self.real*other.real) + (self.imag*other.imag))/((other.real*other.real)+(other.imag*other.imag))
        imag_Part = ((self.imag*other.real)-(self.real*other.imag))/((other.real*other.real)+(other.imag*other.imag))
        return ComplexNumber(real_Part,imag_Part)
    


def test_ComplexNumber(a, b):
    py_cnum, my_cnum = complex(a, b), ComplexNumber(a, b)
    # Validate the constructor.
    if my_cnum.real != a or my_cnum.imag != b:
        print("__init__() set self.real and self.imag incorrectly")

    # Validate conjugate() by checking the new number's imag attribute.
    if py_cnum.conjugate().imag != my_cnum.conjugate().imag:
        print("conjugate() failed for", py_cnum)


if __name__ == "__main__":
    test_ComplexNumber(2,-3)
