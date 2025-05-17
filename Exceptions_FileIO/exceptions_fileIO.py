# exceptions_fileIO.py
"""Python Essentials: Exceptions and File Input/Output.
Jacob Francis
Lab V1
26 Sept 2023
"""

from random import choice


# Problem 1
def arithmagic():
    """
    Takes in user input to perform a magic trick and prints the result.
    Verifies the user's input at each step and raises a
    ValueError with an informative error message if any of the following occur:

    The first number step_1 is not a 3-digit number.
    The first number's first and last digits differ by less than $2$.
    The second number step_2 is not the reverse of the first number.
    The third number step_3 is not the positive difference of the first two numbers.
    The fourth number step_4 is not the reverse of the third number.
    """

    step_1 = input("Enter a 3-digit number where the first and last "
                                           "digits differ by 2 or more: ")
    if len(step_1) != 3:
        raise ValueError("Input is not 3 digits long.")
    if abs(int(step_1[0])-int(step_1[2])) < 2 :
        raise ValueError("First and last digits differ by less than 2.")

    step_2 = input("Enter the reverse of the first number, obtained "
                                              "by reading it backwards: ")
    if step_2 != step_1[::-1]:
        raise ValueError("The input is not the reverse of the first number.")
    
    step_3 = input("Enter the positive difference of these numbers: ")
    if int(step_3) != abs(int(step_1)-int(step_2)):
        raise ValueError("The input is not the positive difference of the first two numbers.")
    
    step_4 = input("Enter the reverse of the previous result: ")
    if step_4 != step_3[::-1]:
        raise ValueError("The input is not the reverse of the third number.")
    print(str(step_3), "+", str(step_4), "= 1089 (ta-da!)")


# Problem 2
def random_walk(max_iters=1e12):
    """
    If the user raises a KeyboardInterrupt by pressing ctrl+c while the
    program is running, the function should catch the exception and
    print "Process interrupted at iteration $i$".
    If no KeyboardInterrupt is raised, print "Process completed".

    Return walk.
    """
    try:
        walk = 0
        directions = [1, -1]
        for i in range(int(max_iters)):
            walk += choice(directions)
    except KeyboardInterrupt:
        print("Process interrupted at iteration ",i)
    else:
        print("Process Completed.")
    finally:
        return walk
    



# Problems 3 and 4: Write a 'ContentFilter' class.
class ContentFilter(object):
    """Class for reading in file

    Attributes:
        filename (str): The name of the file
        contents (str): the contents of the file

    """
    # Problem 3
    def __init__(self, filename):
        # Loop until a valid file is entered
        while True: 

            try:
                # Read the file and store its name and contents as attributes
                with open(filename, 'r') as my_file:
                    self.filename = filename
                    self.contents = my_file.read()
                break
            # Ask for a valid file if an exception is raised 
            # and continue the loop.
            except (FileNotFoundError, TypeError, OSError):
                filename = input("Please enter a valid file name: ")
        # Set values to be used in the __str__() magic method.
        self.char = len(self.contents)  
        self.alph = sum(c.isalpha() for c in self.contents) 
        self.num  = sum(c.isdigit() for c in self.contents)  
        self.wht  = sum(c.isspace() for c in self.contents)
        self.lns  = '?'
        """ Read from the specified file. If the filename is invalid, prompt
        the user until a valid filename is given.
        """

 # Problem 4 ---------------------------------------------------------------
    def check_mode(self, mode):
        """ Raise a ValueError if the mode is invalid. """
        if mode != 'w' or 'x' or 'a':
            raise ValueError("File access incorrect.")
        
    def uniform(self, outfile, mode='w', case='upper'):
        """ Write the data to the outfile with uniform case. Include an additional
        keyword argument case that defaults to "upper". If case="upper", write
        the data in upper case. If case="lower", write the data in lower case.
        If case is not one of these two values, raise a ValueError. """
        # If case is 'upper' write the file in uppercase only.
        if case == "upper":
            outfile.write(self.contents.upper())
        # If case is 'lower' write the file in lowercase only.
        elif case == "lower":
            outfile.write(self.contents.lower())
        # Raise an error if case isn't 'upper' or 'lower'.
        else:
            raise ValueError("case is not 'upper' or 'lower'.")

    def reverse(self, outfile, mode='w', unit='line'):
        """ Write the data to the outfile in reverse order. Include an additional
        keyword argument unit that defaults to "line". If unit="word", reverse
        the ordering of the words in each line, but write the lines in the same
        order as the original file. If units="line", reverse the ordering of the
        lines, but do not change the ordering of the words on each individual
        line. If unit is not one of these two values, raise a ValueError. """
        if unit == 'word':
            rev_words = self.contents.split('')
            oufile = ''.join(rev_words)
        if unit == 'line':
            rev_lines = [l[::-1] for l in self.contents]
            outfile = '\n'.join(rev_lines)
        else:
            raise ValueError("unit can only be 'line' or 'word'.")

    def transpose(self, outfile, mode='w'):
        """ Write a transposed version of the data to the outfile. That is, write
        the first word of each line of the data to the first line of the new file,
        the second word of each line of the data to the second line of the new
        file, and so on. Viewed as a matrix of words, the rows of the input file
        then become the columns of the output file, and viceversa. You may assume
        that there are an equal number of words on each line of the input file. """
        rows = self.contents.splitlines()
        cols = [l.split(' ') for l in rows]
        r2c  = cols.join()
        
    def __str__(self):
        """ Printing a ContentFilter object yields the following output:

        Source file:            <filename>
        Total characters:       <The total number of characters in file>
        Alphabetic characters:  <The number of letters>
        Numerical characters:   <The number of digits>
        Whitespace characters:  <The number of spaces, tabs, and newlines>
        Number of lines:        <The number of lines>
        """
        # Use f strings to print the info desired.
        return (f"Source file:\t{self.filename}\n"
                f"Total characters:\t{self.char}\n"
                f"Alphabetic characters:\t{self.alph}\n"
                f"Numerical characters:\t{self.num}\n"
                f"Whitespace characters:\t{self.wht}\n"
                f"Number of lines:\t{self.lns}"
                )
if __name__ == "__main__":
    ContentFilter("a_file")