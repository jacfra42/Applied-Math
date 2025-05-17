# matplotlib_intro.py
"""Python Essentials: Intro to Matplotlib.
<Name>
<Class>
<Date>
"""
import numpy as np
from matplotlib import pyplot as plt


# Problem 1
def var_of_means(n):
    """ Create an (n x n) array of values randomly sampled from the standard
    normal distribution. Compute the mean of each row of the array. Return the
    variance of these means.

    Parameters:
        n (int): The number of rows and columns in the matrix.

    Returns:
        (float) The variance of the means of each row.
    """
    rand_arr = np.random.normal(size=(n,n))
    mn_arr = np.mean(rand_arr,axis=1)
    var = np.var(mn_arr)
    return var

def prob1():
    """ Create an array of the results of var_of_means() with inputs
    n = 100, 200, ..., 1000. Plot and show the resulting array.
    """
    array = []
    """ Append array to include the results of var_of_means(). """
    for i in range(100, 1100, 100):
        array.append(var_of_means(i))
    """ Plot the array. """
    plt.plot(np.array(array))

    return plt.show()


# Problem 2
def prob2():
    """ Plot the functions sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi]. Make sure the domain is refined enough to produce a figure
    with good resolution.
    """
    # Define the domain x.
    x = np.linspace(-2*np.pi, 2*np.pi, 75)
    #Define sin and plot it.
    sin = np.sin(x)
    plt.plot(x, sin)
    #Define cos and plot it.
    cos = np.cos(x)
    plt.plot(x, cos)
    #Define arctan nd plot it.
    arctan = np.arctan(x)
    plt.plot(x, arctan)

    return plt.show() #Show the figure.


# Problem 3
def prob3():
    """ Plot the curve f(x) = 1/(x-1) on the domain [-2,6].
        1. Split the domain so that the curve looks discontinuous.
        2. Plot both curves with a thick, dashed magenta line.
        3. Set the range of the x-axis to [-2,6] and the range of the
           y-axis to [-6,6].
    """
    # Plot the function from [-2,1)
    x1 = np.linspace(-2,1,50)
    y1 = 1/(x1-1)
    plt.plot(x1, y1, 'm--',lw=4)
    # Plot the function from (1,6]
    x2 = np.linspace(1,6,50)
    y2 = 1/(x2-1)
    plt.plot(x2,y2, 'm--',lw=4)
    # Set the range of the x and y axes
    plt.xlim(-2,6)
    plt.ylim(-6,6)

    
    return plt.show()


# Problem 4
def prob4():
    """ Plot the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi], each in a separate subplot of a single figure.
        1. Arrange the plots in a 2 x 2 grid of subplots.
        2. Set the limits of each subplot to [0, 2pi]x[-2, 2].
        3. Give each subplot an appropriate title.
        4. Give the overall figure a title.
        5. Use the following line colors and styles.
              sin(x): green solid line.
             sin(2x): red dashed line.
             2sin(x): blue dashed line.
            2sin(2x): magenta dotted line.
    """
    # Set the domain x
    x = np.linspace(0, 2*np.pi, 100)
    # Make a subplot for sin(x)
    ax1 = plt.subplot(221)
    ax1.plot(x, np.sin(x),'g-')
    plt.axis([0,2*np.pi,-2,2])
    plt.title("sin(x)")
    # Make a subplot for sin(2x)
    ax2 = plt.subplot(222)
    ax2.plot(x, np.sin(2*x),'r--')
    plt.axis([0,2*np.pi,-2,2])
    plt.title("sin(2x)")
    # Make a subplot for 2sin(x)
    ax3 = plt.subplot(223)
    ax3.plot(x, 2*np.sin(x),'b--')
    plt.axis([0,2*np.pi,-2,2])
    plt.title("2sin(x)")
    # Make a subplot for 2sin(2x)
    ax4 = plt.subplot(224)
    ax4.plot(x, 2*np.sin(2*x),'m:')
    plt.axis([0,2*np.pi,-2,2])
    plt.title("2sin(2x)")

    plt.suptitle("Sine Functions")

    return plt.show()


# Problem 5
def prob5():
    """ Visualize the data in FARS.npy. Use np.load() to load the data, then
    create a single figure with two subplots:
        1. A scatter plot of longitudes against latitudes. Because of the
            large number of data points, use black pixel markers (use "k,"
            as the third argument to plt.plot()). Label both axes.
        2. A histogram of the hours of the day, with one bin per hour.
            Label and set the limits of the x-axis.
    """
    # Retrieve the dat
    my_data = np.load("FARS.npy")

    # Set x as the longitude column and y as the lattitude column
    x = my_data[:,1]
    y = my_data[:,2]

    # Create a scatter plot in the first subplot and title it
    ax1 = plt.subplot(121)
    ax1.plot(x,y, 'k,')
    plt.axis("equal")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Create a histogram of the hours of the day
    z = my_data[:,0] 
    ax2 = plt.subplot(122)
    ax2.hist(z, bins=24, range=[0,24])
    plt.xlabel("Hour")

    return plt.show()


# Problem 6
def prob6():
    """ Plot the function g(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi].
        1. Create 2 subplots: one with a heat map of g, and one with a contour
            map of g. Choose an appropriate number of level curves, or specify
            the curves yourself.
        2. Set the limits of each subplot to [-2pi, 2pi]x[-2pi, 2pi].
        3. Choose a non-default color scheme.
        4. Include a color scale bar for each subplot.
    """
    # Create a 2-D domain with x and y
    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    y = np.linspace(-2*np.pi, 2*np.pi, 100)
    X, Y = np.meshgrid(x,y)
    # Define the function Z
    Z = (np.sin(X)*np.sin(Y))/(X*Y)
    # Create a heat map in the first subplot
    plt.subplot(121)
    plt.pcolormesh(X,Y,Z, cmap = "seismic")
    plt.colorbar()
    # Set the limit of the subplot
    plt.xlim(-2*np.pi, 2*np.pi)
    plt.ylim(-2*np.pi, 2*np.pi)
    # Create a contour map in the second subplot
    plt.subplot(122)
    plt.contour(X,Y,Z, 10, cmap = "ocean")
    plt.colorbar()
    # Set the limit of the subplot
    plt.xlim(-2*np.pi, 2*np.pi)
    plt.ylim(-2*np.pi, 2*np.pi)

    return plt.show()

if __name__ == "__main__":
    prob4()