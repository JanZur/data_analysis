import numpy as np
import matplotlib.pyplot as plt


def mean(x):
    """Calculate the mean for an array-like object x.

    Parameters
    ----------
    x : array-like
        Array-like object containing the data.

    Returns
    -------
    mean : float
        The mean of the data.
    """
    # here goes your code


    mean = np.sum(x, axis=0)/len(x)
    numpy_mean = np.mean(x)
    return mean


def std(x):
    """Calculate the standard deviation for an array-like object x."""
    # here goes your code
    std = np.sqrt(variance(x))
    numpy_std = np.std(x)

    return std



def variance(x):
    """Calculate the variance for an array-like object x."""
    # here goes your code
    var = np.sum(np.power(x - mean(x),2)) / (len(x)-1)
    # with or without minus one numy does it without
    numpy_var = np.var(x)
    return var  # replace this with your code


def mean_uncertainty(x):
    """Calculate the uncertainty in the mean for an array-like object x."""
    # here goes your code
    mean_uncertainty = std(x)
    return mean_uncertainty  # replace this with your code


def ex1():
    data = np.loadtxt("ironman.txt")
    age = 2010 - data[:, 1]

    # a)
    mean_age = mean(age)
    mean_age_uncertainty = mean_uncertainty(age)
    # .2f means that the number is printed with two decimals. Check if that makes sense

    print(f"The mean age of the participants is {mean_age:.1f} +/- {mean_age_uncertainty:.1f} years.")


def ex2():
    radiation = np.loadtxt("radiation.txt")


if __name__ == '__main__':
    ex1()
    # ex2()  # uncomment to run ex2
