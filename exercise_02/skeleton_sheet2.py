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
    print("Calculating")  # replace this with your code
    return 5  # replace this with your code


def std(x):
    """Calculate the standard deviation for an array-like object x."""
    # here goes your code
    print("Calculating")  # replace this with your code
    return 5  # replace this with your code


def variance(x):
    """Calculate the variance for an array-like object x."""
    # here goes your code
    print("Calculating")  # replace this with your code
    return 5  # replace this with your code


def mean_uncertainty(x):
    """Calculate the uncertainty in the mean for an array-like object x."""
    # here goes your code
    print("Calculating")  # replace this with your code
    return 5  # replace this with your code


def ex1():
    data = np.loadtxt("ironman.txt")
    age = 2010 - data[:, 1]

    # a)
    mean_age = mean(age)
    mean_age_uncertainty = mean_uncertainty(age)
    # .2f means that the number is printed with two decimals. Check if that makes sense
    print(f"The mean age of the participants is {mean_age:.2f} +/- {mean_age_uncertainty:.2f} years.")


def ex2():
    radiation = np.loadtxt("radiation.txt")


if __name__ == '__main__':
    ex1()
    # ex2()  # uncomment to run ex2
