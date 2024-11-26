"""Skeleton for Data Analysis Fall 2024 sheet 6.

This sheet helps to structure the code for the exercises of sheet 6.
It shows idioms and best practices for writing code in Python.

You can directly use this sheet and modify it.

(It is not guaranteed to be bug free)
"""
import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt



def load_data(filename):
    """Load the data from a file.

    Args:
        filename (str): The name of the file to load.

    Returns:
        array: The data as a numpy array.
    """
    return np.loadtxt(filename)


def current_ohmslaw(U, R):
    r"""Calculate the current according to Ohm's Law given the voltage U and resistance R.

    Ohm's Law states that the current is given by:

    .. math::

        I = \frac{U}{R}

    Args:
        U (float, array): The measured voltage.
        R (float, array): The resistance.

    Returns:
        float or array: Value of the linear function. Shape is the broadcast shape of
            the inputs.
    """
    return U / R


def current_ohmslaw_bias(U, R, bias=None):
    """Calculate the current according to Ohm's Law given the voltage U and resistance R with a bias.

    Ohm's Law states that the current is given by:

    .. math::

        I = \frac{U}{R}

    We can add a bias to the current by adding a constant to the voltage:

    .. math::

        I = \frac{U + bias}{R}

    Args:
        U (float, array): The measured voltage.
        R (float, array): The resistance.
        bias (float, array): The bias to add to the voltage. If None, no bias is added.

    Returns:
        float or array: Value of the linear function. Shape is the broadcast shape of
            the inputs.
    """
    if bias is None:
        bias = 0  # with this, we can also use the function without bias.
    return (U + bias) / R


def chi2(x, y, err):
    """Calculate the chi2 statistic for a dataset and its predictions.

    Args:
        x (array): The first data set.
        y (array): Predicted values for the first data set.
        err (array): The error on the measurements of the first data set.

    Returns:
        float: The chi2 statistic.
    """
    difference = np.array(np.abs(x - y))
    squared = np.square(difference)
    divided = squared  / np.square(err)
    sum = np.sum(divided)

    return sum


def optimal_resistance_analytical(measurements, uncertainties):
    """Calculate the optimal resistance analytically."""
    data = measurements
    voltage = data[:, 0]
    current = data[:, 1]

    analytical_resistance = (np.mean(current * voltage) - np.mean(current) * np.mean(voltage)) / (
            np.mean(current * current) - np.mean(current) ** 2)
    return analytical_resistance


def ex_1a(measurements):
    """Run exercise 1a."""
    plt.figure()  # ALWAYS create a new figure before plotting.
    plt.plot(measurements[:, 0], measurements[:, 1], "o", label="Data")
    plt.xlabel("Voltage [V]")
    plt.ylabel("Current [A]")
    plt.xlim(0, 15)
    plt.ylim(0, 10)
    plt.savefig("ex1a.png")
    # plt.show()
    plt.close()


# This is an example of creating 1b composing different functions together.
def chi2_1b(R, measurements, uncertainties):
    """Calculate chi2 in dependence of the resistance."""

    # Here is your code for exercise 1b.
    data = measurements

    voltage = data[:, 0]
    current = data[:, 1]
    # current_pred = current_ohmslaw(voltage, R)
    current_pred = current_ohmslaw(voltage, R)
    chi2val = chi2(current, current_pred, uncertainties)
    return chi2val


def ex_1c(measurements, uncertainties):
    """Run exercise 1c."""

    # Here is your code for exercise 1c.
    resistances = np.linspace(0.1, 10, num=1000)

    # start, stop, number of steps
    chi2val = np.zeros(len(resistances))
    for i, R in enumerate(resistances):
        chi2val[i] = chi2_1b(R, measurements, uncertainties)

    print(np.argmin(chi2val))
    optimal_resistance = resistances[np.argmin(chi2val)]

    # plot the chi2 value as a function of the resistance.
    plt.figure()
    plt.plot(resistances, chi2val)
    plt.xlabel("Resistance [Ohm]")
    plt.ylabel("chi^2")
    plt.xlim(0, 10)
    plt.ylim(0, 1700)
    plt.savefig("ex1c.png")
    #plt.show()
    plt.close()
    print(
        f"The best fit for R is {optimal_resistance:.2f} + - {resistances[1] - resistances[0]:.2f} Ohm.")


def ex_1d(measurements, uncertainties):
    """Run exercise 1d."""
    optimal_analytical_resistance = optimal_resistance_analytical(measurements, uncertainties)
    print(f"The optimal analytical calculated resistance is {optimal_analytical_resistance:.2f} Ohm.")
    print("The values is not the same as the one optained from the plotting method, this is due to the facct that the "
          "plotting assumes a linear relation without a bias. and therefore is off by some small amount because the "
          "measurement has a small bias.")


def ex_1e(measurements, measurements_uncertainties):
    # calculate the uncertainty on the analytical resistance

    optimal_analytical_resistance = optimal_resistance_analytical(measurements, measurements_uncertainties)
    uncertainty_analytical_resistance = 1 / np.sqrt(np.sum(1 / measurements_uncertainties ** 2))
    print(
        f"The optimal analytical calculated resistance is {optimal_analytical_resistance:.2f} +- {uncertainty_analytical_resistance:.2f} Ohm.")
    # get the uncertainty from the graph information
    # when is chi2 = chi2 + 1
    # get the resistances
    resistances = np.linspace(1, 5, num=1000)
    chi2val = np.zeros(len(resistances))
    for i, R in enumerate(resistances):
        chi2val[i] = chi2_1b(R, measurements, measurements_uncertainties)

    min_chi2 = np.min(chi2val)
    min_chi2_index = np.argmin(chi2val)
    chi2_plus_1 = min_chi2 + 1
    chi2_plus_1_index = np.argmin(np.abs(chi2val - chi2_plus_1))
    resistance_plus_1 = resistances[chi2_plus_1_index]
    uncertainty_resistance = np.abs(resistance_plus_1 - resistances[min_chi2_index])
    chi_minus_1 = min_chi2 - 1
    chi_minus_1_index = np.argmin(np.abs(chi2val - chi_minus_1))
    resistance_minus_1 = resistances[chi_minus_1_index]
    uncertainty_resistance_minus = np.abs(resistance_minus_1 - resistances[min_chi2_index])

    print(
        f"The optimal resistance is {resistances[min_chi2_index]:.2f} + {uncertainty_resistance:.2f} - "
        f"{uncertainty_resistance_minus:2f} Ohm.")

def chi2_2a(R, measurements):
    voltage = measurements[:, 0]
    current = measurements[:, 1]
    uncertainties_current = measurements[:, 2]

    predictions = current_ohmslaw(voltage,R)
    chi2_varying_uncertainties = chi2(current,predictions, uncertainties_current)
    return chi2_varying_uncertainties



def ex_2b(measurements):
    resistances = np.linspace(0.1, 5, num = 100)
    chi2_varying_uncertainties = []
    for R in resistances:
        chi2_varying_uncertainties.append(chi2_2a(R, measurements))

    plt.figure()
    plt.plot(resistances, chi2_varying_uncertainties)
    plt.xlabel("Resistance [Ohm]")
    plt.ylabel("chi^2")
    plt.xlim(0, 5)
    plt.ylim(0, 1000)
    plt.savefig("ex2b.png")
    plt.show()
    plt.close()

    minimal_chi2 = np.min(chi2_varying_uncertainties)
    minimal_chi2_index = np.argmin(chi2_varying_uncertainties)
    optimal_resistance = resistances[minimal_chi2_index]

    print(f"The optimal resistance is {optimal_resistance:.2f} Ohm.")
    print(f"The minimal chi^2 value is {minimal_chi2:.2f}.")

    return optimal_resistance

def ex_2c(measurements):
    optimal_resistance = ex_2b(measurements)
    plt.figure()
    plt.plot(measurements[:, 0], measurements[:, 1], "o", label="Data")
    plt.plot(measurements[:, 0], current_ohmslaw(measurements[:, 0], optimal_resistance), label="Fit")
    plt.xlabel("Voltage [V]")
    plt.ylabel("Current [A]")
    plt.xlim(0, 15)
    plt.ylim(0, 10)
    plt.legend()
    plt.savefig("ex2c.png")
    #plt.show()
    plt.close()
    print("The plot that is saved in ex2c.png shows a pretty good fit to the data, if one considers that it has to go through the 0,0 point, because of the model is constrained."
          "Otherwise it woul be possible to get a better fit by adding a bias to the model.")

def ex_2d(measurements):
    #find the errors on the resistance
    pass



# def ex_2g():
#     """Run exercise 2g."""
#     # Here we need to use scipy.optimize.curve_fit to fit the data.
#     # make sure to first read the documentation for curve_fit
#     # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
#
#     # NOTE: curve_fit already calculates the chi2 value (including error) for us!
#     # hint: maybe look for simple examples around or play around if it is not clear on how to use curve_fit.
#
#     popt, pcov = opt.curve_fit(...)
#     print("ex2g executed.")


if __name__ == '__main__':
    # You can uncomment the exercises that you don't want to run. Here we have just one,
    # but in general you can have more.
    measurements = load_data("data/current_measurements.txt")
    measurements_uncertainties = np.array([0.2] * len(measurements))
    print("Exercise 1:")
    print("-----------------------")
    print("a)")
    print("see figure ex1a.png")
    ex_1a(measurements)
    print("b)")
    print("see in the code no output needed")
    print("c)")
    ex_1c(measurements, measurements_uncertainties)
    print("d)")
    ex_1d(measurements, measurements_uncertainties)
    print("e)")
    ex_1e(measurements, measurements_uncertainties)

    print("Exercise 2:")
    print("-----------------------")
    measurements = load_data("data/current_measurements_uncertainties.txt")
    print("a) See the code function chi2_2a")
    print("b)")
    ex_2b(measurements)
    print("c)")
    ex_2c(measurements)
    print("d)")
    ex_2d(measurements)


    #ex_2g()

