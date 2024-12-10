"""Skeleton for Data Analysis Fall 2024 sheet 6.

This sheet helps to structure the code for the exercises of sheet 6.
It shows idioms and best practices for writing code in Python.

You can directly use this sheet and modify it.

(It is not guaranteed to be bug free)
"""
import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import argmin
from scipy.stats import chi2



def load_data(filename):
    return np.loadtxt(filename)

def current_ohmslaw(U, R):
    return U / R

def current_ohmslaw_bias(U, R, bias=None):
    if bias is None:
        bias = 0  # with this, we can also use the function without bias.
    return (U + bias) / R

def chi2_own_implementation(x, y, err):
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

def chi2_1b(R, measurements, uncertainties, bias = None):
    data = measurements
    voltage = data[:, 0]
    current = data[:, 1]

    current_pred = current_ohmslaw_bias(voltage, R, bias)
    chi2val = chi2_own_implementation(current, current_pred, uncertainties)
    return chi2val

def get_chi2_array(measurements, resistances, uncertainties, bias=None):
    chi2val = np.zeros(len(resistances))
    for i, R in enumerate(resistances):
        chi2val[i] = chi2_1b(R, measurements, uncertainties, bias)
    return chi2val

def get_optimal_resistance(resistances, chi2_array):
    idx_min = argmin(chi2_array)
    return resistances[idx_min]

def ex_1c(measurements, uncertainties):
    resistances = np.linspace(0.2, 10, num=1000)
    chi2val = get_chi2_array(measurements, resistances, uncertainties)
    optimal_resistance = get_optimal_resistance(resistances,chi2val)

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
    print("The values are not the same as the one optained from the plotting method, this is due to the fact that the "
          "plotting assumes a linear relation without a bias and therefore is off by some small amount because the "
          "measurement has a small bias.")


def ex_1e(measurements, measurements_uncertainties):
    # calculate the uncertainty on the analytical resistance

    optimal_analytical_resistance = optimal_resistance_analytical(measurements, measurements_uncertainties)
    uncertainty_analytical_resistance = 1 / np.sqrt(np.sum(1 / measurements_uncertainties ** 2))
    print(
        f"The optimal analytical calculated resistance is {optimal_analytical_resistance:.2f} +- {uncertainty_analytical_resistance:.2f} Ohm.")

    resistances = np.linspace(1, 5, 1000)
    chi2val = get_chi2_array(measurements,resistances, measurements_uncertainties)
    min_chi2 = min(chi2val)
    optimal_resistance = get_optimal_resistance(resistances, chi2val)

    uncertainty_resistance, uncertainty_resistance_minus = get_deltachi_uncertainty(chi2val,min_chi2,resistances,optimal_resistance)

    print(
        f"The optimal resistance is {optimal_resistance:.2f} + {uncertainty_resistance:.2f} - "
        f"{uncertainty_resistance_minus:2f} Ohm.")


def get_deltachi_uncertainty(chi2_values , chi2_min : float, resistances , optimal_resistance : float):
    chi2_plus_1 = chi2_min + 1
    indx_min = np.argmin(chi2_values)
    chi2_values_below_min = chi2_values[:indx_min]
    chi2_values_above_min = chi2_values[indx_min:]
    resistance_below_min = resistances[:len(chi2_values_below_min)]
    resistance_above_min = resistances[len(chi2_values_below_min):]

    idx_below_min = argmin(np.abs(chi2_values_below_min-chi2_plus_1))
    idx_above_min = argmin(np.abs(chi2_values_above_min-chi2_plus_1))
    uncertainty_below = np.abs(optimal_resistance-resistances[idx_below_min])
    uncertainty_above = np.abs(optimal_resistance-resistances[idx_above_min])

    return uncertainty_below, uncertainty_above


# def chi2_2a(R, measurements, bias = None):
#     voltage = measurements[:, 0]
#     current = measurements[:, 1]
#     uncertainties_current = measurements[:, 2]
#
#     predictions = current_ohmslaw_bias(voltage,R, bias)
#     chi2_varying_uncertainties = chi2(current,predictions, uncertainties_current)
#     return chi2_varying_uncertainties



def ex_2b(measurements):
    resistances = np.linspace(0.1, 5, num = 100)
    uncertainties = measurements[:, 2]
    chi2_varying_uncertainties = get_chi2_array(measurements, resistances, uncertainties)
    # for R in resistances:
    #     chi2_varying_uncertainties.append(chi2_1b(R, measurements,uncertainties))

    plt.figure()
    plt.plot(resistances, chi2_varying_uncertainties)
    plt.xlabel("Resistance [Ohm]")
    plt.ylabel("chi^2")
    plt.xlim(0, 5)
    plt.ylim(0, 1000)
    plt.savefig("ex2b.png")
    #plt.show()
    plt.close()

    minimal_chi2 = np.min(chi2_varying_uncertainties)
    minimal_chi2_index = np.argmin(chi2_varying_uncertainties)
    optimal_resistance = resistances[minimal_chi2_index]

    print(f"The optimal resistance is {optimal_resistance:.2f} Ohm.")
    print(f"The minimal chi^2 value is {minimal_chi2:.2f}.")

    #return optimal_resistance, chi2_varying_uncertainties

def ex_2c(measurements):
    resistances = np.linspace(0.1, 5, num = 100)
    chi2_array = get_chi2_array(measurements,resistances, measurements[:,2])
    optimal_resistance = get_optimal_resistance(resistances, chi2_array)
    plt.figure()
    plt.plot(measurements[:, 0], measurements[:, 1], "o", label="Data")
    plt.plot(measurements[:, 0], current_ohmslaw_bias(measurements[:, 0], optimal_resistance), label="Fit")
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
    resistances = np.linspace(0.1, 5, num=100)
    chi2_array = get_chi2_array(measurements, resistances, measurements[:, 2])
    optimal_resistance = get_optimal_resistance(resistances, chi2_array)

    uncertainty_resistance, uncertainty_resistance_minus = get_deltachi_uncertainty(chi2_array, np.min(chi2_array),
                                                                                    np.linspace(
        0.1, 5, num=100), optimal_resistance)
    print(f"The uncertainties are: {uncertainty_resistance:.2f} and {uncertainty_resistance_minus:.2f} Ohm.")
    print("These values are smaller than the given 2 Ohm and therefore is not compatible.")

def ex_2e(measurements):
    #use the offset function for prediciton
    resistances = np.linspace(0.1, 5, num=100)
    chi2_array = get_chi2_array(measurements, resistances, measurements[:, 2],0.7)
    optimal_resistance = get_optimal_resistance(resistances, chi2_array)
    uncertainty_upper, uncertainty_lower = get_deltachi_uncertainty(chi2_array,min(chi2_array),resistances, optimal_resistance)

    print(f"When using the bias of 0.7 we get that the optimal resistance is: {optimal_resistance} + "
          f"{uncertainty_upper} - {uncertainty_lower}")
    print("This is still not compatible with the given 2 Ohm uncertainty.")

    #plot the line to reassure it makes sense
    # plt.figure()
    # plt.plot(measurements[:, 0], measurements[:, 1], "o", label="Data")
    # plt.plot(measurements[:, 0], current_ohmslaw_bias(measurements[:, 0], optimal_resistance,                                              0.7), label="Fit with bias")
    # plt.xlabel("Voltage [V]")
    # plt.ylabel("Current [A]")
    # plt.xlim(0, 15)
    # plt.ylim(0, 10)
    # plt.legend()
    # plt.show()
    # plt.close()

def chi2_ndf(chi2_values, ndf):
    return chi2_values / ndf

def ex_2f(measurements):
    resistances = np.linspace(0.1, 5, num=100)
    chi2_array_bias = get_chi2_array(measurements, resistances, measurements[:, 2],0.7)
    chi2_array_no_bias = get_chi2_array(measurements, resistances, measurements[:, 2])

    chi2_ndf_bias = chi2_ndf(min(chi2_array_bias), len(measurements)-2)
    chi2_ndf_no_bias = chi2_ndf(min(chi2_array_no_bias), len(measurements)-1)

    print("The chi2 ndf with the bias is: ", chi2_ndf_bias)
    print("The chi2 ndf without the bias is: ", chi2_ndf_no_bias)
    print("This shows that the one with the bias as an additional parameter works better.")

    goodness_with_bias = chi2.cdf(chi2_ndf_bias * len(measurements - 2), len(measurements) - 2)
    goodness_without_bias = chi2.cdf(chi2_ndf_no_bias * len(measurements - 1), len(measurements) - 1)


    print("The goodness of fit with the bias is: ", goodness_with_bias)
    print("The goodness of fit without the bias is: ", goodness_without_bias)

    



def ex_2g(measurements):
    """Run exercise 2g."""
    # Here we need to use scipy.optimize.curve_fit to fit the data.
    # make sure to first read the documentation for curve_fit
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

    # NOTE: curve_fit already calculates the chi2 value (including error) for us!
    # hint: maybe look for simple examples around or play around if it is not clear on how to use curve_fit.

    fit_params, fit_covariance_matrix = opt.curve_fit(current_ohmslaw_bias,measurements[:, 0], measurements[:, 1],
                                                      sigma=measurements[:, 2])

    print("The optimal resistance is: ", fit_params[0])
    print("The offset is : ", fit_params[1])
    print("The uncertainty on the resistance is: ", np.sqrt(fit_covariance_matrix[0,0]))
    print("h)")
    print("The uncertainty on the offset is: ", np.sqrt(fit_covariance_matrix[1,1]))
    print("Therefore we get a value that lies in between the two values we got before in exercise e.")





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
    print("e)")
    ex_2e(measurements)
    print("f)")
    ex_2f(measurements)
    print("g)")
    ex_2g(measurements)


