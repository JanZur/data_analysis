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
    uncertainty_mean = std(x)/np.sqrt(len(x))
    return uncertainty_mean

def bin_values(data, n_bins):
    bin_edges = np.linspace(data.min(), data.max(), n_bins+1)
    bin_number = np.digitize(data, bin_edges ) -1
    bin_count = np.bincount(bin_number)

    #calculate the error bars of each bin and store it in a array
    data_bin = np.column_stack((data, bin_number))
    error = np.zeros(bin_count.size)
    for i in range(n_bins):
        filtered_array = data_bin[data_bin[:,1] == i]
        error[i] = np.sqrt(len(filtered_array))


    return bin_edges, bin_count, error


def ex1():
    data = np.loadtxt("ironman.txt")
    age = 2010 - data[:, 1]

    # a)
    #age distribution
    mean_age = mean(age)
    mean_age_uncertainty = mean_uncertainty(age)
    var_age = variance(age)
    std_age = std(age)
    # .2f means that the number is printed with two decimals. Check if that makes sense
    print(f"The mean age of the participants is {mean_age:.1f} +/- {mean_age_uncertainty:.1f} years.")
    print(f"The variance of the  age of the participants is {var_age:.1f} the standard deviation {std_age:.1f} years.")

    #total time
    total_time = data[:,2]
    mean_time = mean(total_time)
    mean_time_uncertainty = mean_uncertainty(total_time)
    var_time = variance(total_time)
    std_time = std(total_time)
    # .2f means that the number is printed with two decimals. Check if that makes sense
    print(f"The mean time of the participants is {mean_time:.0f} +/- {mean_time_uncertainty:.0f} minutes.")
    print(f"The variance of the  total_time of the participants is {var_time:.0f} the standard deviation {std_time:.0f} minutes.")


    #b)
    #filter the data with age older under 35
    age_time = np.column_stack((age, total_time))
    age_time_under_35 = age_time[age_time[:,0] < 35]
    mean_time_under_35 = mean(age_time_under_35[:,1])
    mean_time_uncertainty_under_35 = mean_uncertainty(age_time_under_35[:,1])
    print(f"The mean time of the participants under 35 is {mean_time_under_35:.0f} +/- {mean_time_uncertainty_under_35:.0f} minutes.")

    age_time = np.column_stack((age, total_time))
    age_time_over_35 = age_time[age_time[:, 0] > 34]
    mean_time_over_35 = mean(age_time_over_35[:, 1])
    mean_time_uncertainty_over_35 = mean_uncertainty(age_time_over_35[:, 1])
    print(
        f"The mean time of the participants over 35 is {mean_time_over_35:.0f} +/- {mean_time_uncertainty_over_35:.0f} minutes.")

    #c)
    #age distribution
    num_bins_age = 10
    width_age = (max(age)-min(age))/num_bins_age
    bin_edges, number_of_people, error = bin_values(age, num_bins_age)
    plt.bar(bin_edges, height = number_of_people, align='edge', width = width_age, yerr = error)
    plt.xticks(bin_edges)
    plt.xlabel('age')
    plt.ylabel('number of people')
    plt.show()
    plt.close()

    # time distribution
    num_bins_time = 10
    width_time = (max(total_time)-min(total_time))/num_bins_time
    bin_edges, number_of_people, error = bin_values(total_time, num_bins_time)
    plt.bar(bin_edges, height=number_of_people, align='edge', width=width_time, yerr=error)
    plt.xticks(bin_edges)
    plt.xlabel('total time')
    plt.ylabel('number of people')
    plt.show()
    plt.close()

    #d
    #mean_age_from_bins =






def ex2():
    radiation = np.loadtxt("radiation.txt")


if __name__ == '__main__':
    ex1()
    # ex2()  # uncomment to run ex2
