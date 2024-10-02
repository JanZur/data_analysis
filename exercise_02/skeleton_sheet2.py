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
    mean = np.sum(x, axis=0)/len(x)
    numpy_mean = np.mean(x)
    return mean


def std(x):
    """Calculate the standard deviation for an array-like object x."""
    std = np.sqrt(variance(x))
    numpy_std = np.std(x)
    return std

def variance(x):
    """Calculate the variance for an array-like object x."""
    var = np.sum(np.power(x - mean(x),2)) / (len(x)-1)
    numpy_var = np.var(x)
    return var

def mean_uncertainty(x):
    """Calculate the uncertainty in the mean for an array-like object x."""
    uncertainty_mean = std(x)/np.sqrt(len(x))
    return uncertainty_mean

def bin_values(data, n_bins):
    bin_edges = np.linspace(data.min(), data.max(), n_bins)
    bin_number = np.digitize(data, bin_edges ) -1
    bin_count = np.bincount(bin_number)

    #calculate the error bars of each bin and store it in a array
    data_bin = np.column_stack((data, bin_number))
    error = np.zeros(bin_count.size)
    bin_mean = np.zeros(bin_count.size)
    for i in range(n_bins):
        filtered_array = data_bin[data_bin[:,1] == i]
        error[i] = np.sqrt(len(filtered_array))

    bin_edges_inclusive= np.linspace(data.min(), data.max(), n_bins+1)
    bin_mean = [(bin_edges_inclusive[i] + bin_edges_inclusive[i+1]) / 2 for i in range(len(bin_edges_inclusive)-1)]
    return bin_edges, bin_count, error,bin_mean

def custom_covariance(x, y):
    covariance_custom = 1/len(x) * np.dot(x.T- mean(x), y- mean(y))
    return covariance_custom

def correlation(x, y):
    cov = custom_covariance(x, y)
    cor = cov / (std(x) * std(y))
    return cor

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
    print(f"The mean age of the participants is {mean_age:.1f}")
    print(f"The uncertainty of the mean age is {mean_age_uncertainty:.1f} years.")
    print(f"The variance of the  age of the participants is {var_age:.1f}")
    print(f"the standard deviation {std_age:.1f} years.")

    #total time
    total_time = data[:,2]
    mean_time = mean(total_time)
    mean_time_uncertainty = mean_uncertainty(total_time)
    var_time = variance(total_time)
    std_time = std(total_time)
    # .2f means that the number is printed with two decimals. Check if that makes sense
    print(f"The mean time of the participants is {mean_time:.0f}")
    print(f"The uncertainty of the mean time is  {mean_time_uncertainty:.0f} minutes.")
    print(f"The variance of the  total_time of the participants is {var_time:.0f}")
    print(f"The standard deviation {std_time:.0f} minutes.")


    #b)
    #filter the data with age older under 35
    age_time = np.column_stack((age, total_time))
    age_time_under_35 = age_time[age_time[:,0] < 35]
    mean_time_under_35 = mean(age_time_under_35[:,1])
    mean_time_uncertainty_under_35 = mean_uncertainty(age_time_under_35[:,1])
    print(f"The mean time of the participants under 35 is {mean_time_under_35:.0f} with an uncertainty to the mean of {mean_time_uncertainty_under_35:.0f} minutes.")

    age_time = np.column_stack((age, total_time))
    age_time_over_35 = age_time[age_time[:, 0] > 34]
    mean_time_over_35 = mean(age_time_over_35[:, 1])
    mean_time_uncertainty_over_35 = mean_uncertainty(age_time_over_35[:, 1])
    print(
        f"The mean time of the participants over 35 is {mean_time_over_35:.0f} with an uncertainty to the mean of {mean_time_uncertainty_over_35:.0f} minutes.")

    #c)
    #age distribution
    num_bins_age = 10
    width_age = (max(age)-min(age))/num_bins_age
    age_bin_edge, age_number_of_people, age_error, age_bin_mean = bin_values(age, num_bins_age)
    plt.bar(age_bin_mean, height = age_number_of_people, align='center', width = width_age, yerr = age_error)
    plt.xticks(age_bin_mean)
    plt.xlabel('age')
    plt.ylabel('number of people')
    plt.savefig('age_distribution_histogram.png')
    plt.close()

    # time distribution
    num_bins_time = 10
    width_time = (max(total_time)-min(total_time))/num_bins_time
    time_bin_edges, time_number_of_people, time_error, time_bin_mean = bin_values(total_time, num_bins_time)
    plt.bar(time_bin_mean, height=time_number_of_people, align='center', width=width_time, yerr=time_error)
    plt.xticks(time_bin_mean)
    plt.xlabel('total time')
    plt.ylabel('number of people')
    plt.savefig('total_time_distribution_histogram.png')
    plt.close()

    #d
    mean_age_from_bins = np.dot(age_bin_mean, age_number_of_people) / len(age)
    uncertainty_age_from_bins = uncertainty_weighted_average(age_error)
    print(f'The mean age calculated from the bins is  {mean_age_from_bins:.0f} with an uncertainty to the mean of {uncertainty_age_from_bins:.0f}  years')
    # variance and standard deviation
    # calculated by assuming the mean value of each bin as a datapoint
    # create array with the values from the num_per_bin and the center values
    age_bin = np.repeat(age_bin_mean, age_number_of_people)
    variance_age_bin = variance(age_bin)
    std_age_bin = std(age_bin)
    print(f'The variance of age calculated from the bins is  {variance_age_bin:.0f} with an std of {std_age_bin:.0f}  years')


    mean_time_from_bins = np.dot(time_bin_mean, time_number_of_people) / len(age)
    uncertainty_time_from_bins = uncertainty_weighted_average(time_error)
    print(f'The mean age calculated from the bins is  {mean_time_from_bins:.0f} with an uncertainty to the mean of  {uncertainty_time_from_bins:.0f} years')
    time_bin = np.repeat(time_bin_mean, time_number_of_people)
    variance_time_from_bins = variance(time_bin)
    std_time_bin = std(time_bin)
    print(f'The variance of total time calculated from the bins is  {variance_time_from_bins:.0f} with an std of {std_time_bin:.0f}  minutes')


    #e
    #Calculate covariance and correlation coefficents
    #total rank and total time
    cov_rank_total_time = custom_covariance(data[:, 0],data[:, 2])
    cor_rank_total_time = correlation(data[:, 0],data[:, 2])
    print(
        f"The covariance of rank and total time is  {cov_rank_total_time:.2f}")
    print(
        f"The correlation of rank and total time is  {cor_rank_total_time:.2f}")

    #age  versus total time
    cov_age_total_time = custom_covariance(age, data[:, 2])
    cor_age_total_time = correlation(age, data[:, 2])
    print(
        f"The covariance of age and total time is  {cov_age_total_time:.2f}"
    )
    print(
        f"The correlation of age and total time is  {cor_age_total_time:.2f}")

    #age versus total time in seconds
    cov_age_total_time_seconds = custom_covariance(age, data[:, 2]*60)
    cor_age_total_time_seconds = correlation(age, data[:, 2]*60)
    print(
        f"The covariance of age and total time in seconds is  {cov_age_total_time_seconds:.2f}"
    )
    print(
        f"The correlation of age and total time in seconds is  {cor_age_total_time_seconds:.2f}")

    #

def weighted_average(x, uncertainty):
    weights = 1 / np.power(uncertainty,2)
    weighted_mean = np.average(x, weights=weights)
    return weighted_mean

def uncertainty_weighted_average(uncertainty):
    sum_inverse_squared = 0
    for i in range(len(uncertainty)):
      sum_inverse_squared += 1/(uncertainty[i]**2)

    uncertainty = 1/np.sqrt(sum_inverse_squared)
    return uncertainty

def ex2():
    print('Exercise 2')
    radiation = np.loadtxt("radiation.txt")
    weighted_mean = weighted_average(radiation[:,0], radiation[:,1])
    weighted_average_uncertainty = uncertainty_weighted_average(radiation[:,1])
    radiation_per_year = weighted_mean * 24 *365
    radiation_per_year_uncertainty = weighted_average_uncertainty * 24 *365
    print(f"The weighted mean radiation per year is {radiation_per_year:.2f} msV with an uncertainty of {radiation_per_year_uncertainty:.2f} mSv")




if __name__ == '__main__':
    ex1()
    ex2()  # uncomment to run ex2
