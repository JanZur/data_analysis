"""Skeletоn sheet 3 Datenanalyse University of Zurich"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from mkl_random.mklrand import poisson
from numpy import number


def integrate(dist, lower, upper):
    """Integrate the pdf of a distribution between lower and upper.

    Parameters
    ----------
    dist : scipy.stats.rv_continuous
        A scipy.stats distribution object.
    lower : float
        Lower limit of the integration.
    upper : float
        Upper limit of the integration.

    Returns
    -------
    integral : float
        The integral of the pdf between lower and upper.
    """
    return  # TODO: add your code here


# THIS FUNCTION IS NOT NEEDED, JUST DEMONSTRATION PURPOSE
def example_integrate_shifted_norm(x):
    # to get a "norm distribution" with mean 5 and std 3, we can use
    norm_dist_shifted = scipy.stats.norm(loc=5, scale=3)
    # we can then use different methods of the norm_dist_shifted object to calculate
    # the probability density function (pdf) and the cumulative distribution function (cdf)
    # and more.
    # using the cdf we can also calculate the integral of the pdf:
    integrate_4to7 = norm_dist_shifted.cdf(7) - norm_dist_shifted.cdf(4)  # integral form 4 to 7
    # or just write a function that does it for us
    integral_1to10 = integrate(norm_dist_shifted, 1, 10)


def ex1():
    # Write and execute a function to plot the probability distribution for the number of
    #  signals registered if the particle travels through four detectors given that the probability is 85%
    print("Exercise 1")
    print("a)")
    p = 0.85
    n = 4
    x = np.arange(0, n + 1)
    binom_dist = scipy.stats.binom(n, p)
    y = binom_dist.pmf(x)
    plt.bar(x, y)
    plt.xlabel("Number of signals")
    plt.ylabel("Probability")
    plt.title("Probability distribution for the number of detected signals")
    plt.savefig("probability_distribution_4_signals.png")
    plt.close()
    print("The resulting probabilities are " + str(y.round(3)) + "for the number of detected signals" + str(x))

    print("b)")
    minimal_number_of_detectors = 0
    print(
        "The probability of having a signal in at least three detectors is given by 1 - the probability of not beeing detected 0,1 and 2 detectors")
    number_of_detectors = np.arange(3, 10)
    number_of_detectors = number_of_detectors[::-1]
    for n in number_of_detectors:
        binom_dist_n = scipy.stats.binom(n, p)
        p_min_three = 1 - binom_dist_n.cdf(2)
        print("For " + str(n) + " detectors the probability to get detected by at least three detectors is: " + str(
            p_min_three))
        if (p_min_three >= 0.99):
            minimal_number_of_detectors = n
    print("The minimal number of detectors to reach 99% therefore is: " + str(minimal_number_of_detectors))

    print("c)")
    print(
        "so the question is about how likely it is to get k signals when the experiment is run with 4 detectors and 1000 paritcles")
    number_of_particles = 1000
    p_detected_in_3or4 = 1 - binom_dist.cdf(2)
    binom_dist_4 = scipy.stats.binom(number_of_particles, p_detected_in_3or4)
    x = np.arange(0, number_of_particles)
    y = binom_dist_4.pmf(x)

    # compare to poisson distribution
    poisson_dist = scipy.stats.poisson(number_of_particles * p_detected_in_3or4)
    y_poisson = poisson_dist.pmf(x)
    plt.bar(x, y, alpha=0.5, label="Binomial")
    plt.bar(x, y_poisson, alpha=0.5, label="Poisson")
    plt.legend()
    plt.bar(x, y)
    plt.xlim(800, number_of_particles)
    plt.xlabel("Number of signals")
    plt.ylabel("Probability")
    plt.savefig("probability_distribution_poissonvsbinomial_1000_particles.png")
    plt.close()
    print(
        "As one can see in plot probability_distribution_poissonvsbinomial_1000_particles.png the poisson distribution "
        "is not a good approximation for the binomial distribution in this case because 1000 particles are not enough to"
        " approximate to infinity what would be needed in the case of the poisson distribution.")


def ex3():
    print("Exercise 3")
    normal_dist = scipy.stats.norm(1, 0.01)
    lower_bound_1 = 0.97
    upper_bound_1 = 1.03
    probability_1 = normal_dist.cdf(upper_bound_1) - normal_dist.cdf(lower_bound_1)

    lower_bound_2 = 0.99
    upper_bound_2 = 1
    probability_2 = normal_dist.cdf(upper_bound_2) - normal_dist.cdf(lower_bound_2)

    lower_bound_3 = 0.95
    upper_bound_3 = 1.05
    probability_3 = normal_dist.cdf(upper_bound_3) - normal_dist.cdf(lower_bound_3)

    upper_bound_4 = 1.015
    probability_4 = normal_dist.cdf(upper_bound_4)

    print("Probabilities:")
    print(f"3a) {probability_1:.3f} to be within 0.97 and 1.03")
    print(f"3b) {probability_2:.2f} to be within 0.99 and 1.00")
    print(f"3c) {probability_3:.7f} to be within 0.95 and 1.05")
    print(f"3d) {probability_4:.2f} to be below 1.015")

    # 3f means that the number is printed with three decimals. Check if that makes sense!


def ex4():
    print("Exercise 4")
    print("a)")
    p_into_charged = 0.82
    n_boson = 500
    t_running = 125
    # more than 390 bosons are detected to decay into charged particles
    binomial_dist = scipy.stats.binom(n_boson, p_into_charged)
    probability_more_than_390 = 1 - binomial_dist.cdf(390)
    print(f"The probability to detect more than 390 z-bosons decaying into charged particles is: "
          f"{probability_more_than_390:.2f}")

    print("b)")
    mean_binomial = binomial_dist.mean()
    # mean_binomial = p_into_charged * n_boson
    print(f"The mean number of z-bosons decaying into charged particles is: {mean_binomial:.2f}")

    std_binomial = binomial_dist.std()
    print(f"The standard deviation of the number of z-bosons decaying into charged particles is: {std_binomial:.2f}")

    gaussian_approximation = scipy.stats.norm(mean_binomial, std_binomial)
    probability_more_than_390_gaussian = 1 - gaussian_approximation.cdf(390)
    print("The probability for more than 390 detections with the gaussian approximation is: " + str(
        probability_more_than_390_gaussian.round(2)))

    plt.bar(np.arange(0, 500), binomial_dist.pmf(np.arange(0, 500)), label="Binomial")
    plt.bar(np.arange(0, 500), gaussian_approximation.pdf(np.arange(0, 500)), label="Gaussian")
    plt.legend()
    plt.xlim(350, 450)
    plt.xlabel("Number of charged particles")
    plt.ylabel("Probability")
    plt.savefig("gaussian_approximation_binomial_compareison.png")
    plt.show()
    plt.close()

    print("The difference between the binomial distribution and the gaussian approximaiton is very small as can be "
          "seen in the plot gaussian_approximation_binomial_comparison.png")

    print("c)")
    poisson_dist = scipy.stats.poisson(mean_binomial)
    probability_more_than_390_poisson = 1 - poisson_dist.cdf(390)
    print("The probability for more than 390 detections with the poisson approximation is: " + str(
        probability_more_than_390_poisson.round(2)))
    plt.bar(np.arange(0, 500), poisson_dist.pmf(np.arange(0, 500)), label="Poisson")
    plt.bar(np.arange(0, 500), binomial_dist.pmf(np.arange(0, 500)), label="Binomial")
    plt.legend()
    plt.xlim(350, 750)
    plt.xlabel("Number of charged particles")
    plt.ylabel("Probability")
    plt.savefig("poisson_approximation_binomial_comparison.png")
    plt.close()
    print("The poisson approximation is not good as can be seen in the plot "
          "poisson_approximation_binomial_comparison.png it's width is too small and the peak is too high.")

    print("d)")
    # probability that at least one z-boson was not detected because of decay to neutrinos
    p_neutrino = 0.18
    binomial_dist_neutrino = scipy.stats.binom(n_boson, p_neutrino)
    probability_at_least_one_neutrino = 1 - binomial_dist_neutrino.pmf(0)

    print(f"The probability that at least one z-boson was not detected because of decay to neutrinos is: "
          f"{probability_at_least_one_neutrino:}")

    poisson_dist_neutrino = scipy.stats.poisson(n_boson * p_neutrino)
    probability_at_least_one_neutrino_poisson = 1 - poisson_dist_neutrino.pmf(0)
    print(f"The probability that at least one z-boson was not detected because of decay to neutrinos is: "
          f"{probability_at_least_one_neutrino_poisson}")

    plt.bar(np.arange(0, 500), binomial_dist_neutrino.pmf(np.arange(0, 500)), label="Binomial")
    plt.bar(np.arange(0, 500), poisson_dist_neutrino.pmf(np.arange(0, 500)), label="Poisson")
    plt.legend()
    plt.xlabel("Number of neutrinos")
    plt.ylabel("Probability")
    plt.savefig("poisson_approximation_binomial_comparison_neutrinos.png")

    print("The poisson approximation is better in theis case because the used probability is a lot smaller than in "
          "the previous case. Therefore the ration between the number of evaluations and the probability is mush "
          "higher, resulting in the better approximation. This can be "
          "seen in the plot poisson_approximation_binomial_comparison_neutrinos.png")


if __name__ == '__main__':
    ex1()
    ex3()  # uncomment to run ex3
    ex4()  # uncomment to run ex4
