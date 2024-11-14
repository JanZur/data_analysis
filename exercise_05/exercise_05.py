import scipy.stats as stats
import numpy as np
from numpy.core.defchararray import upper

print('Exercise 1')
print('----------')
print('a)')
# the p-value indicates what is the probability to observe a given value
# assuming it comes from the background data
p_val = stats.norm.cdf(0.83, 1, 0.06)
p_val = 2 * p_val
print(f'It is a two-tailed problem and the p-value is: {p_val:.4f}')

print('b)')
# this reduces to the problem if the difference is 0 and how far off the std is
diff = 9.9 - 9.7
uncertainty = np.sqrt(0.1 ** 2 + 0.09 ** 2)
# how likely is it to observe a difference of 0.2 given the uncertainty
p_val = 1 - stats.norm.cdf(diff, 0, uncertainty)
p_val = 2 * p_val
print(f'It is a two-tailed problem and the p-value is: {p_val:.4f}')

print('c)')
# the difference in signals is
diff = 6 - 1.5
p_val = 1 - stats.poisson.cdf(6, 1.5)
# does not make sencse to multiply by 2 since the signal is positive
print(f'It is a one tailed problem because negative signals do not make sense. The p-value is: {p_val:.4f}')

print('d)')
incidents_2019 = 50
incidents_2020 = 60

# Calculate the p-value using binomial test
p_value = stats.poisson.cdf(60, 50)
p_val = 1 - p_value
print(f'It is a one tailed problem, the p-value is: {p_val:.4f}')

print('e)')
infection_rate_without_vaccine = 3000 / 1_000_000
trial_group_size = 8924
infected_in_trial = 3
expected_infections = infection_rate_without_vaccine * trial_group_size
p_value = stats.poisson.cdf(infected_in_trial, expected_infections)
print(
    f'It is a one tailored problem because we are only interested if the vaccine decreases the infection rate the p-value is: {p_value:.4f}')

print('f)')
volleyball_heights = np.array([187, 185, 183, 176, 190])
football_heights = np.array([170, 174, 186, 178, 185, 176, 182, 184, 179, 189, 177])

mean_volleyball = np.mean(volleyball_heights)
mean_football = np.mean(football_heights)
std_volleyball = np.std(volleyball_heights, ddof=1)
std_football = np.std(football_heights, ddof=1)

t_score, p_value_t_test = stats.ttest_ind(volleyball_heights, football_heights, equal_var=False)
print(f'It is a two tailed problem and if we perform the t-test given the std are unknown the p-value is:'
      f' {p_value_t_test:.4f}')

known_std = 5
t_score_known_std = (mean_volleyball - mean_football) / np.sqrt(
    (known_std ** 2 / len(volleyball_heights)) + (known_std ** 2 / len(football_heights)))
p_value_known_std = 2 * stats.norm.cdf(-abs(t_score_known_std))
print(f'If we assume a std of 5 cm the p-value is: {p_value_known_std:.4f}')

print('Exercise 2')
print('----------')
print('a)')
counts = 240
interval = 5
sievert_per_hour = 0.1

# determine intervall of 68% oof the 240 counts
std = np.sqrt(counts)
upper_bound = counts + std
lower_bound = counts - std
# determine the interval in sievert
upper_sievert = upper_bound / counts * sievert_per_hour
lower_sievert = lower_bound / counts * sievert_per_hour
# test if it is correct
test = 1 - 2 * stats.norm.cdf(lower_bound, counts, std)
# print(f'tested {test}')
print(f'The interval in micro Sievert is: {lower_sievert:.4f} to {upper_sievert:.4f}')

print('b)')

# we search for an upper limit such that all 90 % of te distribution is below the given value
upper_limit = stats.norm.ppf(0.9, counts, std)
upper_limit_sievert = upper_limit / counts * sievert_per_hour
test_cdf = stats.norm.cdf(upper_limit, counts, std)
# print(f'The resulting percentage is: test {test_cdf}')
print(f'The resulting upper limit in micro Sievert per hour is: {upper_limit_sievert:.4f}')

print('c)')
# calculate the yearly radiation of the upper limit
yearly_radiation = upper_limit_sievert * 24 * 365
print(f'The yearly maximal radiation within the 90% is: {yearly_radiation:4f} µSv therefore it is below the '
      f'threshhold of 1000  µSv per year.')

print('Exercise 3')
print('----------')

m_measured = 90e24
d_measured = 5.2e7
sigma_m = 5e24
sigma_d = 0.2e7
rho = -0.6

uranus_mass = 86.8e24
uranus_diameter = 51.1e6
neptune_mass = 102.0e24
neptune_diameter = 49.5e6

cov_matrix = np.array([[sigma_m ** 2, rho * sigma_m * sigma_d],
                       [rho * sigma_m * sigma_d, sigma_d ** 2]])

inv_cov_matrix = np.linalg.inv(cov_matrix)


# making use of the mahalanobis distance see: https://de.wikipedia.org/wiki/Mahalanobis-Abstand
def distance(measured, planet, inv_cov_matrix):
    diff = measured - planet
    return np.sqrt(diff.T @ inv_cov_matrix @ diff)


measured_values = np.array([m_measured, d_measured])
uranus_values = np.array([uranus_mass, uranus_diameter])
neptune_values = np.array([neptune_mass, neptune_diameter])
distance_uranus = distance(measured_values, uranus_values, inv_cov_matrix)
distance_neptune = distance(measured_values, neptune_values, inv_cov_matrix)
difference_dist = np.abs(distance_uranus - distance_neptune)

print(f'the distance to neptune is: {distance_neptune:.4f} times the standard deviation.')
print(f'the distance to uranus is: {distance_uranus:.4f} times the standard deviation.')

print(f'Therefore uranus is the better choice by {difference_dist:.4f} standard deviations')
