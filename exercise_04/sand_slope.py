import numpy as np
import matplotlib.pyplot as plt

slope = 16.1
offset = -2.61
m_unc = 1
q_unc = 0.34
# import the sand data
sand_data = np.loadtxt('sand.txt')

linear_fit_given = np.linspace(0, 3) * 16.1 - 2.61

plt.plot(np.linspace(0, 3), linear_fit_given, 'b')
plt.plot(sand_data[:, 0], sand_data[:, 1], 'o')
plt.errorbar(sand_data[:, 0], sand_data[:, 1], yerr=sand_data[:, 2], fmt='o')
plt.xlabel('Granularity (mm)')
plt.ylabel('Slope (degrees)')
plt.xlim(0, 1.7)
plt.ylim(0, 20)
plt.savefig('sand_slope.png')


def get_slope_ignore_covariance(granularity, m, q, granularity_unc, offset_unc):
    slope = m * granularity + q
    slope_unc = np.sqrt((granularity * granularity_unc) ** 2 + (offset_unc) ** 2)
    return slope, slope_unc


def get_slope_with_covariance(granularity, m, q, granularity_unc, offset_unc):
    slope = m * granularity + q
    # get uncertainty from variables m and q
    slope_unc = np.sqrt((granularity * granularity_unc) ** 2 + (offset_unc) ** 2 + 2 * granularity * offset_unc)
    return slope, slope_unc


slope_1, slope_1_unc = get_slope_ignore_covariance(1.5, slope, offset, m_unc, q_unc)
slope_2, slope_2_unc = get_slope_with_covariance(1.5, slope, offset, m_unc, q_unc)

print(f'4b) The slope without considering the covariance is: {slope_1:.1f} +/- {slope_1_unc:.1f}')
print(f'4b) The slope considering the covariance is: {slope_2:.1f} +/- {slope_2_unc:.1f}')
print('The uncertainty considering the covariance is higher because it takes the effect of two indeendent variables '
      'into account. This shows that it is important to consider the variance bbecause otherwise one would '
      'underestumate the uncertainty.')
