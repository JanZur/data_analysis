import numpy as np
import matplotlib.pyplot as plt

# imprt the sand data
sand_data = np.loadtxt('sand.txt')
linear_fit_given = sand_data[:, 0] * 16.1 - 2.61
plt.plot(sand_data[:, 0], linear_fit_given, 'b')
plt.plot(sand_data[:, 0], sand_data[:, 1], 'o')
plt.errorbar(sand_data[:, 0], sand_data[:, 1], yerr=sand_data[:, 2], fmt='o')
plt.xlabel('Granularity (mm)')
plt.ylabel('Slope (degrees)')
plt.savefig('sand_slope.png')
