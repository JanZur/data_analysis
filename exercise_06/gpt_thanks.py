import numpy as np
import matplotlib.pyplot as plt

# Load the data
measurements = np.loadtxt('data/current_measurements.txt')
voltage = measurements[:, 0]
current = measurements[:, 1]
uncertainties = np.loadtxt('data/current_measurements_uncertainties.txt')

# (a) Plot the current measurements as a function of the voltage
plt.figure()
plt.errorbar(voltage, current, yerr=uncertainties, fmt='o', label='Data')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.title('Current vs Voltage')
plt.legend()
plt.savefig('current_vs_voltage.png')
plt.show()


# (b) Functions to calculate chi2 and represent Ohm's law
def ohms_law(voltage, R):
    return voltage / R


def chi2(observed, predicted, uncertainties):
    return np.sum(((observed - predicted) / uncertainties) ** 2)


# Function to evaluate chi2 as a function of resistance
def evaluate_chi2(voltage, current, uncertainties, resistances):
    chi2_values = []
    for R in resistances:
        predicted_current = ohms_law(voltage, R)
        chi2_value = chi2(current, predicted_current, uncertainties)
        chi2_values.append(chi2_value)
    return np.array(chi2_values)


# (c) Plot chi2 as a function of resistance
resistances = np.linspace(1, 10, 1000)
chi2_values = evaluate_chi2(voltage, current, uncertainties, resistances)

plt.figure()
plt.plot(resistances, chi2_values, label='chi2')
plt.xlabel('Resistance (Ohm)')
plt.ylabel('chi2')
plt.title('chi2 vs Resistance')
plt.legend()
plt.savefig('chi2_vs_resistance.png')
plt.show()

# Find the best fit value of R
best_fit_index = np.argmin(chi2_values)
best_fit_R = resistances[best_fit_index]
print(f"The best fit value of R is {best_fit_R:.2f} Ohm")

# (d) Analytical solution for linear regression
R_analytical = np.dot(voltage, current) / np.dot(current, current)
print(f"The analytical solution for R is {R_analytical:.2f} Ohm")

# (e) Determine the uncertainties on R using the Delta(chi2) = 1 rule
chi2_min = chi2_values[best_fit_index]
chi2_plus_1 = chi2_min + 1
uncertainty_indices = np.where(np.abs(chi2_values - chi2_plus_1) < 0.01)[0]
R_uncertainty = np.abs(resistances[uncertainty_indices] - best_fit_R).max()
print(f"The uncertainty on R is ±{R_uncertainty:.2f} Ohm")
