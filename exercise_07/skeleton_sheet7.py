import numpy as np
import scipy
from PyQt5.QtCore import center
from matplotlib import pyplot as plt
import pandas as pd



def nll(data, prob):
    """Calculate the negative log likelihood for a dataset and its predictions.

    Args:
        data (array): The data set.
        prob (array): Predicted probabilities for the data set.

    Returns:
        float: The negative log likelihood.
    """
    nll = - np.sum(np.log(prob), axis=0)

    return nll

def two_nll(data, prob):
    """Calculate 2 times the negative log likelihood for a dataset and its probabilites.

    Args:
        data (array): The data set.
        prob (array): Predicted probabilities for the data set.

    Returns:
        float: 2 times the negative log likelihood.
    """
    return 2 * nll(data, prob)  # an easy way to re-use existing code.

def nll1a(alpha):
    """Calculate the negative log likelihood for exercise 1a.

    Args:
        alpha (float): The alpha parameter.

    Returns:
        float: The negative log likelihood.
    """
    # Here is your code for exercise 1a.
    data = pd.read_csv('MLE.txt')
    data = np.array(data)

    #data = data.to_numpy()
    prob = 0.5 * (1 + alpha * data)
    #normalize with the area under the curve between -1 and 1


    return nll(data, prob)

def two_bin_nll(number_per_bin, bin_probs):
    return -2*np.sum(number_per_bin * np.log(bin_probs)- bin_probs)

def ex_1a():
    """Run exercise 1a."""
    print("1a)")
    print("see figure 1a.png")
    nll_values = []
    alphas = np.linspace(0,1,1000)
    for alpha in alphas:
        nll_values.append(nll1a(alpha))
    plt.plot(alphas, nll_values)
    plt.xlabel('Alpha')
    plt.ylabel('Negative Log-Likelihood')
    plt.xlim([0, 1])
    plt.ylim([0, 14])
    plt.savefig("1a")
    plt.close()


def ex_1b():
    """Run exercise 1b."""
    # Here is your code for exercise 1b.
    nll_values = []
    alphas = np.linspace(0, 1, 1000)
    for alpha in alphas:
        nll_values.append(nll1a(alpha))

    optimal_alpha = alphas[np.argmin(nll_values)]
    print("1b)")
    print(f"From the plot in 1a) we can see that the minimum lies around 0.5 if we calculate it we get:"
          f" {optimal_alpha:.2f}")
def exponential_pdf(data, tau):
    return 1 / (tau * (1 - np.exp(-5 / tau))) * np.exp(-data / tau)



def ex_2a():
    """Run exercise 2a."""
    exponential_data = pd.read_csv('exponential_data.txt')

    taus = np.linspace(1.8, 2.2, 1000)
    nlls = []
    for tau in taus:
        probs = exponential_pdf(exponential_data, tau)
        nlls.append(two_nll(exponential_data, probs))

    nlls = nlls - np.min(nlls)
    plt.plot(taus, nlls)
    plt.xlabel('Tau')
    plt.ylabel('Negative Log-Likelihood')
    plt.xlim([1.8, 2.2])
    plt.savefig("2a")
    plt.close()

def bin_probs_approximation(centers, tau, bin_width):
    return exponential_pdf(centers, tau) * bin_width

def bin_probs_exact(centers, tau, bin_edges):
    bin_probs = []
    for i in range(len(centers)):
        bin_probs.append(scipy.integrate.quad(exponential_pdf, bin_edges[i], bin_edges[i+1], args=(tau,))[0])
    return bin_probs


def ex_2b():
    """Run exercise 2b."""
    exponential_data = pd.read_csv('exponential_data.txt', header=None).to_numpy().flatten()
    bins = 40
    number_per_bin, bin_edges = np.histogram(exponential_data, bins=bins)
    centers_bin = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    taus = np.linspace(1.8, 2.2, 1000)
    #nlls_exact = []
    nlls_approx = []
    nlls_unbinned = []
    for tau in taus:
        exact_bin_probs = bin_probs_exact(centers_bin, tau, bin_edges)
        #nlls_exact.append(two_bin_nll(number_per_bin, exact_bin_probs))

        approx_bin_probs = bin_probs_approximation(centers_bin, tau, 5 / bins)
        nlls_approx.append(two_bin_nll(number_per_bin, approx_bin_probs))

        probs = exponential_pdf(exponential_data, tau)
        nlls_unbinned.append(two_nll(exponential_data, probs))

    nlls_unbinned =  nlls_unbinned - np.min(nlls_unbinned)
    #nlls_exact = nlls_exact - np.min(nlls_exact)
    nlls_approx = nlls_approx - np.min(nlls_approx)

    #plt.plot(taus, nlls_exact)
    plt.plot(taus, nlls_approx)
    plt.plot(taus, nlls_unbinned)
    plt.legend(["exact integration", "approximated integration", "unbinned data"])
    plt.xlabel('Tau')
    plt.ylabel('Negative Log-Likelihood')
    plt.xlim(1.8, 2.2)
    plt.ylim(0, 5)
    plt.savefig("2b.png")

    #get the tau that is corresponding to minimum of loglikelyhood
    #exact_tau = taus[np.argmin(nlls_exact)]
    approx_tau = taus[np.argmin(nlls_approx)]
    unbinned_tau = taus[np.argmin(nlls_unbinned)]
    print("2b)")
    print("From the plot we already see that there is only a small difference between the exact and the approximated "
          "curve.")
    print(f"If we compare the optimal tau values we get for the exact integration: {approx_tau:.2f} and for the "
          f"unbinned data: {unbinned_tau:.2f}")
    print("This shows that the approximated integration is a good approximation for the exact integration")

def get_chi2(data, pred):
    pred = pred * np.sum(data)
    diff = data - pred
    mault = diff * diff
    sum = np.sum(mault)

    return sum
def ex_2c(numbin):
    #chi 2 is the squared error of the data
    exponential_data = pd.read_csv('exponential_data.txt', header=None).to_numpy().flatten()
    taus = np.linspace(1.8, 2.2, 1000)
    bins = numbin
    number_per_bin, bin_edges = np.histogram(exponential_data, bins=bins)
    centers_bin = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    chi2 = []
    nlls_approx = []
    nlls_unbinned = []
    for i, tau in enumerate(taus):
        pred_number_per_bin =  bin_probs_approximation(centers_bin, tau, 5 / bins)
        #to numpy array
        pred_number_per_bin = np.array(pred_number_per_bin)
        number_per_bin = np.array(number_per_bin)

        approx_bin_probs = bin_probs_approximation(centers_bin, tau, 5 / bins)
        nlls_approx.append(two_bin_nll(number_per_bin, approx_bin_probs))

        probs = exponential_pdf(exponential_data, tau)
        nlls_unbinned.append(two_nll(exponential_data, probs))

        chi2.append(get_chi2(number_per_bin, pred_number_per_bin))

    nlls_unbinned = nlls_unbinned - np.min(nlls_unbinned)
    nlls_approx = nlls_approx - np.min(nlls_approx)
    chi2 = chi2 - min(chi2)
    plt.plot(taus, nlls_approx)
    plt.plot(taus, chi2)
    plt.plot(taus, nlls_unbinned)
    plt.xlabel('Tau')
    plt.ylabel('Negative Log-Likelihood / Chi2')
    plt.xlim(1.8, 2.2)
    plt.ylim(0, 5)
    plt.legend(["binned data", "chi2", "unbinned data"])
    plt.close()

    print("2c)")
    print("From the plot we can see that the chi2 is not the same as the negative log likelyhood. This means"
          " that the least squares method is not optimizing for the same objective as the maximum likelyhood.")

def ex_2d(numbin):
    # chi 2 is the squared error of the data
    exponential_data = pd.read_csv('exponential_data.txt', header=None).to_numpy().flatten()
    taus = np.linspace(1.8, 3, 100)
    bins = numbin
    number_per_bin, bin_edges = np.histogram(exponential_data, bins=bins)
    centers_bin = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    chi2 = []
    nlls_approx = []
    nlls_unbinned = []
    for i, tau in enumerate(taus):
        pred_number_per_bin = bin_probs_approximation(centers_bin, tau, 5 / bins)
        # to numpy array
        pred_number_per_bin = np.array(pred_number_per_bin)
        number_per_bin = np.array(number_per_bin)

        approx_bin_probs = bin_probs_approximation(centers_bin, tau, 5 / bins)
        nlls_approx.append(two_bin_nll(number_per_bin, approx_bin_probs))

        probs = exponential_pdf(exponential_data, tau)
        nlls_unbinned.append(two_nll(exponential_data, probs))

        chi2.append(get_chi2(number_per_bin, pred_number_per_bin))

    nlls_unbinned = nlls_unbinned - np.min(nlls_unbinned)
    nlls_approx = nlls_approx - np.min(nlls_approx)
    chi2 = chi2 - min(chi2)
    plt.plot(taus, nlls_approx)
    plt.plot(taus, chi2)
    plt.plot(taus, nlls_unbinned)
    plt.xlabel('Tau')
    plt.ylabel('Negative Log-Likelihood / Chi2')
    plt.xlim(1.8, 3)
    plt.ylim(0, 5)
    plt.legend(["binned data", "chi2", "unbinned data"])
    plt.savefig("2d")
    plt.close()

    print("2d)")
    print("Because there are only 2 bins the binned approximation is pretty bad in comparison to the unbinned data. "
          "The chi2 did not change and is still the same as in 2c). But performes better than the large bins.")

def ex_3a():
    # Here is your code for exercise 3a.
    data = pd.read_csv('polynomial_data.txt').to_numpy().flatten()
    bins =  20
    number_per_bin, bin_edges = np.histogram(data, bins=bins)
    uncertainties_per_bin = np.sqrt(number_per_bin)
    centers_bin = 0.5 * (bin_edges[1:] + bin_edges[:-1])



    plt.hist(data, bins=bins)
    plt.errorbar(centers_bin, number_per_bin, yerr=uncertainties_per_bin, fmt='o')

    plt.savefig("3a.png")

def polynomial(x, *coeffs):
    return sum(c * x ** i for i, c in enumerate(coeffs))

def ex_3b():
    print("3b) see the plot 3b.png")
    print("3c)")
    data = pd.read_csv('polynomial_data.txt').to_numpy().flatten()
    bins = 20
    number_per_bin, bin_edges = np.histogram(data, bins=bins)
    uncertainties_per_bin = np.sqrt(number_per_bin)
    centers_bin = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    orders = [1, 2, 3, 4]
    chi2_ndf = []

    for order in orders:
        popt, pcov = scipy.optimize.curve_fit(lambda x, *params: polynomial(x, *params), centers_bin, number_per_bin,
                                              p0=[1] * (order + 1))
        uncertainties = np.sqrt(np.diag(pcov))
        print(f"Order {order}:")
        for i, p in enumerate(popt):
            print(f"p{i} = {p:.2f} +- {uncertainties[i]:.2f}")

        resid = number_per_bin - polynomial(centers_bin, *popt)
        chi2 = np.sum((resid / uncertainties_per_bin) ** 2)
        ndf = len(number_per_bin) - len(popt)
        chi2_ndf.append(chi2 / ndf)

        if order == 4:
            plt.plot(centers_bin, polynomial(centers_bin, *popt), label=f'{order} order polynomial', linestyle='dotted')
        else:
            plt.plot(centers_bin, polynomial(centers_bin, *popt), label=f'{order} order polynomial')

    plt.legend()
    plt.hist(data, bins=bins)
    plt.errorbar(centers_bin, number_per_bin, yerr=uncertainties_per_bin, fmt='o')
    plt.xlim([-1, 1])
    plt.ylim([0, 1750])
    plt.xlabel('x')
    plt.ylabel('number of data points')
    plt.savefig("3b.png")
    plt.close()


    plt.plot(orders, chi2_ndf)
    plt.xlabel('Degree of polynomial')
    plt.ylabel('chi2/ndf')
    plt.savefig("3d.png")

    print("3d) see figure 3d.png")
    print("3e) It was a polynomial of degree 3 becuase the chi2/ndf is the smallest for this polynomial and for "
          "degree 4 it does not change much anymore. Additionally we could already see in 3b) that the 3rd and 4th "
          "order look the same.")





if __name__ == '__main__':
    # You can uncomment the exercises that you don't want to run. Here we have just one,
    # but in general you can have more.
    ex_1a()
    ex_1b()
    ex_2a()
    ex_2b()
    ex_2c(40)
    ex_2d(2)
    ex_3a()
    ex_3b()