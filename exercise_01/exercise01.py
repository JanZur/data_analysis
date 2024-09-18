import numpy as np
import matplotlib.pyplot as plt
from dateutil.utils import today

marathon_data = np.loadtxt("ironman.txt")
num_bins_total_time = 50
num_bins_age = 8


def plot_scatter(x_axis, y_axis, x_axis_name, y_axis_name, title, file_name):
    plt.scatter(x_axis, y_axis)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.title(title)
    plt.savefig(file_name)
    plt.close()


# total rank vs time
plot_scatter(marathon_data[:, 2], marathon_data[:, 0], "total time in minutes ", "rank",
             "total rank vs time",
             "a1 total rank vs time")
# age versus time
plot_scatter(today().year - marathon_data[:, 1], marathon_data[:, 2], "age in years", "total time in minutes",
             "age versus time",
             "a2 age versus time.png")
# running time versus swimming time
plot_scatter(marathon_data[:, 7], marathon_data[:, 3], "running time in minutes",
             "swimming time in minutes",
             "running time versus swimming time",
             "a3 running time versus swimming time.png")
# swimming time versus total time
plot_scatter(marathon_data[:, 3], marathon_data[:, 2], "swimming time in minutes",
             "total time in minutes",
             "swimming time versus total time",
             "a4 swimming time versus total time.png")
# cycling time versus total time
plot_scatter(marathon_data[:, 5], marathon_data[:, 2], "cycling time in minutes",
             "total time in minutes",
             "cycling time versus total time",
             "a5 cycling time versus total time.png")
# running time versus the total time
plot_scatter(marathon_data[:, 7], marathon_data[:, 2], "cycling time in minutes",
             "total time in minutes",
             "running time versus the total time",
             "a6 running time versus the total time.png")


def plot_hist(x_axis, num_bins, x_axis_name, y_axis_name, title, file_name):
    plt.hist(x_axis, num_bins)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.title(title)
    plt.savefig(file_name)
    plt.close()


plot_hist(marathon_data[:, 2], num_bins_total_time, "total time in minutes", "number of persons",
          "distribution of the achieved total times",
          "b1 distribution of the achieved total times.png")
plot_hist(today().year - marathon_data[:, 1], num_bins_age, "rank distribution", "number of persons",
          "age distribution ",
          "b2 age distribution.png")

# min and max of the historgam data
min_total_time = marathon_data[:, 2].min()
max_total_time = marathon_data[:, 2].max()
min_age = (today().year - marathon_data[:, 1]).min()
max_age = (today().year - marathon_data[:, 1]).max()

range_age = max_age - min_age
range_total_time = max_total_time - min_total_time

print(
    "The range of ages is " + str(range_age) + " years with a minimum of " + str(
        min_age) + " and a maximum of " + str(
        max_age) + "years")
print("The range of total time is " + str(range_total_time) + " minutes with a minimum of " + str(
    min_total_time) + " and a maximum of " + str(max_total_time) + "minutes")
