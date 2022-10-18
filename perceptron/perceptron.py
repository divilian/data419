
# 2d perceptron

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from genData import gen_data, label_of_pts_wrt_weights

# 12, 123, 123456
np.random.seed(1234)

N = 50 # number of training points
MAX_ITER = 1000  # (Necessary if data is not linearly separable.)


training_data = gen_data(N,2,separable=True)


def plot_frame(curr_weights, msg):
    plt.clf()
    reds = training_data[training_data.y == 1]
    greens = training_data[~training_data.index.isin(reds.index)]
    plt.scatter(reds.x1,reds.x2,color="red",marker="x")
    plt.scatter(greens.x1,greens.x2,color="green",marker="o")
    plt.axhline(y=0,color="grey",linestyle="dotted")
    plt.axvline(x=0,color="grey",linestyle="dotted")
    slope = -curr_weights[1]/curr_weights[2]
    intercept = -curr_weights[0]/curr_weights[2]
    plt.axline(xy1=(0,intercept), slope=slope, color="black",
        linestyle="dashed")
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.title(msg)
    plt.show()


def choose_misclassified_pt(curr_weights):
    mis = training_data[ 
        label_of_pts_wrt_weights(training_data, curr_weights) !=
        training_data.y ]
    if len(mis) == 0:
        return -1, len(mis)
    else:
        return np.random.choice(mis.index), len(mis)


curr_weights = np.random.uniform(-10,10,3)

mis_pt, num_mis = choose_misclassified_pt(curr_weights)
num_iter = 0
plot_frame(curr_weights, f"Misclassified {num_mis} of {N} ({num_iter} iters)")
input("Press enter to start perceptron.")
while mis_pt != -1 and num_iter < MAX_ITER:
    num_iter += 1
    # Update current weights in "the right direction"
    curr_weights += (np.concatenate([np.array([10]),
            training_data.iloc[mis_pt][["x1","x2"]]]) * 
        training_data.iloc[mis_pt].y) * num_mis / 5
    mis_pt, num_mis = choose_misclassified_pt(curr_weights)
    plot_frame(curr_weights, f"Misclassified {num_mis} of {N} ({num_iter} iters)")
    plt.pause(.05)
