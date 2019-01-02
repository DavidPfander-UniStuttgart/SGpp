#!/usr/bin/python3

import re
import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
import common

print("plotting clustering quality graphs")

# change these when backporting to DissertationText
results_folder = "results_WPDM18/"
img_folder= "results_WPDM18/"

file_pattern = "tune_clustering_results_{}s_noise.csv"
dataset_sizes = ["1000000", "10000000"]
dataset_sizes_map = {"1000000": "1M", "10000000": "10M"}

for dataset_size in dataset_sizes:
    file_name = file_pattern.format(dataset_size)
    print(file_name)
    df = pd.read_csv(os.path.join(results_folder, file_name))
    # print(df)
    print(df.columns.values)
    fig, ax = plt.subplots()
    splot = ax.scatter(df["lambda_value"], df[" threshold"], c=df[" percent_correct"], vmin=0.0, vmax=1.0, linewidths=0.5)
    ax.set_xticks([1E-7, 1E-6, 1E-5])
    ax.set_xticklabels([1E-7, 1E-6, 1E-5])
    ax.set_xlim((1E-7/2.0, 1E-5*2.0))
    ax.set_xscale('log', basex=10)
    fig.colorbar(splot)
    ax.set_title("correctly assigned data points, " + dataset_sizes_map[dataset_size] + ", 10d, 100 clusters")
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel("threshold")
    plt.savefig(os.path.join(img_folder, "tune_quality_scatter_" + str(dataset_size) + common.img_suffix))
    plt.clf()
    # plt.show()
