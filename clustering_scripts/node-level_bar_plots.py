#!/usr/bin/python3

import common
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

results_folder = 'results_WPDM18/'
devices = ['P100', 'w8100', 'hazelhen']
# device_names = ['Tesla P100', 'FirePro W8100', '2xXeon E5-2680v3']
device_names = ['P100', 'W8100', '2xE5-2680v3']
kernel_index=['generate_b_dur', 'density_dur', 'density_single_it_dur', 'create_graph_dur', 'prune_graph_dur']
flops_kernel_index=['generate_b_flops', 'density_flops', 'create_graph_flops', 'prune_graph_flops']
kernel_names={'generate_b_dur': 'density right-hand side', 'density_dur': 'dens. matrix-vector (sum)', 'density_single_it_dur': 'dens. matrix-vector mult. (1 it.)', 'create_graph_dur': 'create graph', 'prune_graph_dur': 'prune graph'}
runs = ["0", "1", "2", "3"]

num_re = r'(\d+(?:\.\d*)?)'

def create_bar_plot(dataset_size, num_clusters, lambda_value, level):
    df = pd.DataFrame(np.zeros(shape=(len(kernel_index), len(device_names))), columns=device_names, index=kernel_index, dtype=float)

    df_flops = pd.DataFrame(np.zeros(shape=(len(flops_kernel_index), len(device_names))), columns=device_names, index=flops_kernel_index, dtype=float)
    # print(df)
    # df = pd.DataFrame()

    for i in range(len(devices)):
        # col_df = pd.DataFrame.from_dict([generate_b_dur, density_dur, create_graph_dur, prune_graph_dur], columns=[device_names[i]])
        col_df = df[device_names[i]]
        flops_col_df = df_flops[device_names[i]]
        # print(col_df)
        for j in range(len(runs)):
            file_name = results_folder + devices[i] + "_perf_quality_" + str(dataset_size) + "s_" + str(num_clusters) + "c_" + str(lambda_value) + "lam_" + str(level) + "l_run" + runs[j] + ".log"
            print(file_name)
            f = open(file_name, 'r')
            content = f.read()

            # print(len(content))

            generate_b_pattern = 'last_duration_generate_b: ' + num_re + 's'
            density_pattern = 'acc_duration_density: ' + num_re + 's'
            it_pattern = 'act_it: ' + num_re + '\n'
            create_graph_pattern = 'last_duration_create_graph: ' + num_re + 's'
            prune_graph_pattern = 'last_duration_prune_graph: ' + num_re + 's'

            generate_b_dur = re.search(generate_b_pattern, content).group(1)
            density_dur = float(re.search(density_pattern, content).group(1))
            its = re.search(it_pattern, content).group(1)
            density_single_it_dur = density_dur / float(its)
            create_graph_dur = re.search(create_graph_pattern, content).group(1)
            prune_graph_dur = re.search(prune_graph_pattern, content).group(1)
            temp_df = pd.Series([generate_b_dur, density_dur, density_single_it_dur, create_graph_dur , prune_graph_dur], name=device_names[i], index=kernel_index, dtype=float)
            col_df = col_df + temp_df

            generate_b_flops_pattern = r"flops_generate_b: " + num_re + " GFLOPS"
            density_flops_pattern = r"flops_density: " + num_re + " GFLOPS"
            create_graph_flops_pattern = r"flops_create_graph: " + num_re + " GFLOPS"
            prune_graph_flops_pattern = r"flops_prune_graph: " + num_re + " GFLOPS"

            generate_b_flops = re.search(generate_b_flops_pattern, content).group(1)
            density_flops = float(re.search(density_flops_pattern, content).group(1))
            create_graph_flops = re.search(create_graph_flops_pattern, content).group(1)
            prune_graph_flops = re.search(prune_graph_flops_pattern, content).group(1)
            flops_temp_df = pd.Series([generate_b_flops, density_flops, create_graph_flops , prune_graph_flops], name=device_names[i], index=flops_kernel_index, dtype=float)
            flops_col_df = flops_col_df + flops_temp_df

        df[device_names[i]] = col_df
        df_flops[device_names[i]] = flops_col_df

    df = df / float(len(runs))
    # print(df)
    df.to_csv(results_folder + "node-level_runtime_" + str(dataset_size) + "s_" + str(num_clusters) + "c_" + str(lambda_value) + "lam_" + str(level) + "l.csv")
    df = df.T

    # print(df)

    df_flops = df_flops / float(len(runs))
    print(df_flops)

    # matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    # matplotlib.rcParams['axes.color_cycle'] = ['r', 'k', 'c']
    fig, ax = plt.subplots(figsize=(2.7, 2.3))

    ind = np.arange(len(device_names))
    cur_bottom = pd.Series([0.0 for i in range(len(device_names))], index=device_names)
    for col_name in df.columns:
        if col_name == 'density_single_it_dur':
            continue
        # print(cur_bottom)
        # print(df[col_name])
        ax.bar(ind, df[col_name], label=kernel_names[col_name], bottom=cur_bottom)
        cur_bottom += df[col_name]

    ax.set_ylabel('Duration (s)')
    ax.set_xticks(ind) #
    ax.set_xticklabels(device_names) # , rotation=30
    ax.legend()
    ax.set_title(str(int(dataset_size/1E6)) + "M Data Points, " + str(num_clusters) + " Clusters")
    # plt.show()
    fig.savefig(results_folder + "node-level_bars_" + str(dataset_size) + "s_" + str(num_clusters) + "c_" + str(lambda_value) + "lam_" + str(level) + "l.pdf")
    plt.clf()

# dataset_size, num_clusters, lambda_value, level
experiments = [(int(1E6), 10, "1E-5", 6), (int(1E6), 100, "1E-6", 7), (int(1E6), 100, "1E-7", 7), (int(1E7), 10, "1E-5", 7), (int(1E7), 100, "1E-6", 7)]

for dataset_size, num_clusters, lambda_value, level in experiments:
    create_bar_plot(dataset_size, num_clusters, lambda_value, level)
