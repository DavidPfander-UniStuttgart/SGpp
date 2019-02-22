#!/usr/bin/python3

import pandas as pd
import numpy as np
import re
import sys

# flops_kernel_index=['generate_b_flops', 'density_flops', 'create_graph_flops', 'prune_graph_flops']
# kernel_names={'generate_b_dur': 'dens. right-hand side', 'density_dur': 'sum density mult.', 'density_single_it_dur': 'dens. matrix-vector mult. (1 it.)', 'create_graph_dur': 'create graph', 'prune_graph_dur': 'prune graph'}
kernel_index=['dataset_name', 'precision', 'device', 'level', 'lambda_value', 'threshold', 'k', 'generate_b_dur', 'density_dur', 'density_single_it_dur', 'create_graph_dur', 'prune_graph_dur', 'total_duration', 'ARI']

num_re = r'(\d+(?:\.\d*)?)'
sci_num_re = r'(\d+(?:\.\d*)?[eE][-+]?\d+)'
# any_num_re = r'(?:' + num_re + r'|' + sci_num_re + r')'

def filter_log_files(log_files, with_ARI = False):

    df = pd.DataFrame(np.zeros(shape=(len(kernel_index), 0)), index=kernel_index, dtype=object)

    num_column = 0;
    for file_name in log_files:
        f = open(file_name, 'r')
        content = f.read()
        # print(content)

        # device
        precision_device_pattern = 'OpenCL configuration file: OCL_configs/config_ocl_(.*?)_(.*?).cfg'
        m = re.search(precision_device_pattern, content)
        precision = m.group(1)
        device = m.group(2)

        # dataset
        binary_dataset_pattern = r'binary_header_filename: (.*?)$'
        m = re.search(binary_dataset_pattern, content, re.MULTILINE)
        dataset_name = None
        if m == None:
            dataset_pattern = r'datasetFileName: (.*?)$'
            m = re.search(binary_dataset_pattern, content, re.MULTILINE)
            dataset_name = m.group(1)
        else:
            dataset_name = m.group(1)
        print(dataset_name)

        # experiments setup

        level_pattern = 'level: ' + num_re + '\n'
        lambda_pattern = 'lambda: ' + sci_num_re  + '\n'
        threshold_pattern = 'threshold: ' + num_re + '\n'
        k_pattern = 'k: ' + num_re + '\n'

        level = re.search(level_pattern, content).group(1)
        lambda_value = re.search(lambda_pattern, content).group(1)
        threshold = re.search(threshold_pattern, content).group(1)
        k = re.search(k_pattern, content).group(1)

        # durations
        generate_b_pattern = 'last_duration_generate_b: ' + num_re + 's'
        density_pattern = 'acc_duration_density: ' + num_re + 's'
        it_pattern = 'act_it: ' + num_re + '\n'
        create_graph_pattern = 'last_duration_create_graph: ' + num_re + 's'
        prune_graph_pattern = 'last_duration_prune_graph: ' + num_re + 's'
        total_duration_pattern = 'total_duration: ' + num_re + '\n'

        generate_b_dur = float(re.search(generate_b_pattern, content).group(1))
        density_dur = float(re.search(density_pattern, content).group(1))
        its = re.search(it_pattern, content).group(1)
        density_single_it_dur = density_dur / float(its)
        create_graph_dur = float(re.search(create_graph_pattern, content).group(1))
        prune_graph_dur = float(re.search(prune_graph_pattern, content).group(1))
        total_dur = float(re.search(total_duration_pattern, content).group(1))

        ARI_value = None
        if with_ARI:
            ARI_pattern = r'ARI: ' + num_re + r'\n'
            ARI_value = float(re.search(ARI_pattern, content).group(1))

        temp_df = pd.Series([dataset_name, precision, device, level, lambda_value, threshold, k, generate_b_dur, density_dur, density_single_it_dur, create_graph_dur, prune_graph_dur, total_dur, ARI_value], index=kernel_index, dtype=object)
        # df[device_names[0]] = temp_df
        df[num_column] = temp_df.values
        num_column += 1
    print(df)
    return df

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filter_log_files(sys.argv[1:])
    else:
        print("no log files specified")
