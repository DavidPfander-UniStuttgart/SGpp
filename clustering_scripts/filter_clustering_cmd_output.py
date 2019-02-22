#!/usr/bin/python3

import pandas as pd
import re
import sys

def filter_log_files(log_files):

    df = pd.DataFrame(np.zeros(shape=(len(kernel_index), len(device_names))), columns=device_names, index=kernel_index, dtype=float)

    for file_name in log_files:
        f = open(file_name, 'r')
        content = f.read()

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
    return df

if __name__ == "__main__":
    filter_log_files([i for i in sys.argv])
