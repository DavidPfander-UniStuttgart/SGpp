#!/usr/bin/python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

results_folder = 'results_WPDM18/'
devices = ['QuadroGP100', 'w8100', 'XeonE5-2680v3']
device_names = ['Quadro GP100', 'FirePro W8100', '2xXeon E5-2680v3']
kernel_index=['generate_b_dur', 'density_dur', 'create_graph_dur', 'prune_graph_dur']
kernel_names={'generate_b_dur': 'density right-hand side', 'density_dur': 'dens. matrix-vector mult.', 'create_graph_dur': 'create graph', 'prune_graph_dur': 'prune graph'}
runs = ["0", "1", "2"]

num_re = r'(\d+(?:\.\d*)?)'

df = pd.DataFrame(np.zeros(shape=(4, len(device_names))), columns=device_names, index=kernel_index, dtype=float)
# print(df)
# df = pd.DataFrame()

for i in range(len(devices)):
    # col_df = pd.DataFrame.from_dict([generate_b_dur, density_dur, create_graph_dur, prune_graph_dur], columns=[device_names[i]])
    col_df = df[device_names[i]]
    # print(col_df)
    for j in range(len(runs)):
        file_name = results_folder + devices[i] + "_perf_1m_run" + runs[j] + ".log"
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
        density_dur /= float(its)
        create_graph_dur = re.search(create_graph_pattern, content).group(1)
        prune_graph_dur = re.search(prune_graph_pattern, content).group(1)
        temp_df = pd.Series([generate_b_dur, density_dur, create_graph_dur , prune_graph_dur], name=device_names[i], index=kernel_index, dtype=float)
        # temp_df['generate_b_dur'] = generate_b_dur
        # temp_df['density_dur'] = density_dur
        # temp_df['create_graph_dur'] = create_graph_dur
        # temp_df['prune_graph_dur'] = prune_graph_dur
        # print(temp_df)
        # print(col_df)
        col_df = col_df + temp_df
        # print(col_df)
        # print("--------------------------------------")
        # print("generate_b_dur:", generate_b_dur)
        # print("density_dur:", density_dur)
        # print("its:", its)
        # print("create_graph_dur:", create_graph_dur)
        # print("prune_graph_dur:", prune_graph_dur)

        # col_df = pd.DataFrame.from_dict([generate_b_dur, density_dur, create_graph_dur, prune_graph_dur], columns=[device_names[i]])
    df[device_names[i]] = col_df

    # df = df.append(col_df, columns=[device_names[i]])

df = df / 3.0
df = df.T

ind = np.arange(len(device_names))



print(df)
    # print(content)
cur_bottom = pd.Series([0.0 for i in range(len(device_names))], index=device_names)
for col_name in df.columns:
    print(cur_bottom)
    print(df[col_name])
    plt.bar(ind, df[col_name], label=kernel_names[col_name], bottom=cur_bottom)
    cur_bottom += df[col_name]


plt.legend()
plt.show()
# for dataset_size in dataset_sizes:
#     file_name = file_pattern.format(dataset_size)
#     print(file_name)
#     df = pd.read_csv(os.path.join(results_folder, file_name))
