#!/usr/bin/python

from filter_clustering_cmd_output import *

results_folder = 'results_diss/'
log_files = ['gaussian_c100_size10000000_dim10_noise_optimal.log', 'gaussian_c100_size1000000_dim10_noise_optimal.log', 'gaussian_c10_size10000000_dim10_noise_optimal.log', 'gaussian_c10_size1000000_dim10_noise_optimal.log']
log_files = [results_folder + i for i in log_files]

df = filter_log_files(log_files, with_ARI = True)
print(df)

filter_cols=['level', 'lambda_value', 'threshold', 'k', 'generate_b_dur', 'density_dur', 'density_single_it_dur', 'create_graph_dur', 'prune_graph_dur', 'total_kernels', 'total_duration', 'ARI']

df_filtered = df.loc[filter_cols,:]
print(df_filtered.index.values)
# print(df_filtered.loc['dataset_name', :]) #
# df_filtered.set_index(df_filtered.loc['dataset_name', :])
# df_filtered = df_filtered.set_index([2])
# df_filtered = df_filtered.set_index(["hallo", "fdjsio"])
df_filtered = df_filtered.rename(index=str, columns=df.loc['dataset_name', :])
print(df_filtered)
df_filtered.to_csv('results_diss/clustering_optimal_table.csv')
