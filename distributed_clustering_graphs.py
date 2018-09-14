#!/usr/bin/python3
import re
import os
import os.path

import numpy as np
import matplotlib.pyplot as plt

# table format
# data size - > nodes | total dur. | dur. rhs | sum dur. density mult. | create + prune dur. | grid points
NODES_INDEX = 0
TOT_DUR_INDEX = 1
RHS_DUR_INDEX = 2
SUM_DEN_DUR_INDEX = 3
KNN_DUR_INDEX = 4
FIND_CLUSTER_INDEX = 5
GP_INDEX = 6
SOLVER_IT_INDEX = 7
DIM_INDEX = 8
K_INDEX = 9

NUM_COLS = 10

results_folder = 'clustering_results'

num_re = r'\d+(\.\d*)?'

# grid_level_size_map = {6: 77505}

def extract_metadata(file_name):
    m = re.search(r'\_(' + num_re + ')N', file_name)
    nodes = float(m.group(1))
    # m = re.search(r'\_(' + num_re + ')M', file_name)
    # dataset_size = float(m.group(1))
    # m = re.search(r'\_(' + num_re + ')l', file_name)
    # grid_level = int(m.group(1))
    # m = re.search(r'\_(' + num_re + ')it', file_name)
    # solver_it = float(m.group(1))
    # m = re.search(r'\_(' + num_re + ')d', file_name)
    # dim = float(m.group(1))
    # m = re.search(r'\_(' + num_re + ')k', file_name)
    # k = float(m.group(1))
    # return (nodes, dataset_size, grid_level, solver_it, dim, k)
    return nodes

def find_result_files(results_folder):
    total_files = []
    for root, dirs, files in os.walk(results_folder):
        with_root = [os.path.join(root, file_name) for file_name in files if file_name.endswith(".log")]
        total_files += with_root
    return total_files

def create_plottable_timings(files):
    timing_table = {}
    for file_name in files:
        print("file_name:", file_name)
        # nodes, dataset_size, grid_level, solver_it, dim, k = extract_metadata(file_name)
        nodes = extract_metadata(file_name)
        # print('nodes:', nodes)
        # print('dataset_size (in million)', dataset_size)
        f = open(file_name, "r")
        read_file = f.read()

        m = re.search(r'dataset_size: (' + num_re + ')\n', read_file);
        dataset_size = float(m.group(1))
        m = re.search(r'level: (' + num_re + ')\n', read_file);
        level = int(m.group(1))
        m = re.search(r'Grid created! Number of grid points:     (' + num_re + ')\n', read_file);
        grid_size = int(m.group(1))
        m = re.search(r'counted_mult_calls: (' + num_re +')\n', read_file);
        solver_it = int(m.group(1))
        m = re.search(r'dim: (' + num_re +')\n', read_file);
        dim = int(m.group(1))
        m = re.search(r'k: (' + num_re +')\n', read_file);
        k = int(m.group(1))


        m = re.search(r'rhs creation duration: (' + num_re + ')s', read_file);
        rhs_duration = float(m.group(1))
        m = re.search(r'solver duration: (' + num_re + ')s', read_file);
        sum_solver_duration = float(m.group(1))
        m = re.search(r'create knn operation duration: (' + num_re + ')s', read_file);
        knn_create_prune_comm_duration = float(m.group(1))
        m = re.search(r'find clusters duration: (' + num_re + ')s', read_file);
        find_clusters_duration = float(m.group(1))
        m = re.search(r'elapsed time: (' + num_re + ')s', read_file);
        total_duration = float(m.group(1))


        print("rhs_duration:", rhs_duration)
        print("sum_solver_duration:", sum_solver_duration)
        print("knn_create_prune_comm_duration:", knn_create_prune_comm_duration)
        print("find_clusters_duration:", find_clusters_duration)

        if not dataset_size in timing_table:
            timing_table[dataset_size] = [[] for i in range(NUM_COLS)]

        timing_table[dataset_size][NODES_INDEX].append(nodes)
        timing_table[dataset_size][TOT_DUR_INDEX].append(total_duration)
        timing_table[dataset_size][RHS_DUR_INDEX].append(rhs_duration)
        timing_table[dataset_size][SUM_DEN_DUR_INDEX].append(sum_solver_duration)
        timing_table[dataset_size][KNN_DUR_INDEX].append(knn_create_prune_comm_duration)
        timing_table[dataset_size][FIND_CLUSTER_INDEX].append(find_clusters_duration)
        timing_table[dataset_size][GP_INDEX].append(grid_size)
        timing_table[dataset_size][SOLVER_IT_INDEX].append(solver_it)
        timing_table[dataset_size][DIM_INDEX].append(dim)
        timing_table[dataset_size][K_INDEX].append(k)


    return timing_table

def timings_to_flops(timing_table):
    flops_table = {}
    for dataset_size in timing_table.keys():
        flops_table[dataset_size] = [[] for i in range(NUM_COLS)]
        for i in range(len(timing_table[dataset_size][NODES_INDEX])):
            nodes = timing_table[dataset_size][NODES_INDEX][i]
            total_duration = timing_table[dataset_size][TOT_DUR_INDEX][i]
            rhs_duration = timing_table[dataset_size][RHS_DUR_INDEX][i]
            sum_solver_duration = timing_table[dataset_size][SUM_DEN_DUR_INDEX][i]
            knn_create_prune_comm_duration = timing_table[dataset_size][KNN_DUR_INDEX][i]
            grid_size = timing_table[dataset_size][GP_INDEX][i]
            solver_it = timing_table[dataset_size][SOLVER_IT_INDEX][i]
            dim = timing_table[dataset_size][DIM_INDEX][i]
            k = timing_table[dataset_size][K_INDEX][i]

            print("nodes:", nodes, "total_duration:", total_duration, "rhs_duration:", rhs_duration, "sum_solver_duration:", sum_solver_duration, "knn_create_prune_comm_duration:", knn_create_prune_comm_duration, "grid_size:", grid_size, "solver_it:", solver_it, "dim:", dim, "k:", k)

            ops_rhs = grid_size * dataset_size * (6 * dim + 1) * 1E-9
            ops_density = grid_size**2 * solver_it * (14 * dim + 2) * 1E-9
            ops_create_graph = dataset_size**2 * 4 * dim * 1E-9;
            ops_prune_graph = dataset_size * grid_size * (k + 1) * (6 * dim + 2) * 1E-9;
            ops_find_clusters = 0

            ops_total = ops_rhs + ops_density + ops_create_graph + ops_prune_graph + ops_find_clusters

            total_flops = ops_total / total_duration
            rhs_flops = ops_rhs / rhs_duration
            sum_solver_flops = ops_density / sum_solver_duration
            knn_create_prune_comm_flops = (ops_create_graph + ops_prune_graph) / knn_create_prune_comm_duration

            print("total_flops:", total_flops, "rhs_flops:", rhs_flops, "sum_solver_flops:", sum_solver_flops, "knn_create_prune_comm_flops:", knn_create_prune_comm_flops)

            flops_table[dataset_size][NODES_INDEX].append(nodes)
            flops_table[dataset_size][TOT_DUR_INDEX].append(total_flops)
            flops_table[dataset_size][RHS_DUR_INDEX].append(rhs_flops)
            flops_table[dataset_size][SUM_DEN_DUR_INDEX].append(sum_solver_flops)
            flops_table[dataset_size][KNN_DUR_INDEX].append(knn_create_prune_comm_flops)
            flops_table[dataset_size][FIND_CLUSTER_INDEX].append(0.0)
            flops_table[dataset_size][GP_INDEX].append(grid_size)
            flops_table[dataset_size][SOLVER_IT_INDEX].append(solver_it)
            flops_table[dataset_size][DIM_INDEX].append(dim)
    return flops_table


files = find_result_files(results_folder)
print("files:", files)
timing_table = create_plottable_timings(files)
print(timing_table)

flops_table = timings_to_flops(timing_table)
print(flops_table)


def sort_by_node(nodes_list, other_list):
    first, second = zip(*sorted(zip(nodes_list, other_list)))
    return first, second

for dataset_size in timing_table.keys():
    # timing picture
    fig = plt.figure()
    ax = fig.add_subplot(111)
    first, second = sort_by_node(timing_table[dataset_size][NODES_INDEX], timing_table[dataset_size][TOT_DUR_INDEX])
    ax.plot(first, second, label='total')
    first, second = sort_by_node(timing_table[dataset_size][NODES_INDEX], timing_table[dataset_size][RHS_DUR_INDEX])
    ax.plot(first, second, label='right-hand side')
    first, second = sort_by_node(timing_table[dataset_size][NODES_INDEX], timing_table[dataset_size][SUM_DEN_DUR_INDEX])
    ax.plot(first, second, label='sum density mult.')
    first, second = sort_by_node(timing_table[dataset_size][NODES_INDEX], timing_table[dataset_size][KNN_DUR_INDEX])
    ax.plot(first, second, label='knn and prune')
    first, second = sort_by_node(timing_table[dataset_size][NODES_INDEX], timing_table[dataset_size][FIND_CLUSTER_INDEX])
    ax.plot(first, second, label='find clusters')
    ax.set_title("scaling duration")
    ax.set_xlabel("nodes")
    ax.set_ylabel("duration (s)")
    ax.legend()
    plt.savefig("clustering_graphs/timing_" + str(dataset_size) + "s.png")
    plt.clf()
    # flops picture
    fig = plt.figure()
    ax = fig.add_subplot(111)
    first, second = sort_by_node(flops_table[dataset_size][NODES_INDEX], flops_table[dataset_size][TOT_DUR_INDEX])
    ax.plot(first, second, label='total')
    first, second = sort_by_node(flops_table[dataset_size][NODES_INDEX], flops_table[dataset_size][RHS_DUR_INDEX])
    ax.plot(first, second, label='right-hand side')
    first, second = sort_by_node(flops_table[dataset_size][NODES_INDEX], flops_table[dataset_size][SUM_DEN_DUR_INDEX])
    ax.plot(first, second, label='sum density mult.')
    first, second = sort_by_node(flops_table[dataset_size][NODES_INDEX], flops_table[dataset_size][KNN_DUR_INDEX])
    ax.plot(first, second, label='knn and prune')
    ax.set_title("scaling flops")
    ax.set_xlabel("nodes")
    ax.set_ylabel("GFLOPS")
    ax.legend()
    plt.savefig("clustering_graphs/flops_" + str(dataset_size) + "s.png")
