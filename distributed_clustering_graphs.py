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
LEVEL_INDEX = 10
NETWORK_SETUP_INDEX = 11
DATA_LOAD_INDEX = 12
GRID_CREATE_INDEX = 13
RHS_OPS_CREATE_INDEX = 14
DEN_MULT_OPS_CREATE_INDEX = 15
KNN_OPS_CREATE = 16

NUM_COLS = 17

results_folder = 'clustering_results'

num_re = r'\d+(?:\.\d*)?'

# grid_level_size_map = {6: 77505}

def extract_metadata(file_name):
    m = re.search(r'\_(' + num_re + ')N', file_name)
    nodes = int(m.group(1))
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
        break
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
        dataset_size = int(m.group(1))
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


        m = re.search(r'rhs creation duration: (' + num_re + ')s', read_file)
        rhs_duration = float(m.group(1))
        m = re.search(r'solver duration: (' + num_re + ')s', read_file)
        sum_solver_duration = float(m.group(1))
        m = re.search(r'create knn operation duration: (' + num_re + ')s', read_file)
        knn_create_prune_comm_duration = float(m.group(1))
        m = re.search(r'find clusters duration: (' + num_re + ')s', read_file)
        find_clusters_duration = float(m.group(1))
        m = re.search(r'elapsed time: (' + num_re + ')s', read_file)
        total_duration = float(m.group(1))

        m = re.search(r"Network setup duration: (" + num_re + ")\n", read_file)
        network_setup_duration = float(m.group(1))
        m = re.search(r"Dataset load \(on master\) duration: (" + num_re + ")\n", read_file)
        print("test:", m.group(1))
        data_load_duration = float(m.group(1))
        print("data_load_duration:", data_load_duration)
        m = re.search(r"Grid creation \(on master\) duration: (" + num_re + ")\n", read_file)
        grid_create_duration = float(m.group(1))
        m = re.search(r"RHS operation creation \(includes grid and dataset transfers\) duration: (" + num_re + ")\n", read_file)
        rhs_ops_create_duration = float(m.group(1))
        m = re.search(r"Density mult operation creation \(includes grid transfer\) duration: (" + num_re + ")\n", read_file)
        density_mult_ops_create_duration = float(m.group(1))
        m = re.search(r"KNN \(create and prune\) operation creation \(includes cached dataset transfer\) duration: (" + num_re + ")\n", read_file)
        knn_prune_ops_create_duration = float(m.group(1))

        # print("rhs_duration:", rhs_duration)
        # print("sum_solver_duration:", sum_solver_duration)
        # print("knn_create_prune_comm_duration:", knn_create_prune_comm_duration)
        # print("find_clusters_duration:", find_clusters_duration)

        if not level in timing_table:
            print("creating new level")
            timing_table[level] = {}

        if not dataset_size in timing_table[level]:
            print("creating new dataset_size")
            timing_table[level][dataset_size] = [[] for i in range(NUM_COLS)]

        print("nodes:", nodes)

        timing_table[level][dataset_size][NODES_INDEX].append(nodes)
        timing_table[level][dataset_size][TOT_DUR_INDEX].append(total_duration)
        timing_table[level][dataset_size][RHS_DUR_INDEX].append(rhs_duration)
        timing_table[level][dataset_size][SUM_DEN_DUR_INDEX].append(sum_solver_duration)
        timing_table[level][dataset_size][KNN_DUR_INDEX].append(knn_create_prune_comm_duration)
        timing_table[level][dataset_size][GP_INDEX].append(grid_size)
        timing_table[level][dataset_size][SOLVER_IT_INDEX].append(solver_it)
        timing_table[level][dataset_size][DIM_INDEX].append(dim)
        timing_table[level][dataset_size][K_INDEX].append(k)
        timing_table[level][dataset_size][LEVEL_INDEX].append(level)

        timing_table[level][dataset_size][FIND_CLUSTER_INDEX].append(find_clusters_duration)
        timing_table[level][dataset_size][NETWORK_SETUP_INDEX].append(network_setup_duration)
        timing_table[level][dataset_size][DATA_LOAD_INDEX].append(data_load_duration)
        timing_table[level][dataset_size][GRID_CREATE_INDEX].append(grid_create_duration)
        timing_table[level][dataset_size][RHS_OPS_CREATE_INDEX].append(rhs_ops_create_duration)
        timing_table[level][dataset_size][DEN_MULT_OPS_CREATE_INDEX].append(density_mult_ops_create_duration)
        timing_table[level][dataset_size][KNN_OPS_CREATE].append(knn_prune_ops_create_duration)

    return timing_table

def timings_to_flops(timing_table):
    flops_table = {}
    for level in timing_table.keys():
        flops_table[level] = {}
        for dataset_size in timing_table[level].keys():
            flops_table[level][dataset_size] = [[] for i in range(NUM_COLS)]
            for i in range(len(timing_table[level][dataset_size][NODES_INDEX])):
                nodes = timing_table[level][dataset_size][NODES_INDEX][i]
                total_duration = timing_table[level][dataset_size][TOT_DUR_INDEX][i]
                rhs_duration = timing_table[level][dataset_size][RHS_DUR_INDEX][i]
                sum_solver_duration = timing_table[level][dataset_size][SUM_DEN_DUR_INDEX][i]
                knn_create_prune_comm_duration = timing_table[level][dataset_size][KNN_DUR_INDEX][i]
                grid_size = timing_table[level][dataset_size][GP_INDEX][i]
                solver_it = timing_table[level][dataset_size][SOLVER_IT_INDEX][i]
                dim = timing_table[level][dataset_size][DIM_INDEX][i]
                k = timing_table[level][dataset_size][K_INDEX][i]
                level = timing_table[level][dataset_size][LEVEL_INDEX][i]

                print("nodes:", nodes, "total_duration:", total_duration, "rhs_duration:", rhs_duration, "sum_solver_duration:", sum_solver_duration, "knn_create_prune_comm_duration:", knn_create_prune_comm_duration, "grid_size:", grid_size, "solver_it:", solver_it, "dim:", dim, "k:", k)

                ops_rhs = grid_size * dataset_size * (6 * dim + 1) * 1E-12
                ops_density = grid_size**2 * solver_it * (14 * dim + 2) * 1E-12
                ops_create_graph = dataset_size**2 * 4 * dim * 1E-12;
                ops_prune_graph = dataset_size * grid_size * (k + 1) * (6 * dim + 2) * 1E-12;
                ops_find_clusters = 0

                ops_total = ops_rhs + ops_density + ops_create_graph + ops_prune_graph + ops_find_clusters

                total_flops = ops_total / total_duration
                rhs_flops = ops_rhs / rhs_duration
                sum_solver_flops = ops_density / sum_solver_duration
                knn_create_prune_comm_flops = (ops_create_graph + ops_prune_graph) / knn_create_prune_comm_duration

                print("total_flops:", total_flops, "rhs_flops:", rhs_flops, "sum_solver_flops:", sum_solver_flops, "knn_create_prune_comm_flops:", knn_create_prune_comm_flops)

                flops_table[level][dataset_size][NODES_INDEX].append(nodes)
                flops_table[level][dataset_size][TOT_DUR_INDEX].append(total_flops)
                flops_table[level][dataset_size][RHS_DUR_INDEX].append(rhs_flops)
                flops_table[level][dataset_size][SUM_DEN_DUR_INDEX].append(sum_solver_flops)
                flops_table[level][dataset_size][KNN_DUR_INDEX].append(knn_create_prune_comm_flops)
                flops_table[level][dataset_size][GP_INDEX].append(grid_size)
                flops_table[level][dataset_size][SOLVER_IT_INDEX].append(solver_it)
                flops_table[level][dataset_size][DIM_INDEX].append(dim)
                flops_table[level][dataset_size][LEVEL_INDEX].append(level)
    return flops_table


files = find_result_files(results_folder)
print("files:", files)

timing_table = create_plottable_timings(files)
print("timing_table:")
print(timing_table)

flops_table = timings_to_flops(timing_table)
# print("flops_table:")
# print(flops_table)


def sort_by_node(nodes_list, other_list):
    first, second = zip(*sorted(zip(nodes_list, other_list)))
    return first, second

for level in timing_table.keys():
    for dataset_size in timing_table[level].keys():

        grid_size = timing_table[level][dataset_size][GP_INDEX][0]
        dim = timing_table[level][dataset_size][DIM_INDEX][0]
        k = timing_table[level][dataset_size][K_INDEX][0]
        level = timing_table[level][dataset_size][LEVEL_INDEX][0]

        formatted_dataset_size = float(dataset_size) / 1E6
        if formatted_dataset_size > 1.0:
            formatted_dataset_size = int(formatted_dataset_size)
        formatted_dataset_size = str(formatted_dataset_size)

        # timing picture
        fig = plt.figure()
        ax = fig.add_subplot(111)
        first, second = sort_by_node(timing_table[level][dataset_size][NODES_INDEX], timing_table[level][dataset_size][TOT_DUR_INDEX])
        ax.loglog(first, second, label='total')
        first, second = sort_by_node(timing_table[level][dataset_size][NODES_INDEX], timing_table[level][dataset_size][RHS_DUR_INDEX])
        ax.loglog(first, second, label='right-hand side')
        first, second = sort_by_node(timing_table[level][dataset_size][NODES_INDEX], timing_table[level][dataset_size][SUM_DEN_DUR_INDEX])
        ax.loglog(first, second, label='sum density mult.')
        first, second = sort_by_node(timing_table[level][dataset_size][NODES_INDEX], timing_table[level][dataset_size][KNN_DUR_INDEX])
        ax.loglog(first, second, label='knn and prune')
        ax.set_title("strong scaling, level: " + str(level) + ", dim: " + str(dim) + ", k: " + str(k) + ", #data: " + formatted_dataset_size + "M")
        ax.set_xlabel("nodes")
        print("level:", level, "dataset_size:", dataset_size)
        print("first:", first)
        # ax.set_xticks(first)
        # ax.set_xticks([4, 8, 16, 32, 64], minor=True)
        # ax.set_xticklabels([4, 8, 16, 32, 64])
        ax.set_xticks([], minor=True)
        ax.set_xticks(first, minor=False)
        ax.set_xticklabels(first)
        ax.set_ylabel("duration (s)")
        ax.legend()
        plt.savefig("clustering_graphs/timing_" + str(level) + "l_" + str(dataset_size) + "s.png")
        fig.clf()
        # flops picture
        fig = plt.figure()
        ax = fig.add_subplot(111)
        first, second = sort_by_node(flops_table[level][dataset_size][NODES_INDEX], flops_table[level][dataset_size][TOT_DUR_INDEX])
        ax.plot(first, second, label='total')
        first, second = sort_by_node(flops_table[level][dataset_size][NODES_INDEX], flops_table[level][dataset_size][RHS_DUR_INDEX])
        ax.plot(first, second, label='right-hand side')
        first, second = sort_by_node(flops_table[level][dataset_size][NODES_INDEX], flops_table[level][dataset_size][SUM_DEN_DUR_INDEX])
        ax.plot(first, second, label='sum density mult.')
        first, second = sort_by_node(flops_table[level][dataset_size][NODES_INDEX], flops_table[level][dataset_size][KNN_DUR_INDEX])
        ax.plot(first, second, label='knn and prune')
        ax.set_title("strong scaling, level: " + str(level) + ", dim: " + str(dim) + ", k: " + str(k) + ", #data: " + formatted_dataset_size + "M")
        ax.set_xlabel("nodes")
        ax.set_xticks([], minor=True)
        ax.set_xticks(first, minor=False)
        ax.set_xticklabels(first)
        ax.set_ylabel("TFLOPS")
        ax.legend()
        plt.savefig("clustering_graphs/flops_" + str(level) + "l_" + str(dataset_size) + "s.png")
        fig.clf()
        # overheads picture
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # first, second = sort_by_node(timing_table[level][dataset_size][NODES_INDEX], timing_table[level][dataset_size][TOT_DUR_INDEX])
        # ax.plot(first, second, label='total')
        first, second = sort_by_node(timing_table[level][dataset_size][NODES_INDEX], timing_table[level][dataset_size][FIND_CLUSTER_INDEX])
        ax.plot(first, second, label='find clusters')

        first, second = sort_by_node(timing_table[level][dataset_size][NODES_INDEX], timing_table[level][dataset_size][NETWORK_SETUP_INDEX])
        ax.plot(first, second, label='network setup')
        first, second = sort_by_node(timing_table[level][dataset_size][NODES_INDEX], timing_table[level][dataset_size][DATA_LOAD_INDEX])
        print("nodes: ", first)
        print("data_load_index:", second)
        ax.plot(first, second, label='dataset load (master)')
        first, second = sort_by_node(timing_table[level][dataset_size][NODES_INDEX], timing_table[level][dataset_size][GRID_CREATE_INDEX])
        ax.plot(first, second, label='grid creation (master)')
        first, second = sort_by_node(timing_table[level][dataset_size][NODES_INDEX], timing_table[level][dataset_size][RHS_OPS_CREATE_INDEX])
        ax.plot(first, second, label='right-hand side ops. create')
        first, second = sort_by_node(timing_table[level][dataset_size][NODES_INDEX], timing_table[level][dataset_size][DEN_MULT_OPS_CREATE_INDEX])
        ax.plot(first, second, label='density mult. ops. create')
        ax.set_title("overhead, level: " + str(level) + ", dim: " + str(dim) + ", k: " + str(k) + ", #data: " + formatted_dataset_size + "M")
        ax.set_xlabel("nodes")
        ax.set_xticks([], minor=True)
        ax.set_xticks(first, minor=False)
        ax.set_xticklabels(first)
        ax.set_ylabel("duration (s)")
        ax.legend()
        plt.savefig("clustering_graphs/overheads_duration_" + str(level) + "l_" + str(dataset_size) + "s.png")
        fig.clf()

temp = "\n".join([str(["NODES", "TOTAL_FLOPS", "RHS_FLOPS", "SUM_SOLVER_FLOPS", "KNN_CREATE_PRUNE_COMM_FLOPS"]),str([x[0] for x in timing_table[7][10000000]]), str([float(x[0]) / 4.0 if len(x) > 0 else None for x in flops_table[7][10000000]])])
print(temp)
