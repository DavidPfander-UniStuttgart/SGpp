from generate_datasets import generate_dataset
from generate_datasets import write_arff_header
from generate_datasets import write_csv_without_erg
import argparse
import os
import subprocess
import atexit
import numpy as np
import json
import time
import filecmp

def cleanup():
    subprocess.run(["rm", "-rf", "dataset-tmp"])
    subprocess.run(["rm", "-rf", "ocl-results"])
    subprocess.run(["rm", "-rf", "mpi-results"])

def get_subworker_count(worker_node_json):
    if 'SLAVES' not in worker_node_json:
        return 1
    else:
        slaves = worker_node_json['SLAVES']
        total_workers = 1 #self
        for slave_id in slaves:
            total_workers = total_workers + get_subworker_count(slaves[slave_id])
        return total_workers

def get_number_mpi_nodes(config_file):
    with open(config_file) as data_file:
        data_loaded = json.load(data_file)
        return get_subworker_count(data_loaded)

def count_correct_cluster_hits(correct_output, actual_output):
    # Now count the correct hits .. to do so we must first assigne the
    # clustering IDs from the dataset to the IDs our clustering program produced

    # coutner for correct hits
    counter_correct = 0

    # IDs of clusters from the dataset
    cluster_ids = np.unique(correct_output)
    # IDs of clusters from the clustering program
    found_cluster_ids = np.unique(actual_output)
    # For sanity reasons check for outliners
    for i in found_cluster_ids:
        if i > 1000:
            print("WARNING: outliner in output detected! Clusterid: ", i)
    # dict for possible assignements between the two
    bins = {}
    # create bins for each possible assignement
    for key in [(x,y) for x in cluster_ids for y in found_cluster_ids]:
        bins[key] = 0
    # Iterate over data and record hits for each bin
    for i in range(0, actual_output.shape[0]):
        bins[(correct_output[i], actual_output[i])] = bins[(correct_output[i], actual_output[i])] + 1
    #print(bins)

    # Now find the optimal assignement for each cluster id from the
    # dataset. We want to maximize the correct hits (important if clusters
    # of differenct size were merged by the clustering program)
    correct_assignements = {}
    # find optimal assignement by looking for the bins with the most hits
    for cluster in cluster_ids:
        current_maximum = -1
        to_remove_key = (0, 0)
        # find current best bin
        for (x,y) in bins:
            if bins[(x,y)] > current_maximum:
                current_maximum = bins[(x,y)]
                combination_key = (x,y)
                correct_assignements[x] = y
                to_remove_key = (x, y)
        # Add this bin to correct hits
        counter_correct = counter_correct + current_maximum
        # Remove all assignement combinations made impossible by this assignement
        to_remove_list = []
        for (x,y) in bins:
            if x == to_remove_key[0] or y == to_remove_key[1]:
                to_remove_list.append((x,y)) #cannot del in bins whilst iterating
        for d in to_remove_list:
            del bins[d] # but we can delete in this loop

    #print(correct_assignements)
    return counter_correct

if __name__ == '__main__':
    # dimensions, clusters, setsize, abweichung, rauschensize
    parser = argparse.ArgumentParser(description='Test the OpenCL and/or MPI clustering\
    pipeline for correctness. This script will generate simple (but huge) datasets and verify\
    whether the OpenCL (or) MPI clustering examples are sorting the datapoints in the right clusters.')
    parser.add_argument('--opencl_config_folder', type=str, required=True,
                        help='Folder containing the OpenCL config files for the machine we are\
                        testing upon. Warning: Since this script will iterate over all configs\
                        in this folder, the folder should only contain valid OpenCL cfg files.')
    parser.add_argument('--mpi_config_folder', type=str, required=True,
                        help='Folder containing the MPI config files for the network configurations\
                    we want to test. Warning: Since this script will iterate over the configs\
                        this folder should only contain valid MPI cfg files!')
    parser.add_argument('--test_opencl', dest='opencl_flag', action='store_const', const=True, default=False,
                        help='If used, the script will test the OpenCL example clustering_cmd \
                        for correctness')
    parser.add_argument('--test_mpi', dest='mpi_flag', action='store_const', const=True, default=False,
                        help='If used, the script will test the MPI example examples_mpi \
                        for correctness')
    parser.add_argument('--compare_full_output', dest='compare_full_flag',
                        action='store_const',
                        const=True, default=False,
                        help='If used, the script will the output of the OpenCL and the MPI\
                        clustering examples. To be used together with --test_opencl and --test_mpi')
    parser.add_argument('--dimension_args', type=int, required=True, nargs=3,
                        help='Dimension range for to testing datasets. Order of arguments: START_DIMENSION\
                    STEP END_DIMENSION ')
    parser.add_argument('--size_args', type=int, required=True, nargs=3,
                        help='Size (number of datapoints) range for to testing datasets.\
                              Order of arguments: START_SIZE STEP END_SIZE')
    parser.add_argument('--level_args', type=int, required=True, nargs=3,
                        help='Level of grid.\
                              Order of arguments: START_SIZE STEP END_SIZE')
    args = parser.parse_args()

    # Check arguments
    if (args.compare_full_flag and (not args.opencl_flag)) or (args.compare_full_flag and (not args.mpi_flag)):
        print("ERROR: --compare_full_output requires both --test_opencl and --test_mpi")
        quit()
    if not args.opencl_config_folder.endswith("/"):
        args.opencl_config_folder = args.opencl_config_folder + "/"
    if not args.mpi_config_folder.endswith("/"):
        args.mpi_config_folder = args.mpi_config_folder + "/"

    subprocess.run(["mkdir", "dataset-tmp"])
    subprocess.run(["mkdir", "ocl-results"])
    subprocess.run(["mkdir", "mpi-results"])

    # Cleanup trap
    atexit.register(cleanup)

    print("#--------------------------------------------------------------------------------------------")
    print("Status , Recall\t\t, Pipeline, Dim, Size, Level, OCL Config, MPI Config")
    # Iterate over dataset dimensions
    dataset_arg = "--datasetFileName=dataset-tmp/input-data.arff"
    for dim in range(args.dimension_args[0], args.dimension_args[2], args.dimension_args[1]):
        # Iterate over dataset size
        for size in range(args.size_args[0], args.size_args[2], args.size_args[1]):
            # Generate and store dataset (using random seed 0)
            dataset1, Y1, centers = generate_dataset(dim, 10, size, 0.05, 0, 20, True)
            f = open("dataset-tmp/input-data.arff", "w")
            write_arff_header(f, dim, "dataset-tmp/input-data.arff", False)
            write_csv_without_erg(f, dim, dataset1)
            f.close()
            print("#--------------------------------------------------------------------------------------------")
            # Iterate over level
            for level in range(args.level_args[0], args.level_args[2], args.level_args[1]):
                level_arg = "--level=" + str(level)
                # Iterate over OCL configurations
                for opencl_config_name in os.listdir(args.opencl_config_folder):
                    if not opencl_config_name.endswith(".cfg"):
                        continue
                    oclconf_arg = "--config=" + args.opencl_config_folder + opencl_config_name
                    if args.opencl_flag:
                        if os.path.exists("ocl-results/raw-clusters.txt"):
                            subprocess.run(["rm", "ocl-results/raw-clusters.txt"])
                        if os.path.exists("ocl-results/rhs.txt"):
                            subprocess.run(["rm", "ocl-results/rhs.txt"])
                        if os.path.exists("ocl-results/density-coefficients.txt"):
                            subprocess.run(["rm", "ocl-results/density-coefficients.txt"])
                        if os.path.exists("ocl-results/pruned-knn.txt"):
                            subprocess.run(["rm", "ocl-results/pruned-knn.txt"])
                        # clustering run
                        start = time.time()
                        subprocess.run(["../examplesOCL/clustering_cmd", dataset_arg, "--k=5",
                                        oclconf_arg, level_arg, "--epsilon=0.001","--lambda=0.000001",
                                        "--cluster_file=ocl-results/raw-clusters.txt",
                                        "--rhs_erg_file=ocl-results/rhs.txt",
                                        "--density_coefficients_file=ocl-results/density-coefficients.txt",
                                        "--pruned_knn_file=ocl-results/pruned-knn.txt"],
                                       stdout=subprocess.PIPE)
                        end = time.time()
                        duration = end - start
                        # load real result from clustering run
                        counter_correct = 0
                        if os.path.exists("ocl-results/raw-clusters.txt"):
                            Y1_run = np.loadtxt("ocl-results/raw-clusters.txt", int)
                            counter_correct = count_correct_cluster_hits(Y1, Y1_run)

                        # Output ocl test results
                        percent = round(counter_correct/ Y1.shape[0] * 100.0, 3)
                        if percent >= 99.0:
                            print("SUCCESS, %3.3f"% percent, "%,", " %3.3fs"% duration, "\t, OCL, ", dim, ",",
                                    size, ",", level, ",", opencl_config_name)
                        else:
                            print("FAILED , %3.3f"% percent, "%,", "%3.3fs"% duration, "\t, OCL, ", dim, ",",
                                    size, ",", level, ",", opencl_config_name)
                    if args.mpi_flag:
                        # Iterate over MPI configurations
                        for mpi_config_name in os.listdir(args.mpi_config_folder):
                            if not mpi_config_name.endswith(".cfg"):
                                continue
                            mpiconf_arg = "--MPIconfig=" + args.mpi_config_folder + mpi_config_name
                            number_mpi_processes = "-n=" + str(get_number_mpi_nodes(\
                                                   args.mpi_config_folder + mpi_config_name))
                            if os.path.exists("mpi-results/raw-clusters.txt"):
                                subprocess.run(["rm", "mpi-results/raw-clusters.txt"])
                            if os.path.exists("mpi-results/rhs.txt"):
                                subprocess.run(["rm", "mpi-results/rhs.txt"])
                            if os.path.exists("mpi-results/density-coefficients.txt"):
                                subprocess.run(["rm", "mpi-results/density-coefficients.txt"])
                            if os.path.exists("mpi-results/pruned-knn.txt"):
                                subprocess.run(["rm", "mpi-results/pruned-knn.txt"])
                            start = time.time()
                            subprocess.run(["mpirun", number_mpi_processes, "../examplesMPI/mpi_examples",
                                            dataset_arg, mpiconf_arg,
                                            oclconf_arg, level_arg, "--epsilon=0.001", "--lambda=0.000001",
                                            "--cluster_file=mpi-results/raw-clusters.txt", "--k=5",
                                            "--rhs_erg_file=mpi-results/rhs.txt",
                                            "--density_coefficients_file=mpi-results/density-coefficients.txt",
                                            "--pruned_knn_file=mpi-results/pruned-knn.txt"],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            end = time.time()
                            duration = end - start
                            # load real result from clustering run
                            counter_correct = 0
                            if os.path.exists("mpi-results/raw-clusters.txt"):
                                Y1_run = np.loadtxt("mpi-results/raw-clusters.txt", int)
                                subprocess.run(["rm", "mpi-results/raw-clusters.txt"])
                                counter_correct = count_correct_cluster_hits(Y1, Y1_run)
                            percent = round(counter_correct/ Y1.shape[0] * 100.0, 3)
                            success_rhs = False
                            if os.path.exists("mpi-results/rhs.txt"):
                                success_rhs = filecmp.cmp("ocl-results/rhs.txt",
                                                        "mpi-results/rhs.txt", False)
                            success_density_coefficients = False
                            if os.path.exists("mpi-results/density-coefficients.txt"):
                                success_density_coefficients = filecmp.cmp("ocl-results/density-coefficients.txt",
                                                        "mpi-results/density-coefficients.txt", False)
                            success_pruned_knn = False
                            if os.path.exists("mpi-results/pruned-knn.txt"):
                                success_pruned_knn = filecmp.cmp("ocl-results/pruned-knn.txt",
                                                        "mpi-results/pruned-knn.txt", False)
                            if percent >= 99.0:
                                print("SUCCESS, %3.3f"% percent, "%,", " %3.3fs"% duration, "\t, MPI, ", dim, ",",
                                      size, ",", level, ",", success_rhs, ",",
                                      success_density_coefficients, ",", success_pruned_knn,
                                      ",", opencl_config_name, "\t,", mpi_config_name)
                            else:
                                print("FAILED , %3.3f"% percent, "%,", "%3.3fs"% duration, "\t, MPI, ", dim, ",",
                                      size, ",", level, ",", success_rhs, ",",
                                      success_density_coefficients, ",", success_pruned_knn,
                                      ",", opencl_config_name, "\t,", mpi_config_name)
                            pass
                    pass
