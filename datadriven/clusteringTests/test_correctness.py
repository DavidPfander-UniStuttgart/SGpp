from generate_datasets import generate_dataset
from generate_datasets import write_arff_header
from generate_datasets import write_csv_without_erg
import argparse
import os
import subprocess

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
    print("Starting correctness test...")
    print("----------------------------")
    # Iterate over dataset dimensions
    dataset_arg = "--datasetFileName=dataset-tmp/input-data.arff"
    for dim in range(args.dimension_args[0], args.dimension_args[2], args.dimension_args[1]):
        # Iterate over dataset size
        for size in range(args.size_args[0], args.size_args[2], args.size_args[1]):
            # Generate and store dataset
            dataset1, Y1, centers = generate_dataset(dim, 10, size, 0.05, 0, 20)
            f = open("dataset-tmp/input-data.arff", "w")
            write_arff_header(f, dim, "dataset-tmp/input-data.arff", False)
            write_csv_without_erg(f, dim, dataset1)
            f.close()
            # Iterate over level
            for level in range(args.level_args[0], args.level_args[2], args.level_args[1]):
                level_arg = "--level=" + str(level)
                # Iterate over OCL configurations
                for opencl_config_name in os.listdir(args.opencl_config_folder):
                    if not opencl_config_name.endswith(".cfg"):
                        continue
                    oclconf_arg = "--config=" + args.opencl_config_folder + opencl_config_name
                    if args.opencl_flag:
                        subprocess.run(["../examplesOCL/clustering_cmd", dataset_arg,
                                        oclconf_arg, level_arg], stdout=subprocess.PIPE)
                        print("-> OpenCL test -> Dimension ", dim, "-> Size ",
                              size, "-> Level ", level, "-> OpenCL config ",
                            opencl_config_name,"\t\t:")
                        pass
                    if args.mpi_flag:
                        # Iterate over MPI configurations
                        for mpi_config_name in os.listdir(args.mpi_config_folder):
                            if not mpi_config_name.endswith(".cfg"):
                                continue
                            print("   -> MPI test -> Dimension ", dim, "-> Size ",
                                    size, "-> Level ", level, "-> MPI config ", mpi_config_name,"\t\t:")
                        pass
                    pass
    subprocess.run(["rm", "-rf", "dataset-tmp"])
