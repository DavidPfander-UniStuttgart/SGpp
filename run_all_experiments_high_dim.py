#!/usr/bin/python3

import shlex
import subprocess
import re
import argparse
parser = argparse.ArgumentParser(description='Run all high dim experiments.')
parser.add_argument('--with-compression', dest='with_compression', action='store_true')
parser.set_defaults(with_compression=False)
parser.add_argument('--fixed-grid-points', dest='fixed_grid_points', action='store_true')
parser.set_defaults(with_compression=False)
parser.add_argument('--use-32-bits-compression', dest='use_32_bits_compression', action='store_true')
parser.set_defaults(use_32_bits_compression=False)
parser.add_argument('--use-fewer-registers', dest='use_fewer_registers', action='store_true')
parser.set_defaults(use_fewer_registers=False)
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument('--device_name', required=True)
requiredNamed.add_argument('--config_file', required=True)
requiredNamed.add_argument('--precision', required=True)


args = parser.parse_args()

print("config_file: " + args.config_file)
print("precision: " + args.precision)
print("device_name: " + args.device_name)
print("with_compression: " + str(args.with_compression))

# noise = 0
# print("noise: " + )

# common
# levels = {2: 8, 4: 7, 6: 6, 8: 5, 10: 4}
# levels = {2: 5, 4: 5, 6: 5, 8: 5, 10: 5}

base_dim = 4
base_dim_level = 9
level = 4

print("levels: " + str(level))
lambdas = 1E-2
print("lambda: " + str(lambdas))

if args.use_32_bits_compression:
    max_dim_to_test = 32
else:
    max_dim_to_test = 64

# refinement
refinement_steps=0
refinement_points=0
coarsen_points=10000
coarsen_threshold=1E-3

CSV_SEP = ";"

# for dim in range(2, 3, 2):
for level in [level]:
    if args.with_compression:
        if args.use_32_bits_compression:
            compression_length_string = "_32bits_comp"
        else:
            compression_length_string = "_64bits_comp"
    else:
        compression_length_string = "_no_comp"
    if args.fixed_grid_points:
        fixed_grid_points_string = "_fixed_dim"
    else:
        fixed_grid_points_string = ""
    if args.use_fewer_registers:
        fewer_registers_string = "_fewer_regs"
    else:
        fewer_registers_string = ""

    resultsFileName = "results/results_friedman2_high_dim_" + args.device_name + "_" + args.precision + compression_length_string + fixed_grid_points_string + fewer_registers_string + ".csv"

    # if args.fixed_grid_points:
    #     resultsFileName = "results/results_friedman2_high_dim_" + args.device_name + "_" + args.precision + "_compression" + str(args.with_compression) + "_fixed_dim" + str(base_dim) + ".csv"
    # else:
    #     resultsFileName = "results/results_friedman2_high_dim_" + args.device_name + "_" + args.precision + "_compression" + str(args.with_compression) + ".csv"
    print("resultsFileName:", resultsFileName)
    f_result = open(resultsFileName, "w")
    f_result.write("dim" + CSV_SEP + "dataset_size" + CSV_SEP + "refinement_steps" + CSV_SEP + "total_duration_generate_b" + CSV_SEP + "avr_gflops_generate_b" + CSV_SEP + "total_duration_density" + CSV_SEP + "avr_gflops_density" + CSV_SEP + "density_iterations" + CSV_SEP + "avr_density_duration_per_iteration\n")
    for dim in range(4, max_dim_to_test, 1):

        # for dataset_size in [200]:
        # for dataset_size in chain(range(20000, 110000, 20000), range(200000, 1100000, 200000)):
        for dataset_size in [500000]:
           dataset_file_name = "datasets/friedman/friedman2_filled_dim" + str(dim) + "_" + str(dataset_size) + ".arff"
           print("experiments for " + dataset_file_name)
           if args.fixed_grid_points:
               cmd = "./datadriven/examplesOCL/density_cmd_filled_dim --datasetFileName " + dataset_file_name + " --base_dim " + str(base_dim) + " --base_dim_level " + str(base_dim_level) + " --lambda " + str(lambdas) + " --config " + args.config_file + " --refinement_steps " + str(refinement_steps) + " --refinement_points " + str(refinement_points) + " --coarsen_points " + str(coarsen_points) + " --coarsen_threshold " + str(coarsen_threshold)
           else:
               cmd = "./datadriven/examplesOCL/density_cmd --datasetFileName " + dataset_file_name + " --level " + str(level) + " --lambda " + str(lambdas) + " --config " + args.config_file + " --refinement_steps " + str(refinement_steps) + " --refinement_points " + str(refinement_points) + " --coarsen_points " + str(coarsen_points) + " --coarsen_threshold " + str(coarsen_threshold)
           print("cmd: " + cmd)
           p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE)
           output = p.communicate()[0]
           output = output.decode('utf-8')
           print(output)

           # avr generate_b gflops
           count_generate_b = 0
           avr_gflops_generate_b = 0
           for g in re.finditer(r"flops_generate_b: (.*?) GFLOPS", output):
               avr_gflops_generate_b += float(g.group(1))
               count_generate_b += 1
           avr_gflops_generate_b /= float(count_generate_b)

           # total generate_b duration
           total_duration_generate_b = 0
           for g in re.finditer(r"last_duration_generate_b: (.*?)s", output):
               total_duration_generate_b += float(g.group(1))

           # avr density gflops
           count_density = 0
           avr_gflops_density = 0
           for g in re.finditer(r"flops_density: (.*?) GFLOPS", output):
              avr_gflops_density += float(g.group(1))
              count_density += 1
           avr_gflops_density /= float(count_density)

           # total density duration
           total_duration_density = 0
           for g in re.finditer(r"acc_duration_density: (.*?)s", output):
              total_duration_density += float(g.group(1))

           density_iterations = 0
           for g in re.finditer(r"act_it: (.*?)\n", output):
              density_iterations += float(g.group(1))

           avr_density_duration_per_iteration = total_duration_density / density_iterations

           f_result.write(str(dim) + CSV_SEP + str(dataset_size) + CSV_SEP + str(refinement_steps) + CSV_SEP + str(total_duration_generate_b) + CSV_SEP + str(avr_gflops_generate_b) + CSV_SEP + str(total_duration_density) + CSV_SEP + str(avr_gflops_density) + CSV_SEP + str(density_iterations) + CSV_SEP + str(avr_density_duration_per_iteration) + "\n")
