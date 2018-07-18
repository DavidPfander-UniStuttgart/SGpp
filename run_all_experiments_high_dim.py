#!/usr/bin/python3

# import sys
import shlex
import subprocess
import re
# from itertools import chain

import argparse
parser = argparse.ArgumentParser(description='Run all high dim experiments.')
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument('--device_name', required=True)
requiredNamed.add_argument('--config_file', required=True)
requiredNamed.add_argument('--precision', required=True)
args = parser.parse_args()

print("config_file: " + args.config_file)
print("precision: " + args.precision)
print("device_name: " + args.device_name)

# noise = 0
# print("noise: " + )

# common
# levels = {2: 8, 4: 7, 6: 6, 8: 5, 10: 4}
# levels = {2: 5, 4: 5, 6: 5, 8: 5, 10: 5}
levels = [1]

print("levels: " + str(levels))
lambdas = 1E-2
print("lambda: " + str(lambdas))

# refinement
refinement_steps=0
refinement_points=0
coarsen_points=10000
coarsen_threshold=1E-3

CSV_SEP = ";"

# for dim in range(2, 3, 2):
for level in levels:
    f_result = open("results/results_friedman2_high_dim_gaussian_" + args.device_name + "_" + args.precision + ".csv", "w")
    f_result.write("dataset_size" + CSV_SEP + "refinement_steps" + CSV_SEP + "total_duration_generate_b" + CSV_SEP + "avr_gflops_generate_b" + CSV_SEP + "total_duration_density" + CSV_SEP + "avr_gflops_density\n")
    for dim in range(4, 10, 1):

        # for dataset_size in [200]:
        # for dataset_size in chain(range(20000, 110000, 20000), range(200000, 1100000, 200000)):
        for dataset_size in [500000]:            
           dataset_file_name = "datasets/friedman/friedman2_filled_dim" + str(dim) + "_" + str(dataset_size) + ".arff"
           print("experiments for " + dataset_file_name)
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

           f_result.write(str(dataset_size) + CSV_SEP + str(refinement_steps) + CSV_SEP + str(total_duration_generate_b) + CSV_SEP + str(avr_gflops_generate_b) + CSV_SEP + str(total_duration_density) + CSV_SEP + str(avr_gflops_density) + "\n")
