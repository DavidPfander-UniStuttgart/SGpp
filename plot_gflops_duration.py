#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
font = {'family' : 'Times',
        'weight' : 'normal',
        'size'   : 8}
plt.rc('font', **font)

import csv
import pylab
from pylab import rcParams
rcParams['figure.figsize'] = 4.5, 3.2
# rcParams.update({'figure.autolayout': True})
import argparse

parser = argparse.ArgumentParser(description='Plot graphs')
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument('--device_name', required=True)
requiredNamed.add_argument('--precision', required=True)
clusters=3
args = parser.parse_args()

kernels_gflops = ['avr_gflops_generate_b', 'avr_gflops_density', 'gflops_create_graph', 'gflops_prune_graph']
kernels_duration = ['total_duration_generate_b', 'total_duration_density', 'duration_create_graph', 'duration_prune_graph']
kernels_print_name = {'avr_gflops_generate_b': "density right-hand side", 'avr_gflops_density': "density matrix-vector product", 'gflops_create_graph': "create kNN graph", 'gflops_prune_graph': "prune kNN graph"}
kernels_duration_print_name = {'total_duration_generate_b': "density right-hand side", 'total_duration_density': "density matrix-vector product", 'duration_create_graph': "create kNN graph", 'duration_prune_graph': "prune kNN graph"}

for level in [5, 6]:
    for dim in range(2, 11, 2):

        results_file = "results/results_gaussian_c" + str(clusters) + "_l" + str(level) + "_d" + str(dim) + "_" + args.device_name + "_" + args.precision + ".csv"

        csvfile = open(results_file, 'r')
        reader = csv.reader(csvfile, delimiter=';', quotechar='#')
        headers = next(reader, None)
        print(headers)
        columns = [[] for i in range(len(headers))]
        row_count = 0
        for row in reader:
            for field_index in range(len(headers)):
                columns[field_index] += [float(row[field_index])]

        header_map = {}
        counter = 0
        for header in headers:
            header_map[header] = counter
            counter += 1

        pylab.tight_layout()
        for kernel_gflops in kernels_gflops:
            pylab.plot(columns[header_map['dataset_size']], columns[header_map[kernel_gflops]], label=kernels_print_name[kernel_gflops])
            print("gflops kernel: " + kernel_gflops + ", dim: " + str(dim) + ", level: " + str(level) + " -> last dataset (should be 1m): " + str(columns[header_map[kernel_gflops]][-1]))
        pylab.legend(loc='upper left')
        pylab.title("Performance for gaussian dataset, clusters = " + str(clusters) + ", dim = " + str(dim) + ", level = " + str(level))
        pylab.xlabel("dataset size")
        pylab.ylabel("GFLOPS (" + str(args.precision) + ")")
        pylab.savefig("graphs/" + "gflops_gaussian_c" + str(clusters)  + "_" + args.device_name + "_" + args.precision + "_l" + str(level) + "_d" + str(dim) + ".eps")
        pylab.clf()

        pylab.tight_layout()
        for kernel_duration in kernels_duration:
            pylab.plot(columns[header_map['dataset_size']], columns[header_map[kernel_duration]], label=kernels_duration_print_name[kernel_duration])

        pylab.legend(loc='upper left')
        pylab.title("Duration for gaussian dataset, clusters = " + str(clusters) + ", dim = " + str(dim) + ", " + args.precision + ", level = " + str(level))
        pylab.xlabel("dataset size")
        pylab.ylabel("duration (s)")
        pylab.savefig("graphs/" + "duration_gaussian_c" + str(clusters)  + "_" + args.device_name + "_" + args.precision + "_l" + str(level) + "_d" + str(dim) + ".eps")
        pylab.clf()

    for kernel_gflops in kernels_gflops:
        pylab.tight_layout()
        for dim in range(2, 11, 2):
            results_file = "results/results_gaussian_c" + str(clusters) + "_l" + str(level) + "_d" + str(dim) + "_" + args.device_name + "_" + args.precision + ".csv"
            csvfile = open(results_file, 'r')
            reader = csv.reader(csvfile, delimiter=';', quotechar='#')
            headers = next(reader, None)
            columns = [[] for i in range(len(headers))]
            row_count = 0
            for row in reader:
                for field_index in range(len(headers)):
                    columns[field_index] += [float(row[field_index])]

            header_map = {}
            counter = 0
            for header in headers:
                header_map[header] = counter
                counter += 1

            pylab.plot(columns[header_map['dataset_size']], columns[header_map[kernel_gflops]], label="dim " + str(dim))

        # pylab.legend(loc='upper left')
        pylab.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=5)
        pylab.title("Performance of " + kernels_print_name[kernel_gflops] + " kernel, level = " + str(level))
        pylab.xlabel("dataset size")
        pylab.ylabel("GFLOPS (" + str(args.precision) + ")")
        plt.gcf().subplots_adjust(bottom=0.19)
        pylab.savefig("graphs/" + "kernel_" + kernel_gflops + "_gflops_gaussian_c" + str(clusters) + "_" + args.device_name + "_" + args.precision + "_l" + str(level) + ".eps")
        pylab.clf()
        plt.gcf().subplots_adjust(bottom=0.11)
