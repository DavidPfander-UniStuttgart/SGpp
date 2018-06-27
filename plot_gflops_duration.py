#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import csv
import pylab
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

for dim in range(2, 11, 2):

    results_file = "results/results_gaussian_c" + str(clusters) + "_dim" + str(dim) + "_" + args.device_name + "_" + args.precision + ".csv"

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

    for kernel_gflops in kernels_gflops:
        pylab.plot(columns[header_map['dataset_size']], columns[header_map[kernel_gflops]], label=kernel_gflops)
    pylab.legend(loc='upper left')
    pylab.title("Performance for gaussian dataset, clusters = " + str(clusters) + ", dim = " + str(dim))
    pylab.xlabel("dataset size")
    pylab.ylabel("GFLOPS (" + str(args.precision) + ")")
    pylab.savefig("graphs/" + "gflops_gaussian_c" + str(clusters) + "_dim" + str(dim) + "_" + args.device_name + "_" + args.precision + ".png")
    pylab.clf()

    for kernel_duration in kernels_duration:
        pylab.plot(columns[header_map['dataset_size']], columns[header_map[kernel_duration]], label=kernel_duration)

    pylab.legend(loc='upper left')
    pylab.title("Duration for gaussian dataset, clusters = " + str(clusters) + ", dim = " + str(dim) + ", " + args.precision)
    pylab.xlabel("dataset size")
    pylab.ylabel("duration (s)")
    pylab.savefig("graphs/" + "duration_gaussian_c" + str(clusters) + "_dim" + str(dim) + "_" + args.device_name + "_" + args.precision + ".png")
    pylab.clf()

for kernel_gflops in kernels_gflops:
    for dim in range(2, 11, 2):
        results_file = "results/results_gaussian_c" + str(clusters) + "_dim" + str(dim) + "_" + args.device_name + "_" + args.precision + ".csv"
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

    pylab.legend(loc='upper left')
    pylab.title("Performance of " + kernels_print_name[kernel_gflops] + " kernel")
    pylab.xlabel("dataset size")
    pylab.ylabel("GFLOPS (" + str(args.precision) + ")")
    pylab.savefig("graphs/" + "kernel_" + kernel_gflops + "_gflops_gaussian_c" + str(clusters) + "_" + args.device_name + "_" + args.precision + ".png")
    pylab.clf()
