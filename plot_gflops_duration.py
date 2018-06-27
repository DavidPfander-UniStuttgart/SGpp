#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import csv
import pylab

clusters=3
deviceName="config_ocl_float_i76700k.cfg" # i76700k
precision="float"

# for dim in range(2, 3, 2):
for dim in range(2, 11, 2):

    results_file = "results/results_gaussian_c" + str(clusters) + "_dim" + str(dim) + "_" + deviceName + "_" + precision + ".csv"

    csvfile = open(results_file, 'r')
    reader = csv.reader(csvfile, delimiter=';', quotechar='#')
    headers = next(reader, None)
    print(headers)
    columns = [[] for i in range(len(headers))]
    row_count = 0
    for row in reader:
        for field_index in range(len(headers)):
            columns[field_index] += [float(row[field_index])]

    # for row in columns:
    #     print(row)

    header_map = {}
    counter = 0
    for header in headers:
        header_map[header] = counter
        counter += 1

    print(columns[0])
    print(columns[3])
    pylab.plot(columns[header_map['dataset_size']], columns[header_map['avr_gflops_generate_b']], label='avr_gflops_generate_b')
    pylab.plot(columns[header_map['dataset_size']], columns[header_map['avr_gflops_density']], label='avr_gflops_density')
    pylab.plot(columns[header_map['dataset_size']], columns[header_map['gflops_create_graph']], label='gflops_create_graph')
    pylab.plot(columns[header_map['dataset_size']], columns[header_map['gflops_prune_graph']], label='gflops_prune_graph')
    pylab.legend(loc='upper left')
    # pylab.show()
    pylab.savefig("graphs/" + "gflops_gaussian_c" + str(clusters) + "_dim" + str(dim) + ".png")
    pylab.clf()

    pylab.plot(columns[header_map['dataset_size']], columns[header_map['total_duration_generate_b']], label='avr_duration_generate_b')
    pylab.plot(columns[header_map['dataset_size']], columns[header_map['total_duration_density']], label='avr_duration_density')
    pylab.plot(columns[header_map['dataset_size']], columns[header_map['duration_create_graph']], label='duration_create_graph')
    pylab.plot(columns[header_map['dataset_size']], columns[header_map['duration_prune_graph']], label='duration_prune_graph')
    pylab.legend(loc='upper left')
    # pylab.show()
    pylab.savefig("graphs/" + "duration_gaussian_c" + str(clusters) + "_dim" + str(dim) + ".png")
    pylab.clf()

    # fig, ax = plt.figure()
    # ax.plot(columns[header_map['dataset_size']], header_map['avr_gflops_generate_b'], label='avr_gflops_generate_b')
    # ax.legend()
    # fig.show()
