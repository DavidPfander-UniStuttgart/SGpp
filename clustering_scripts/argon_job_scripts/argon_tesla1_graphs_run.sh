#!/bin/bash

source source_autotunetmp_generic.sh

for set_size in 1000000 10000000 100000000; do
    if [ ${set_size} -eq 1000000 ]; then
        level=6
    elif [ ${set_size} -eq 10000000 ]; then
        level=7
    elif [ ${set_size} -eq 100000000 ]; then
        level=8
    fi
    for noise in "" "_noise"; do
        ./datadriven/examplesOCL/clustering_cmd --config config_ocl_float_P100.cfg --threshold=0.7 --binary_header_filename final_paper_datasets_binary/gaussian_c100_size${set_size}_dim10${noise} --level=${level} --lambda=0.00001 --k 5 --write_knn_graph true --scenario_name gaussian_c100_size${set_size}_dim10${noise} > graphs_${set_size}s_${level}l${noise}.log 2>&1
    done
done
