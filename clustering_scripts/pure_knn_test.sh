#!/bin/bash

num_clusters=10
cluster_size=100000

for dim in 3 5 10; do
    for noise in "" "_noise"; do
        # echo "dim: ${dim} noise: ${noise}"
        echo "dataset: final_paper_datasets/gaussian_c${num_clusters}_size${cluster_size}_dim${dim}.arff"

        ./datadriven/examplesOCL/clustering_cmd --datasetFileName final_paper_datasets/gaussian_c${num_clusters}_size${cluster_size}_dim${dim}.arff --config config_ocl_float_i76700k.cfg --k 6 --lambda 0.00001 --level 4 --threshold 0.0 --print_cluster_sizes 1 | grep 'cluster'
        # r=`./datadriven/examplesOCL/clustering_cmd --datasetFileName final_paper_datasets/gaussian_c${num_clusters}_size${cluster_size}_dim${dim}.arff --config config_ocl_float_i76700k.cfg --k 6 --lambda 0.00001 --level 4 --threshold 0.0 --print_cluster_sizes 1 | grep 'cluster'`
        # echo ${r} | grep 'detected clusters:'

    done
done 
