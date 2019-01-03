#!/bin/bash
set -x

source source_autotunetmp_generic.sh

cd final_paper_datasets

for set_size in 1000000 10000000 100000000; do
    for dim in 5 10; do
        for noise in "" "_noise"; do
            dataset_file=gaussian_c100_size${set_size}_dim${dim}${noise}
            ../datadriven/examplesOCL/split_dataset --input_filename ${dataset_file}.arff  --output_filename ${dataset_file} --compression false
        done
    done
done
