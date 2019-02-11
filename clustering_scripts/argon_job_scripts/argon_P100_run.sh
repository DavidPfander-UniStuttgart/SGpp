#!/bin/bash

source source_autotunetmp_generic.sh
./datadriven/examplesOCL/clustering_cmd --config config_ocl_float_P100.cfg --threshold=0.7 --datasetFileName final_paper_datasets/gaussian_c100_size1000000_dim10.arff --level=4 --lambda=0.00001 --k 5
# ./datadriven/examplesOCL/clustering_cmd --config config_ocl_float_P100.cfg --threshold=0.7 --datasetFileName paper_datasets/gaussian_c100_size10000000_dim10_noise.arff --level=8 --lambda=0.00001 --k 5
