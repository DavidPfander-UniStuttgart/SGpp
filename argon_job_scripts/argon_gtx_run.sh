#!/bin/bash

source source_autotunetmp_generic.sh

set_size=1000000
noise= # empty or "_noise"
level=6
./datadriven/examplesOCL/clustering_cmd --config config_ocl_float_gtx1080ti.cfg --threshold=0.7 --datasetFileName final_paper_datasets/gaussian_c100_size${set_size}_dim10${noise}.arff --level=${level} --lambda=0.00001 --k 5 --write_knn_graph true --scenario_name gaussian_c100_size${set_size}_dim10${noise}

# ./datadriven/examplesOCL/clustering_cmd --config config_ocl_float_gtx1080ti.cfg --threshold=0.7 --datasetFileName final_paper_datasets/gaussian_c100_size1000000_dim10_noise.arff --level=7 --lambda=0.00001 --k 5 --write_knn_graph true --scenario_name gaussian_c100_size1000000_dim10_noise

# ./datadriven/examplesOCL/clustering_cmd --config config_ocl_float_gtx1080ti.cfg --threshold=0.7 --datasetFileName paper_datasets/gaussian_c100_size10000000_dim10_noise.arff --level=8 --lambda=0.00001 --k 5

# ./datadriven/examplesOCL/clustering_cmd --config config_ocl_float_Gold5120.cfg --threshold=0.7 --datasetFileName paper_datasets/gaussian_c100_size10000000_dim10.arff --level=8 --lambda=0.00001 --k 5
# ./datadriven/examplesOCL/clustering_cmd --config config_ocl_float_gtx1080ti.cfg --threshold=0.7 --datasetFileName paper_datasets/gaussian_c100_size10000000_dim10_noise.arff --level=8 --lambda=0.00001 --k 5
