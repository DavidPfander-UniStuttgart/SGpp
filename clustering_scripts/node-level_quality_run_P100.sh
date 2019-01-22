#!/bin/bash
set -x

# 1M, 10C, lambda=1E-5 t=667 l=6 epsilon=1E-2
dataset_size=1000000
clusters=10
lambda=1E-5
threshold=667
level=6
file_suffix=${dataset_size}s_${clusters}c_${lambda}lam_${level}l
nvidia-smi dmon -i 0 -d 1sec -f results_WPDM18/P100_quality_freq_${file_suffix}.log &
RUNNING_PID=$!
sleep 3

./datadriven/examplesOCL/clustering_cmd --config OCL_configs/config_ocl_float_P100.cfg --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c${clusters}_size${dataset_size}_dim10_noise --level=${level} --lambda=${lambda} --k 6 --epsilon 0.01 --threshold ${threshold} --print_cluster_sizes 1 --target_clusters 100 > results_WPDM18/P100_quality_run_${file_suffix}.log 2>&1

kill ${RUNNING_PID}

# 1M, 100C, lambda=1E-6 t=556 l=7 epsilon=1E-2
dataset_size=1000000
clusters=100
lambda=1E-6
threshold=556
level=7
file_suffix=${dataset_size}s_${clusters}c_${lambda}lam_${level}l
nvidia-smi dmon -i 0 -d 1sec -f results_WPDM18/P100_quality_freq_${file_suffix}.log &
RUNNING_PID=$!
sleep 3

./datadriven/examplesOCL/clustering_cmd --config OCL_configs/config_ocl_float_P100.cfg --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c${clusters}_size${dataset_size}_dim10_noise --level=${level} --lambda=${lambda} --k 6 --epsilon 0.01 --threshold ${threshold} --print_cluster_sizes 1 --target_clusters 100 > results_WPDM18/P100_quality_run_${file_suffix}.log 2>&1

kill ${RUNNING_PID}

# 1M, 100C, lambda=1E-7 t=1056 l=7 epsilon=1E-2
dataset_size=1000000
clusters=100
lambda=1E-7
threshold=1056
level=7
file_suffix=${dataset_size}s_${clusters}c_${lambda}lam_${level}l
nvidia-smi dmon -i 0 -d 1sec -f results_WPDM18/P100_quality_freq_${file_suffix}.log &
RUNNING_PID=$!
sleep 3

./datadriven/examplesOCL/clustering_cmd --config OCL_configs/config_ocl_float_P100.cfg --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c${clusters}_size${dataset_size}_dim10_noise --level=${level} --lambda=${lambda} --k 6 --epsilon 0.01 --threshold ${threshold} --print_cluster_sizes 1 --target_clusters 100 > results_WPDM18/P100_quality_run_${file_suffix}.log 2>&1

kill ${RUNNING_PID}

# 10M, 10C, lambda=1E-5 t=1167 l=7 epsilon=1E-2
dataset_size=10000000
clusters=10
lambda=1E-5
threshold=1167
level=7
file_suffix=${dataset_size}s_${clusters}c_${lambda}lam_${level}l
nvidia-smi dmon -i 0 -d 1sec -f results_WPDM18/P100_quality_freq_${file_suffix}.log &
RUNNING_PID=$!
sleep 3

./datadriven/examplesOCL/clustering_cmd --config OCL_configs/config_ocl_float_P100.cfg --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c${clusters}_size${dataset_size}_dim10_noise --level=${level} --lambda=${lambda} --k 6 --epsilon 0.01 --threshold ${threshold} --print_cluster_sizes 1 --target_clusters 100 > results_WPDM18/P100_quality_run_${file_suffix}.log 2>&1

kill ${RUNNING_PID}

# 10M, 100C, lambda=1E-6 t=1000 l=7 epsilon=1E-2
dataset_size=10000000
clusters=100
lambda=1E-6
threshold=1000
level=7
file_suffix=${dataset_size}s_${clusters}c_${lambda}lam_${level}l
nvidia-smi dmon -i 0 -d 1sec -f results_WPDM18/P100_quality_freq_${file_suffix}.log &
RUNNING_PID=$!
sleep 3

./datadriven/examplesOCL/clustering_cmd --config OCL_configs/config_ocl_float_P100.cfg --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c${clusters}_size${dataset_size}_dim10_noise --level=${level} --lambda=${lambda} --k 6 --epsilon 0.01 --threshold ${threshold} --print_cluster_sizes 1 --target_clusters 100 > results_WPDM18/P100_quality_run_${file_suffix}.log 2>&1

kill ${RUNNING_PID}
