#!/bin/bash

no_runs=3
hn=`hostname`
if [ "${hn}" = "pcsgs07" ]; then
    device_name='w8100'
    dataset_directory='datasets_WPDM18/'
elif [ ${hn} = "argon-tesla1" ]; then
    device_name='P100'
    dataset_directory='datasets_WPDM18/'
fi

echo "running node-level experiments on ${hn}"

echo "1M dataset experiments..."
for run in $(seq 0 ${no_runs}); do
    lambda=1.78E-7
    threshold=1277.78
    cmd="./datadriven/examplesOCL/clustering_cmd --binary_header_file ${dataset_directory}gaussian_c100_size1000000_dim10_noise --config OCL_configs/config_ocl_float_${device_name}.cfg --lambda ${lambda}  --epsilon 1E-3 --threshold ${threshold}  --level 7 --k 6 --print_cluster_sizes 1 > results_WPDM18/${hn}_perf_1m_fast_run${run}.log"
    echo "$cmd"
    ${cmd}
done
# echo "10M dataset experiments..."
# for run in $(seq 0 ${no_runs}); do 
#     ./datadriven/examplesOCL/clustering_cmd --binary_header_file ${dataset_directory}gaussian_c100_size10000000_dim10_noise --config OCL_configs/config_ocl_float_${device_name}.cfg --lambda 1.78E-6 --epsilon 1E-3 --threshold 1407.41 --level 8 --k 6 --print_cluster_sizes 1 > results_WPDM18/${hn}_perf_10m_fast_run${run}.log
# done
