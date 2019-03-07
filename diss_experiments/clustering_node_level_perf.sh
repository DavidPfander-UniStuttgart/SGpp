#!/bin/bash
set -x

no_runs=3
hn=`hostname`
if [ "${hn}" = "pcsgs07" ]; then
    device_name='w8100'
    dataset_directory='datasets_WPDM18/'
elif [ ${hn} = "argon-tesla1" ]; then
    device_name='P100'
    dataset_directory='datasets_WPDM18/'
elif [ ${hn} = "argon-epyc" ]; then
    device_name='Vega7'
    dataset_directory='../../DissertationCodeTesla1/SGpp/datasets_WPDM18/'
elif [ ${hn} = "argon-gtx" ]; then
    device_name='P100'
    dataset_directory='../../DissertationCodeTesla1/SGpp/datasets_WPDM18/'
else
    device_name='hazelhen'
    dataset_directory='/lustre/cray/ws8/ws/ipvpfand-SGppClustering/datasets_WPDM18/'
fi

# 1M, 10C, lambda=1E-5 t=667 l=6 epsilon=1E-2
for run in $(seq 0 ${no_runs}); do
    dataset_size=1000000
    clusters=10
    lambda=1E-5
    threshold=667
    level=6
    file_suffix=${dataset_size}s_${clusters}c_${lambda}lam_${level}l_run${run}
    ./datadriven/examplesOCL/clustering_cmd --config OCL_configs/config_ocl_float_${device_name}.cfg --binary_header_filename ${dataset_directory}/gaussian_c${clusters}_size${dataset_size}_dim10_noise --level=${level} --lambda=${lambda} --k 6 --epsilon 0.01 --threshold ${threshold} --print_cluster_sizes 1 --target_clusters ${clusters} > results_diss/${device_name}_perf_quality_${file_suffix}.log 2>&1
done

# 1M, 10C, lambda=1E-6 t=556 l=6 epsilon=1E-2
for run in $(seq 0 ${no_runs}); do
    dataset_size=1000000
    clusters=100
    lambda=1E-6
    threshold=556
    level=7
    file_suffix=${dataset_size}s_${clusters}c_${lambda}lam_${level}l_run${run}
    ./datadriven/examplesOCL/clustering_cmd --config OCL_configs/config_ocl_float_${device_name}.cfg --binary_header_filename ${dataset_directory}/gaussian_c${clusters}_size${dataset_size}_dim10_noise --level=${level} --lambda=${lambda} --k 6 --epsilon 0.01 --threshold ${threshold} --print_cluster_sizes 1 --target_clusters ${clusters} > results_diss/${device_name}_perf_quality_${file_suffix}.log 2>&1
done

# 1M, 100C, lambda=1E-7 t=1056 l=7 epsilon=1E-2
for run in $(seq 0 ${no_runs}); do
    dataset_size=1000000
    clusters=100
    lambda=1E-7
    threshold=1056
    level=7
    file_suffix=${dataset_size}s_${clusters}c_${lambda}lam_${level}l_run${run}
    ./datadriven/examplesOCL/clustering_cmd --config OCL_configs/config_ocl_float_${device_name}.cfg --binary_header_filename ${dataset_directory}/gaussian_c${clusters}_size${dataset_size}_dim10_noise --level=${level} --lambda=${lambda} --k 6 --epsilon 0.01 --threshold ${threshold} --print_cluster_sizes 1 --target_clusters ${clusters} > results_diss/${device_name}_perf_quality_${file_suffix}.log 2>&1
done

# 10M, 10C, lambda=1E-5 t=1167 l=7 epsilon=1E-2
for run in $(seq 0 ${no_runs}); do
    dataset_size=10000000
    clusters=10
    lambda=1E-5
    threshold=1167
    level=7
    file_suffix=${dataset_size}s_${clusters}c_${lambda}lam_${level}l_run${run}
    ./datadriven/examplesOCL/clustering_cmd --config OCL_configs/config_ocl_float_${device_name}.cfg --binary_header_filename ${dataset_directory}/gaussian_c${clusters}_size${dataset_size}_dim10_noise --level=${level} --lambda=${lambda} --k 6 --epsilon 0.01 --threshold ${threshold} --print_cluster_sizes 1 --target_clusters ${clusters} > results_diss/${device_name}_perf_quality_${file_suffix}.log 2>&1
done

# 10M, 100C, lambda=1E-6 t=1000 l=7 epsilon=1E-2
for run in $(seq 0 ${no_runs}); do
    dataset_size=10000000
    clusters=100
    lambda=1E-6
    threshold=1000
    level=7
    file_suffix=${dataset_size}s_${clusters}c_${lambda}lam_${level}l_run${run}
    ./datadriven/examplesOCL/clustering_cmd --config OCL_configs/config_ocl_float_${device_name}.cfg --binary_header_filename ${dataset_directory}/gaussian_c${clusters}_size${dataset_size}_dim10_noise --level=${level} --lambda=${lambda} --k 6 --epsilon 0.01 --threshold ${threshold} --print_cluster_sizes 1 --target_clusters ${clusters} > results_diss/${device_name}_perf_quality_${file_suffix}.log 2>&1
done

# nvidia-smi dmon -i 0 -d 1sec -f results_diss/P100_quality_freq_${file_suffix}.log &
# RUNNING_PID=$!
# sleep 3
# ./datadriven/examplesOCL/clustering_cmd --config OCL_configs/config_ocl_float_${device_name}.cfg --binary_header_filename ${dataset_directory}/gaussian_c${clusters}_size${dataset_size}_dim10_noise --level=${level} --lambda=${lambda} --k 6 --epsilon 0.01 --threshold ${threshold} --print_cluster_sizes 1 --target_clusters ${clusters} > results_diss/${device_name}_perf_quality_run_${file_suffix}.log 2>&1
# kill ${RUNNING_PID}
