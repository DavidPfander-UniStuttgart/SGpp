#!/bin/bash
set -x

source source_autotunetmp_generic.sh

for set_size in 1000000 10000000 100000000; do
    for dim in 5 10; do    
        if [ ${set_size} -eq 1000000 ]; then
            level=7
        elif [ ${set_size} -eq 10000000 ]; then
            level=8
        elif [ ${set_size} -eq 100000000 ]; then
            level=8 # not clear whether level is appropriate
        fi
        for noise in "" "_noise"; do
            dataset_file=gaussian_c100_size${set_size}_dim${dim}${noise}
            mpirun.openmpi -n 3 ./datadriven/examplesMPI/distributed_clustering_cmd --config config_ocl_float_P100.cfg --MPIconfig argon_job_scripts/Tesla1.cfg --threshold=-999999.0 --binary_header_filename datasets_WPDM18/${dataset_file} --level=${level} --lambda=0.00001 --k 6 --epsilon 1E-3 --density_coefficients_file results_WPDM18/${dataset_file}_density_coef.serialized --pruned_knn_file results_WPDM18/${dataset_file}_graph.csv
        done
    done
done
