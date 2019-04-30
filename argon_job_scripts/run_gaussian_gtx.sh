#!/bin/bash

dataset_size=1000000
dim=10
level=7
lambda=1E-6
noise="_noise"
k=8
threshold=950.0
epsilon=1E-3

config=config_ocl_float_gtx1080ti.cfg
mpiconfig=argon_job_scripts/GTXConf8.cfg
num_gpus=8

dataset_base_path=../../DissertationCodeTesla1/SGpp/
let num_tasks="${num_gpus} + 1"
echo "mpirun.openmpi -n ${num_tasks} ./datadriven/examplesMPI/distributed_clustering_cmd --config ${config} --MPIconfig ${mpiconfig} --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size${dataset_size}_dim${dim}${noise} --level=${level} --lambda=${lambda} --k ${k} --epsilon ${epsilon} --threshold ${threshold}"
