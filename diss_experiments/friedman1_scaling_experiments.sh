#!/bin/bash

datasizes=(100000 200000 300000 400000 500000 600000 700000 800000)

# regular experiments
# unified algorithm
for datasetSize in "${datasizes[@]}"; do
    let "num_devices=$datasetSize / 100000"
    echo "work on datasetSize: ${datasetSize}, num_devices: ${num_devices}"
    cmd="./datadriven/examplesOCL/regression_cmd --trainingFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_${datasetSize}.arff --learnerMode LEARN --grid.level 7 --lambda 1E-5 --grid.type ModLinear --verbose true --additionalConfig results_diss/friedman1_ocl_float_gtx1080ti_${num_devices}.cfg --operation.type STREAMING --operation.subType OCLUNIFIED --solverFinal.maxIterations 100 --solverFinal.eps 0 --isRegression true --do_dummy_ocl_init"
    echo ${cmd}
    echo "${cmd}" > results_diss/friedman1_weakscaling_unified_${datasetSize}.log
    ./${cmd} >> results_diss/friedman1_weakscaling_unified_${datasetSize}.log
done

# modmask algorithm
for datasetSize in "${datasizes[@]}"; do
    let "num_devices=$datasetSize / 100000"
    echo "work on datasetSize: ${datasetSize}, num_devices: ${num_devices}"
    cmd="./datadriven/examplesOCL/regression_cmd --trainingFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_${datasetSize}.arff --learnerMode LEARN --grid.level 7 --lambda 1E-5 --grid.type ModLinear --verbose true --additionalConfig results_diss/friedman1_ocl_float_gtx1080ti_${num_devices}.cfg --operation.type STREAMING --operation.subType OCLMASKMP --solverFinal.maxIterations 100 --solverFinal.eps 0 --isRegression true --do_dummy_ocl_init"
    echo ${cmd}
    echo "${cmd}" > results_diss/friedman1_weakscaling_modmask_${datasetSize}.log
    ./${cmd} >> results_diss/friedman1_weakscaling_modmask_${datasetSize}.log
done
