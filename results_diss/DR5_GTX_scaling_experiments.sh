#!/bin/bash

# support experiments
# unified algorithm
for i in $(seq 1 8); do
    cmd="./datadriven/examplesOCL/regression_cmd --trainingFileName ../../DissertationCodeGTX/datasets/DR5/DR5_nowarnings_less05_train.arff --testFileName ../../DissertationCodeGTX/datasets/DR5/DR5_nowarnings_less05_test.arff --learnerMode LEARNTEST --grid.level 10 --lambda 1E-5 --grid.type ModLinear --verbose true --additionalConfig results_diss/DR5_scaling_ocl_float_gtx1080ti_${i}.cfg --operation.type STREAMING --operation.subType OCLUNIFIED --solverFinal.maxIterations 500 --solverFinal.eps 0 --isRegression true --adaptConfig.support_refinement_min_support 500 --adaptConfig.support_refinement_ocl_config results_diss/DR5_scaling_ocl_float_gtx1080ti_${i}.cfg --adaptConfig.use_support_refinement --do_dummy_ocl_init"
    echo "${cmd}" > results_diss/DR5_scaling_support_unified_small_${i}.log
    ./${cmd} >> results_diss/DR5_scaling_support_unified_small_${i}.log
done

# modmask
for i in $(seq 1 8); do
    cmd="./datadriven/examplesOCL/regression_cmd --trainingFileName ../../DissertationCodeGTX/datasets/DR5/DR5_nowarnings_less05_train.arff --testFileName ../../DissertationCodeGTX/datasets/DR5/DR5_nowarnings_less05_test.arff --learnerMode LEARNTEST --grid.level 10 --lambda 1E-5 --grid.type ModLinear --verbose true --additionalConfig results_diss/DR5_scaling_ocl_float_gtx1080ti_${i}.cfg --operation.type STREAMING --operation.subType OCLMASKMP --solverFinal.maxIterations 500 --solverFinal.eps 0 --isRegression true --adaptConfig.support_refinement_min_support 500 --adaptConfig.support_refinement_ocl_config results_diss/DR5_scaling_ocl_float_gtx1080ti_${i}.cfg --adaptConfig.use_support_refinement --do_dummy_ocl_init"
    echo "${cmd}" > results_diss/DR5_scaling_support_modmask_small_${i}.log
    ./${cmd} >> results_diss/DR5_scaling_support_modmask_small_${i}.log
done

# surplus experiments
# unified
for i in $(seq 1 8); do
    cmd="./datadriven/examplesOCL/regression_cmd --trainingFileName ../../DissertationCodeGTX/datasets/DR5/DR5_nowarnings_less05_train.arff --testFileName ../../DissertationCodeGTX/datasets/DR5/DR5_nowarnings_less05_test.arff --learnerMode LEARNTEST --grid.level 6 --lambda 1E-5 --grid.type ModLinear --verbose true --additionalConfig results_diss/DR5_scaling_ocl_float_gtx1080ti_${i}.cfg --operation.type STREAMING --operation.subType OCLUNIFIED --solverFinal.maxIterations 500 --solverFinal.eps 0 --solverRefine.maxIterations 500 --solverRefine.eps 0 --adaptConfig.noPoints 200 --adaptConfig.numRefinements 7 --adaptConfig.threshold 0.0 --adaptConfig.percent 100.0 --isRegression true --do_dummy_ocl_init"
    echo "${cmd}" > results_diss/DR5_scaling_surplus_unified_small_${i}.log
    ./${cmd} >> results_diss/DR5_scaling_surplus_unified_small_${i}.log
done

# modmask
for i in $(seq 1 8); do
    cmd="./datadriven/examplesOCL/regression_cmd --trainingFileName ../../DissertationCodeGTX/datasets/DR5/DR5_nowarnings_less05_train.arff --testFileName ../../DissertationCodeGTX/datasets/DR5/DR5_nowarnings_less05_test.arff --learnerMode LEARNTEST --grid.level 6 --lambda 1E-5 --grid.type ModLinear --verbose true --additionalConfig results_diss/DR5_scaling_ocl_float_gtx1080ti_${i}.cfg --operation.type STREAMING --operation.subType OCLMASKMP --solverFinal.maxIterations 500 --solverFinal.eps 0 --solverRefine.maxIterations 500 --solverRefine.eps 0 --adaptConfig.noPoints 200 --adaptConfig.numRefinements 7 --adaptConfig.threshold 0.0 --adaptConfig.percent 100.0 --isRegression true --do_dummy_ocl_init"
    echo "${cmd}" > results_diss/DR5_scaling_surplus_modmask_small_${i}.log
    ./${cmd} >> results_diss/DR5_scaling_surplus_modmask_small_${i}.log
done
