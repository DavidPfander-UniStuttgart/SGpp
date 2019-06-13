#!/bin/bash

hn=`hostname`
if [ "${hn}" = "pcsgs07" ]; then
    device_name='A10-7850K'

elif [ ${hn} = "argon-tesla1" ]; then
    device_name='XeonSilver4116'

elif [ ${hn} = "argon-tesla2" ]; then
    device_name='XeonSilver4116'

elif [ ${hn} = "argon-epyc" ]; then
    device_name='Epyc7551P'

elif [ ${hn} = "argon-gtx" ]; then
    device_name='XeonGold5120'

elif [ ${hn} = "large" ]; then
    device_name='Xeon2670'
fi

# for tuned subspace results to find correct configuration
eval_repetitions=5

##################
# DR5 experiments
##################

###### mod linear ######

# # recursive
# ./datadriven/examplesOCL/regression_cmd --trainingFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --testFileName ../datasets/DR5/DR5_nowarnings_less05_test.arff --learnerMode LEARNTEST --grid.level 10 --lambda 1E-5 --grid.type ModLinear --verbose true --operation.type DEFAULT --operation.subType DEFAULT --solverFinal.maxIterations 300 --solverFinal.eps 0 --isRegression true --adaptConfig.support_refinement_min_support 500 --adaptConfig.use_support_refinement > results_diss/DR5_regression_comparison_recursive_modlinear_${device_name}.log 2>&1

# # streaming intrinsics -> OCL is not available on all processor platforms
# ./datadriven/examplesOCL/regression_cmd --trainingFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --testFileName ../datasets/DR5/DR5_nowarnings_less05_test.arff --learnerMode LEARNTEST --grid.level 10 --lambda 1E-5 --grid.type ModLinear --verbose true --operation.type STREAMING --operation.subType DEFAULT --solverFinal.maxIterations 300 --solverFinal.eps 0 --isRegression true --adaptConfig.support_refinement_min_support 500 --adaptConfig.use_support_refinement > results_diss/DR5_regression_comparison_streaming_modlinear_${device_name}.log 2>&1

# subspace
./datadriven/examplesOCL/regression_cmd --trainingFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --testFileName ../datasets/DR5/DR5_nowarnings_less05_test.arff --learnerMode LEARNTEST --grid.level 10 --lambda 1E-5 --grid.type ModLinear --verbose true --operation.type SUBSPACE --operation.subType AUTOTUNETMP --solverFinal.maxIterations 300 --solverFinal.eps 0 --isRegression true --adaptConfig.support_refinement_min_support 500 --adaptConfig.use_support_refinement --additionalConfig results_diss/subspace/DR5_${device_name}_Subspace_combined_tuner_line_search_t_${eval_repetitions}av_0r_optimal.cfg > results_diss/DR5_regression_comparison_subspace_modlinear_${device_name}.log 2>&1

###### linear ######

# # recursive
# ./datadriven/examplesOCL/regression_cmd --trainingFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --testFileName ../datasets/DR5/DR5_nowarnings_less05_test.arff --learnerMode LEARNTEST --grid.level 10 --lambda 1E-5 --grid.type Linear --verbose true --operation.type DEFAULT --operation.subType DEFAULT --solverFinal.maxIterations 300 --solverFinal.eps 0 --isRegression true --adaptConfig.support_refinement_min_support 500 --adaptConfig.use_support_refinement > results_diss/DR5_regression_comparison_recursive_linear_${device_name}.log 2>&1

# # streaming intrinsics -> OCL is not available on all processor platforms
# ./datadriven/examplesOCL/regression_cmd --trainingFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --testFileName ../datasets/DR5/DR5_nowarnings_less05_test.arff --learnerMode LEARNTEST --grid.level 10 --lambda 1E-5 --grid.type Linear --verbose true --operation.type STREAMING --operation.subType DEFAULT --solverFinal.maxIterations 300 --solverFinal.eps 0 --isRegression true --adaptConfig.support_refinement_min_support 500 --adaptConfig.use_support_refinement > results_diss/DR5_regression_comparison_streaming_linear_${device_name}.log 2>&1

# subspace
./datadriven/examplesOCL/regression_cmd --trainingFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --testFileName ../datasets/DR5/DR5_nowarnings_less05_test.arff --learnerMode LEARNTEST --grid.level 10 --lambda 1E-5 --grid.type Linear --verbose true --operation.type SUBSPACE --operation.subType AUTOTUNETMP --solverFinal.maxIterations 300 --solverFinal.eps 0 --isRegression true --adaptConfig.support_refinement_min_support 500 --adaptConfig.use_support_refinement --additionalConfig results_diss/subspace/DR5_${device_name}_Subspace_combined_tuner_line_search_t_${eval_repetitions}av_0r_optimal.cfg > results_diss/DR5_regression_comparison_subspace_linear_${device_name}.log 2>&1

########################
# Friedman1 experiments
########################

###### modlinear ######

# # recursive
# ./datadriven/examplesOCL/regression_cmd --trainingFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --learnerMode LEARN --grid.level 7 --lambda 1E-5 --grid.type ModLinear --verbose true --operation.type DEFAULT --operation.subType DEFAULT --solverFinal.maxIterations 100 --solverFinal.eps 0 --isRegression true > results_diss/friedman1_regression_comparison_recursive_modlinear_${device_name}.log 2>&1

# # streaming intrinsics -> OCL not available on all platforms
# ./datadriven/examplesOCL/regression_cmd --trainingFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --learnerMode LEARN --grid.level 7 --lambda 1E-5 --grid.type ModLinear --verbose true --operation.type STREAMING --operation.subType DEFAULT --solverFinal.maxIterations 100 --solverFinal.eps 0 --isRegression true > results_diss/friedman1_regression_comparison_streaming_modlinear_${device_name}.log 2>&1

# subspace
./datadriven/examplesOCL/regression_cmd --trainingFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --learnerMode LEARN --grid.level 7 --lambda 1E-5 --grid.type ModLinear --verbose true --operation.type SUBSPACE --operation.subType AUTOTUNETMP --solverFinal.maxIterations 100 --solverFinal.eps 0 --isRegression true > results_diss/friedman1_regression_comparison_subspace_modlinear_${device_name}.log --additionalConfig results_diss/subspace/friedman1_${device_name}_Subspace_combined_tuner_line_search_t_${eval_repetitions}av_0r_optimal.cfg 2>&1

###### linear ######

# # recursive
# ./datadriven/examplesOCL/regression_cmd --trainingFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --learnerMode LEARN --grid.level 7 --lambda 1E-5 --grid.type Linear --verbose true --operation.type DEFAULT --operation.subType DEFAULT --solverFinal.maxIterations 100 --solverFinal.eps 0 --isRegression true > results_diss/friedman1_regression_comparison_recursive_linear_${device_name}.log 2>&1

# # streaming intrinsics -> OCL not available on all platforms
# ./datadriven/examplesOCL/regression_cmd --trainingFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --learnerMode LEARN --grid.level 7 --lambda 1E-5 --grid.type Linear --verbose true --operation.type STREAMING --operation.subType DEFAULT --solverFinal.maxIterations 100 --solverFinal.eps 0 --isRegression true > results_diss/friedman1_regression_comparison_streaming_linear_${device_name}.log 2>&1

# subspace
./datadriven/examplesOCL/regression_cmd --trainingFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --learnerMode LEARN --grid.level 7 --lambda 1E-5 --grid.type Linear --verbose true --operation.type SUBSPACE --operation.subType AUTOTUNETMP --solverFinal.maxIterations 100 --solverFinal.eps 0 --isRegression true --additionalConfig results_diss/subspace/friedman1_${device_name}_Subspace_combined_tuner_line_search_t_${eval_repetitions}av_0r_optimal.cfg > results_diss/friedman1_regression_comparison_subspace_linear_${device_name}.log 2>&1
