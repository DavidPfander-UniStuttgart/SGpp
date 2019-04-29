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

tuner_repetitions=3
eval_repetitions=5

echo "hostname: ${hn}"
echo "device_name_f1: ${device_name}"

source source_autotunetmp_generic.sh

###### tune all cases with multiple tuners for the autotunetmp evaluation ######

# DR5
./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --file_prefix results_diss/subspace/ --scenarioName DR5_${device_name} --level 10 --tuner_name line_search --repetitions ${tuner_repetitions} --use_support_refinement --support_refinement_min_support 500 --trans true --repetitions_averaged ${eval_repetitions}
./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --file_prefix results_diss/subspace/ --scenarioName DR5_${device_name} --level 10 --tuner_name neighborhood_search --repetitions ${tuner_repetitions} --use_support_refinement --support_refinement_min_support 500 --trans true --repetitions_averaged ${eval_repetitions}
./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --file_prefix results_diss/subspace/ --scenarioName DR5_${device_name} --level 10 --tuner_name monte_carlo --repetitions ${tuner_repetitions} --use_support_refinement --support_refinement_min_support 500 --trans true --repetitions_averaged ${eval_repetitions}

./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --file_prefix results_diss/subspace/ --scenarioName DR5_${device_name} --level 10 --tuner_name line_search --repetitions ${tuner_repetitions} --use_support_refinement --support_refinement_min_support 500 --trans false --repetitions_averaged ${eval_repetitions}
./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --file_prefix results_diss/subspace/ --scenarioName DR5_${device_name} --level 10 --tuner_name neighborhood_search --repetitions ${tuner_repetitions} --use_support_refinement --support_refinement_min_support 500 --trans false --repetitions_averaged ${eval_repetitions}
./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --file_prefix results_diss/subspace/ --scenarioName DR5_${device_name} --level 10 --tuner_name monte_carlo --repetitions ${tuner_repetitions} --use_support_refinement --support_refinement_min_support 500 --trans false --repetitions_averaged ${eval_repetitions}

# friedman1
./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --file_prefix results_diss/subspace/ --scenarioName friedman1_${device_name} --level 7 --tuner_name line_search --repetitions ${tuner_repetitions} --trans true --repetitions_averaged ${eval_repetitions}
./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --file_prefix results_diss/subspace/ --scenarioName friedman1_${device_name} --level 7 --tuner_name neighborhood_search --repetitions ${tuner_repetitions} --trans true --repetitions_averaged ${eval_repetitions}
./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --file_prefix results_diss/subspace/ --scenarioName friedman1_${device_name} --level 7 --tuner_name monte_carlo --repetitions ${tuner_repetitions} --trans true --repetitions_averaged ${eval_repetitions}

./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --file_prefix results_diss/subspace/ --scenarioName friedman1_${device_name} --level 7 --tuner_name line_search --repetitions ${tuner_repetitions} --trans false --repetitions_averaged ${eval_repetitions}
./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --file_prefix results_diss/subspace/ --scenarioName friedman1_${device_name} --level 7 --tuner_name neighborhood_search --repetitions ${tuner_repetitions} --trans false --repetitions_averaged ${eval_repetitions}
./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --file_prefix results_diss/subspace/ --scenarioName friedman1_${device_name} --level 7 --tuner_name monte_carlo --repetitions ${tuner_repetitions} --trans false --repetitions_averaged ${eval_repetitions}

############## tuned single iteration ###############

# friedman1 single it duration
cmd="./datadriven/examplesOCL/regressionGFlops --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --isModLinear true --trans true --level 7  --repetitions 10 --operation.type SUBSPACE --operation.subType AUTOTUNETMP --OpenCLConfigFile results_diss/subspace/friedman1_${device_name}_Subspace_multTrans_tuner_line_search_t_${eval_repetitions}av_0r_optimal.cfg"
echo ${cmd}
echo "${cmd}" > results_diss/subspace/friedman1_subspace_singleit_mult_double_${device_name}.log 2>&1
./${cmd} >> results_diss/subspace/friedman1_subspace_singleit_mult_double_${device_name}.log 2>&1

cmd="./datadriven/examplesOCL/regressionGFlops --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --isModLinear true --trans false --level 7  --repetitions 10 --operation.type SUBSPACE --operation.subType AUTOTUNETMP --OpenCLConfigFile results_diss/subspace/friedman1_${device_name}_Subspace_mult_tuner_line_search_t_${eval_repetitions}av_0r_optimal.cfg"
echo ${cmd}
echo "${cmd}" > results_diss/subspace/friedman1_subspace_singleit_multTranspose_double_${device_name}.log 2>&1
./${cmd} >> results_diss/subspace/friedman1_subspace_singleit_multTranspose_double_${device_name}.log 2>&1

# DR5 single it duration
cmd="./datadriven/examplesOCL/regressionGFlops --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --isModLinear true --trans true --level 10 --use_support_refinement --support_refinement_min_support 500  --repetitions 10 --operation.type SUBSPACE --operation.subType AUTOTUNETMP --OpenCLConfigFile results_diss/subspace/DR5_${device_name}_Subspace_multTrans_tuner_line_search_t_${eval_repetitions}av_0r_optimal.cfg"
echo ${cmd}
echo "${cmd}" > results_diss/subspace/DR5_subspace_singleit_mult_double_${device_name}.log 2>&1
./${cmd} >> results_diss/subspace/DR5_subspace_singleit_mult_double_${device_name}.log 2>&1

cmd="./datadriven/examplesOCL/regressionGFlops --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --isModLinear true --trans false --level 10 --use_support_refinement --support_refinement_min_support 500 --repetitions 10 --operation.type SUBSPACE --operation.subType AUTOTUNETMP --OpenCLConfigFile results_diss/subspace/DR5_${device_name}_Subspace_mult_tuner_line_search_t_${eval_repetitions}av_0r_optimal.cfg"
echo ${cmd}
echo "${cmd}" > results_diss/subspace/DR5_subspace_singleit_multTranspose_double_${device_name}.log 2>&1
./${cmd} >> results_diss/subspace/DR5_subspace_singleit_multTranspose_double_${device_name}.log 2>&1

############### tuned whole scenario ###############

./datadriven/examplesOCL/combined_configs_copy --base_file results_diss/subspace/friedman1_${device_name}_Subspace_multTrans_tuner_line_search_t_${eval_repetitions}av_0r_optimal.cfg --combine_file results_diss/subspace/friedman1_${device_name}_Subspace_mult_tuner_line_search_t_${eval_repetitions}av_0r_optimal.cfg --result_file results_diss/subspace/friedman1_${device_name}_Subspace_combined_tuner_line_search_t_${eval_repetitions}av_0r_optimal.cfg

# friedman1
echo "friedman1, host: ${hn}, device: ${device_name}"
cmd="./datadriven/examplesOCL/regression_cmd --trainingFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --learnerMode LEARN --grid.level 7 --lambda 1E-5 --grid.type ModLinear --verbose true --operation.type SUBSPACE --operation.subType AUTOTUNETMP --solverFinal.maxIterations 100 --solverFinal.eps 0 --isRegression true --additionalConfig results_diss/subspace/friedman1_${device_name}_Subspace_combined_tuner_line_search_t_${eval_repetitions}av_0r_optimal.cfg"
echo ${cmd}
echo "${cmd}" > results_diss/subspace/friedman1_subspace_wholescenario_${hn}_${device_name}.log
./${cmd} >> results_diss/subspace/friedman1_subspace_wholescenario_${hn}_${device_name}.log 2>&1

./datadriven/examplesOCL/combined_configs_copy --base_file results_diss/subspace/DR5_${device_name}_Subspace_multTrans_tuner_line_search_t_${eval_repetitions}av_0r_optimal.cfg --combine_file results_diss/subspace/DR5_${device_name}_Subspace_mult_tuner_line_search_t_${eval_repetitions}av_0r_optimal.cfg --result_file results_diss/subspace/DR5_${device_name}_Subspace_combined_tuner_line_search_t_${eval_repetitions}av_0r_optimal.cfg

# DR5
echo "DR5, host: ${hn}, device: ${device_name}"
cmd="./datadriven/examplesOCL/regression_cmd --trainingFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --testFileName ../../DissertationCodeGTX/datasets/DR5/DR5_nowarnings_less05_test.arff --learnerMode LEARNTEST --grid.level 10 --lambda 1E-5 --grid.type ModLinear --verbose true --operation.type SUBSPACE --operation.subType AUTOTUNETMP --solverFinal.maxIterations 300 --solverFinal.eps 0 --isRegression true --adaptConfig.support_refinement_min_support 500 --adaptConfig.use_support_refinement --additionalConfig results_diss/subspace/DR5_${device_name}_Subspace_combined_tuner_line_search_t_${eval_repetitions}av_0r_optimal.cfg"
echo "${cmd}" > results_diss/subspace/DR5_subspace_wholescenario_${hn}_${device_name}.log
./${cmd} >> results_diss/subspace/DR5_subspace_wholescenario_${hn}_${device_name}.log 2>&1
