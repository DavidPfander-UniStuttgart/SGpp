#!/bin/bash
set -x

hn=`hostname`
if [ "${hn}" = "pcsgs07" ]; then
    device_name_f1='w8100'
    AUTOTUNETMP_SELECT="AMD Accelerated Parallel Processing/Hawaii"
elif [ ${hn} = "argon-tesla1" ]; then
    device_name_f1='P100'
    AUTOTUNETMP_SELECT="NVIDIA CUDA/Tesla P100-PCIE-16GB"
elif [ ${hn} = "argon-tesla2" ]; then
    device_name_f1='QuadroGP100'
    AUTOTUNETMP_SELECT="NVIDIA CUDA/Quadro GP100"
elif [ ${hn} = "argon-epyc" ]; then
    device_name_f1='Vega7'
    AUTOTUNETMP_SELECT="AMD Accelerated Parallel Processing/gfx906"
elif [ ${hn} = "argon-gtx" ]; then
    device_name_f1='gtx1080ti'
    AUTOTUNETMP_SELECT="NVIDIA CUDA/GeForce GTX 1080 Ti"
    device_name_f1_cpu='Gold5120'
fi
echo "hostname: ${hn}"
echo "device_name_f1: ${device_name_f1}"
echo "AUTOTUNETMP_SELECT: ${AUTOTUNETMP_SELECT}"

############################################################################################# DR5 strong scaling #########################################################################

# if [ ${hn} = "argon-gtx" ]; then
#     device_name_DR5='gtx1080ti_8'
#     # tune for 8 devices at the same time, a weak scaling necessity
#     ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/DR5_strongscaling_ocl_float_${device_name_DR5}_template.cfg --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --scenarioName DR5_${device_name_DR5}_strongscaling --level 10 --tuner_name line_search --repetitions 10 --isModLinear true --use_support_refinement --support_refinement_min_support 500 --file_prefix results_diss/ --trans false

#     ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/DR5_strongscaling_ocl_float_${device_name_DR5}_template.cfg --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --scenarioName DR5_${device_name_DR5}_strongscaling --level 10 --tuner_name line_search --repetitions 10 --isModLinear true --use_support_refinement --support_refinement_min_support 500 --file_prefix results_diss/ --trans true
# fi

############################################################################################# friedman1 node-level #########################################################################

./datadriven/examplesOCL/detectPlatform --precision float --file_name results_diss/friedman1_${hn}_${device_name_f1}_ocl_config_single.cfg --select="${AUTOTUNETMP_SELECT}" --remove_unselected

# tune for 1 device as there is enough work guaranteed
./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/friedman1_${hn}_${device_name_f1}_ocl_config_single.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --scenarioName friedman1_${device_name_f1}_weakscaling_parameters --level 7 --tuner_name line_search --repetitions 10 --isModLinear true --file_prefix results_diss/ --trans false

./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/friedman1_${hn}_${device_name_f1}_ocl_config_single.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --scenarioName friedman1_${device_name_f1}_weakscaling_parameters --level 7 --tuner_name line_search --repetitions 10 --isModLinear true --file_prefix results_diss/ --trans true

# this excludes gtx1080ti and the Gold5120 (the Intel OpenCL platform does not support 64bit atomics)
if [ ${hn} != "argon-gtx" ]; then
    ./datadriven/examplesOCL/detectPlatform --precision double --file_name results_diss/friedman1_${hn}_${device_name_f1}_ocl_config_double.cfg --select="${AUTOTUNETMP_SELECT}" --remove_unselected

    # tune for 1 device as there is enough work guaranteed
    ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/friedman1_${hn}_${device_name_f1}_ocl_config_double.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --scenarioName friedman1_${device_name_f1}_weakscaling_parameters --level 7 --tuner_name line_search --repetitions 10 --isModLinear true --file_prefix results_diss/ --trans false

    ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/friedman1_${hn}_${device_name_f1}_ocl_config_double.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --scenarioName friedman1_${device_name_f1}_weakscaling_parameters --level 7 --tuner_name line_search --repetitions 10 --isModLinear true --file_prefix results_diss/ --trans true
fi

if [ ${hn} = "argon-gtx" ]; then
    # use argon-gtx as cpu platform
    AUTOTUNETMP_SELECT="Intel(R) OpenCL/Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz"

    ./datadriven/examplesOCL/detectPlatform --precision float --file_name results_diss/friedman1_${hn}_${device_name_f1_cpu}_ocl_config_single.cfg --select="${AUTOTUNETMP_SELECT}" --remove_unselected

    ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/friedman1_${hn}_${device_name_f1_cpu}_ocl_config_single.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --scenarioName friedman1_${device_name_f1_cpu}_weakscaling_parameters --level 7 --tuner_name line_search --repetitions 10 --isModLinear true --file_prefix results_diss/ --trans false
    ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/friedman1_${hn}_${device_name_f1_cpu}_ocl_config_single.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --scenarioName friedman1_${device_name_f1_cpu}_weakscaling_parameters --level 7 --tuner_name line_search --repetitions 10 --isModLinear true --file_prefix results_diss/ --trans true
fi

# friedman1 - compute performance of final parameters ---------------------- GFLOPS
./datadriven/examplesOCL/regressionGFlops --OpenCLConfigFile results_diss/friedman1_${device_name_f1}_weakscaling_parameters_mult_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --isModLinear true --trans false --level 7 --repetitions 10 > results_diss/friedman1_performance_mult_float_${device_name_f1}.log 2>&1

./datadriven/examplesOCL/regressionGFlops --OpenCLConfigFile results_diss/friedman1_${device_name_f1}_weakscaling_parameters_multTrans_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --isModLinear true --trans true --level 7 --repetitions 10 > results_diss/friedman1_performance_multTranspose_float_${device_name_f1}.log 2>&1

if [ ${hn} != "argon-gtx" ]; then
    ./datadriven/examplesOCL/regressionGFlops --OpenCLConfigFile results_diss/friedman1_${device_name_f1}_weakscaling_parameters_mult_host_${hn}_tuner_line_search_t_double_10r_optimal.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --isModLinear true --trans false --level 7 --repetitions 10 > results_diss/friedman1_performance_mult_double_${device_name_f1}.log 2>&1

    ./datadriven/examplesOCL/regressionGFlops --OpenCLConfigFile results_diss/friedman1_${device_name_f1}_weakscaling_parameters_multTrans_host_${hn}_tuner_line_search_t_double_10r_optimal.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --isModLinear true --trans true --level 7 --repetitions 10 > results_diss/friedman1_performance_multTranspose_double_${device_name_f1}.log 2>&1
fi

if [ ${hn} = "argon-gtx" ]; then
    # compute performance of final parameters
    ./datadriven/examplesOCL/regressionGFlops --OpenCLConfigFile results_diss/friedman1_${device_name_f1_cpu}_weakscaling_parameters_mult_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --isModLinear true --trans false --level 7 --repetitions 10 > results_diss/friedman1_performance_mult_float_${device_name_f1_cpu}.log 2>&1

    ./datadriven/examplesOCL/regressionGFlops --OpenCLConfigFile results_diss/friedman1_${device_name_f1_cpu}_weakscaling_parameters_multTrans_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --isModLinear true --trans true --level 7 --repetitions 10 > results_diss/friedman1_performance_multTranspose_float_${device_name_f1_cpu}.log 2>&1
fi

# friedman1 - compute performance of final parameters ---------------------- duration

./datadriven/examplesOCL/combined_configs --base_file results_diss/friedman1_${device_name_f1}_weakscaling_parameters_mult_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg --combine_file results_diss/friedman1_${device_name_f1}_weakscaling_parameters_multTrans_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg --combine_prefixes TRANS KERNEL_TRANS --kernel_name StreamingModOCLUnified  --result_file results_diss/friedman1_${device_name_f1}_weakscaling_parameters_combined_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg

./datadriven/examplesOCL/regression_cmd --additionalConfig results_diss/friedman1_${device_name_f1}_weakscaling_parameters_combined_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg --trainingFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --grid.type ModLinear --grid.level 7 --operation.type STREAMING --operation.subType OCLUNIFIED --solverFinal.maxIterations 100 --solverFinal.eps 0 --isRegression true  --do_dummy_ocl_init --lambda 1E-5 > results_diss/friedman1_performance_dur_float_${device_name_f1}.log 2>&1

if [ ${hn} != "argon-gtx" ]; then
    ./datadriven/examplesOCL/combined_configs --base_file results_diss/friedman1_${device_name_f1}_weakscaling_parameters_mult_host_${hn}_tuner_line_search_t_double_10r_optimal.cfg --combine_file results_diss/friedman1_${device_name_f1}_weakscaling_parameters_multTrans_host_${hn}_tuner_line_search_t_double_10r_optimal.cfg --combine_prefixes TRANS KERNEL_TRANS --kernel_name StreamingModOCLUnified  --result_file results_diss/friedman1_${device_name_f1}_weakscaling_parameters_combined_host_${hn}_tuner_line_search_t_double_10r_optimal.cfg
    
    ./datadriven/examplesOCL/regression_cmd --additionalConfig results_diss/friedman1_${device_name_f1}_weakscaling_parameters_combined_host_${hn}_tuner_line_search_t_double_10r_optimal.cfg --trainingFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --grid.type ModLinear false --grid.level 7 --operation.type STREAMING --operation.subType OCLUNIFIED  --solverFinal.maxIterations 100 --solverFinal.eps 0 --isRegression true --do_dummy_ocl_init --lambda 1E-5 > results_diss/friedman1_performance_dur_double_${device_name_f1}.log 2>&1
fi

if [ ${hn} = "argon-gtx" ]; then
    ./datadriven/examplesOCL/combined_configs --base_file results_diss/friedman1_${device_name_f1_cpu}_weakscaling_parameters_mult_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg --combine_file results_diss/friedman1_${device_name_f1_cpu}_weakscaling_parameters_multTrans_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg --combine_prefixes TRANS KERNEL_TRANS --kernel_name StreamingModOCLUnified  --result_file results_diss/friedman1_${device_name_f1_cpu}_weakscaling_parameters_combined_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg

    ./datadriven/examplesOCL/regression_cmd --additionalConfig results_diss/friedman1_${device_name_f1_cpu}_weakscaling_parameters_combined_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg --trainingFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --grid.type ModLinear --grid.level 7 --operation.type STREAMING --operation.subType OCLUNIFIED --solverFinal.maxIterations 100 --solverFinal.eps 0 --isRegression true --lambda 1E-5 > results_diss/friedman1_performance_dur_float_${device_name_f1_cpu}.log 2>&1
fi

############################################################################################# DR5 node-level #########################################################################
# DR5 - tune for single device

./datadriven/examplesOCL/detectPlatform --precision float --file_name results_diss/DR5_${hn}_${device_name_f1}_ocl_config_single.cfg --select="${AUTOTUNETMP_SELECT}" --remove_unselected

# tune for 1 device as there is enough work guaranteed
./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/DR5_${hn}_${device_name_f1}_ocl_config_single.cfg --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --scenarioName DR5_${device_name_f1}_weakscaling_parameters --level 10 --use_support_refinement --support_refinement_min_support 500 --tuner_name line_search --repetitions 10 --isModLinear true --file_prefix results_diss/ --trans false

./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/DR5_${hn}_${device_name_f1}_ocl_config_single.cfg --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --scenarioName DR5_${device_name_f1}_weakscaling_parameters --level 10 --use_support_refinement --support_refinement_min_support 500 --tuner_name line_search --repetitions 10 --isModLinear true --file_prefix results_diss/ --trans true

# this excludes gtx1080ti and the Gold5120 (the Intel OpenCL platform does not support 64bit atomics)
if [ ${hn} != "argon-gtx" ]; then
    ./datadriven/examplesOCL/detectPlatform --precision double --file_name results_diss/DR5_${hn}_${device_name_f1}_ocl_config_double.cfg --select="${AUTOTUNETMP_SELECT}" --remove_unselected

    # tune for 1 device as there is enough work guaranteed
    ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/DR5_${hn}_${device_name_f1}_ocl_config_double.cfg --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --scenarioName DR5_${device_name_f1}_weakscaling_parameters --level 10 --use_support_refinement --support_refinement_min_support 500 --tuner_name line_search --repetitions 10 --isModLinear true --file_prefix results_diss/ --trans false

    ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/DR5_${hn}_${device_name_f1}_ocl_config_double.cfg --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --scenarioName DR5_${device_name_f1}_weakscaling_parameters --level 10 --use_support_refinement --support_refinement_min_support 500 --tuner_name line_search --repetitions 10 --isModLinear true --file_prefix results_diss/ --trans true
fi

if [ ${hn} = "argon-gtx" ]; then
    # use argon-gtx as cpu platform
    AUTOTUNETMP_SELECT="Intel(R) OpenCL/Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz"

    ./datadriven/examplesOCL/detectPlatform --precision float --file_name results_diss/DR5_${hn}_${device_name_f1_cpu}_ocl_config_single.cfg --select="${AUTOTUNETMP_SELECT}" --remove_unselected

    ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/DR5_${hn}_${device_name_f1_cpu}_ocl_config_single.cfg --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --scenarioName DR5_${device_name_f1_cpu}_weakscaling_parameters --level 10 --use_support_refinement --support_refinement_min_support 500 --tuner_name line_search --repetitions 10 --isModLinear true --file_prefix results_diss/ --trans false
    ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/DR5_${hn}_${device_name_f1_cpu}_ocl_config_single.cfg --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --scenarioName DR5_${device_name_f1_cpu}_weakscaling_parameters --level 10 --use_support_refinement --support_refinement_min_support 500 --tuner_name line_search --repetitions 10 --isModLinear true --file_prefix results_diss/ --trans true
fi

# DR5 - compute performance of final parameters ---------------------- GFLOPS
./datadriven/examplesOCL/regressionGFlops --OpenCLConfigFile results_diss/DR5_${device_name_f1}_weakscaling_parameters_mult_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --isModLinear true --trans false --level 10--use_support_refinement --support_refinement_min_support 500  --repetitions 10 > results_diss/DR5_performance_mult_float_${device_name_f1}.log 2>&1

./datadriven/examplesOCL/regressionGFlops --OpenCLConfigFile results_diss/DR5_${device_name_f1}_weakscaling_parameters_multTrans_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --isModLinear true --trans true --level 10--use_support_refinement --support_refinement_min_support 500  --repetitions 10 > results_diss/DR5_performance_multTranspose_float_${device_name_f1}.log 2>&1

if [ ${hn} != "argon-gtx" ]; then
    ./datadriven/examplesOCL/regressionGFlops --OpenCLConfigFile results_diss/DR5_${device_name_f1}_weakscaling_parameters_mult_host_${hn}_tuner_line_search_t_double_10r_optimal.cfg --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --isModLinear true --trans false --level 10--use_support_refinement --support_refinement_min_support 500  --repetitions 10 > results_diss/DR5_performance_mult_double_${device_name_f1}.log 2>&1

    ./datadriven/examplesOCL/regressionGFlops --OpenCLConfigFile results_diss/DR5_${device_name_f1}_weakscaling_parameters_multTrans_host_${hn}_tuner_line_search_t_double_10r_optimal.cfg --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --isModLinear true --trans true --level 10--use_support_refinement --support_refinement_min_support 500  --repetitions 10 > results_diss/DR5_performance_multTranspose_double_${device_name_f1}.log 2>&1
fi

if [ ${hn} = "argon-gtx" ]; then
    # compute performance of final parameters
    ./datadriven/examplesOCL/regressionGFlops --OpenCLConfigFile results_diss/DR5_${device_name_f1_cpu}_weakscaling_parameters_mult_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --isModLinear true --trans false --level 10--use_support_refinement --support_refinement_min_support 500  --repetitions 10 > results_diss/DR5_performance_mult_float_${device_name_f1_cpu}.log 2>&1

    ./datadriven/examplesOCL/regressionGFlops --OpenCLConfigFile results_diss/DR5_${device_name_f1_cpu}_weakscaling_parameters_multTrans_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --isModLinear true --trans true --level 10--use_support_refinement --support_refinement_min_support 500  --repetitions 10 > results_diss/DR5_performance_multTranspose_float_${device_name_f1_cpu}.log 2>&1
fi

# DR5 - compute performance of final parameters ---------------------- duration

./datadriven/examplesOCL/combined_configs --base_file results_diss/DR5_${device_name_f1}_weakscaling_parameters_mult_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg --combine_file results_diss/DR5_${device_name_f1}_weakscaling_parameters_multTrans_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg --combine_prefixes TRANS KERNEL_TRANS --kernel_name StreamingModOCLUnified  --result_file results_diss/DR5_${device_name_f1}_weakscaling_parameters_combined_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg

./datadriven/examplesOCL/regression_cmd --additionalConfig results_diss/DR5_${device_name_f1}_weakscaling_parameters_combined_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg --trainingFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --grid.type ModLinear --grid.level 10 --operation.type STREAMING --operation.subType OCLUNIFIED  --solverFinal.maxIterations 500 --solverFinal.eps 0 --isRegression true  --adaptConfig.support_refinement_min_support 500 --adaptConfig.support_refinement_ocl_config results_diss/DR5_${device_name_f1_cpu}_weakscaling_parameters_combined_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg --adaptConfig.use_support_refinement  --do_dummy_ocl_init --lambda 1E-5 > results_diss/DR5_performance_dur_float_${device_name_f1}.log 2>&1

if [ ${hn} != "argon-gtx" ]; then
    ./datadriven/examplesOCL/combined_configs --base_file results_diss/DR5_${device_name_f1}_weakscaling_parameters_mult_host_${hn}_tuner_line_search_t_double_10r_optimal.cfg --combine_file results_diss/DR5_${device_name_f1}_weakscaling_parameters_multTrans_host_${hn}_tuner_line_search_t_double_10r_optimal.cfg --combine_prefixes TRANS KERNEL_TRANS --kernel_name StreamingModOCLUnified  --result_file results_diss/DR5_${device_name_f1}_weakscaling_parameters_combined_host_${hn}_tuner_line_search_t_double_10r_optimal.cfg
    
    ./datadriven/examplesOCL/regression_cmd --additionalConfig results_diss/DR5_${device_name_f1}_weakscaling_parameters_combined_host_${hn}_tuner_line_search_t_double_10r_optimal.cfg --trainingFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --grid.type ModLinear --grid.level 10 --operation.type STREAMING --operation.subType OCLUNIFIED  --solverFinal.maxIterations 300 --solverFinal.eps 0 --isRegression true --adaptConfig.support_refinement_min_support 500 --adaptConfig.support_refinement_ocl_config results_diss/DR5_${device_name_f1_cpu}_weakscaling_parameters_combined_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg --adaptConfig.use_support_refinement --do_dummy_ocl_init --lambda 1E-5 > results_diss/DR5_performance_dur_double_${device_name_f1}.log 2>&1
fi

if [ ${hn} = "argon-gtx" ]; then
    ./datadriven/examplesOCL/combined_configs --base_file results_diss/DR5_${device_name_f1_cpu}_weakscaling_parameters_mult_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg --combine_file results_diss/DR5_${device_name_f1_cpu}_weakscaling_parameters_multTrans_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg --combine_prefixes TRANS KERNEL_TRANS --kernel_name StreamingModOCLUnified  --result_file results_diss/DR5_${device_name_f1_cpu}_weakscaling_parameters_combined_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg
    
    ./datadriven/examplesOCL/regression_cmd --additionalConfig results_diss/DR5_${device_name_f1_cpu}_weakscaling_parameters_combined_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg --trainingFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --grid.type ModLinear --grid.level 10 --operation.type STREAMING --operation.subType OCLUNIFIED  --solverFinal.maxIterations 500 --solverFinal.eps 0 --isRegression true --adaptConfig.support_refinement_min_support 500 --adaptConfig.support_refinement_ocl_config results_diss/DR5_${device_name_f1_cpu}_weakscaling_parameters_combined_host_${hn}_tuner_line_search_t_float_10r_optimal.cfg --adaptConfig.use_support_refinement --lambda 1E-5 > results_diss/DR5_performance_dur_float_${device_name_f1_cpu}.log 2>&1
fi
