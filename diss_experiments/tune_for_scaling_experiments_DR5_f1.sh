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

if [ ${hn} = "argon-gtx" ]; then
    device_name_DR5='gtx1080ti_8'
    # tune for 8 devices at the same time, a weak scaling necessity
    ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/DR5_strongscaling_ocl_float_${device_name_DR5}_template.cfg --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --scenarioName DR5_${device_name_DR5}_strongscaling --level 10 --tuner_name line_search --repetitions 10 --isModLinear true --use_support_refinement --support_refinement_min_support 500 --file_prefix results_diss/ --trans false

    ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/DR5_strongscaling_ocl_float_${device_name_DR5}_template.cfg --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --scenarioName DR5_${device_name_DR5}_strongscaling --level 10 --tuner_name line_search --repetitions 10 --isModLinear true --use_support_refinement --support_refinement_min_support 500 --file_prefix results_diss/ --trans true
fi

# ./datadriven/examplesOCL/detectPlatform --precision float --file_name results_diss/friedman1_${hn}_${device_name_f1}_ocl_config_single.cfg --select="${AUTOTUNETMP_SELECT}" --remove_unselected

# # tune for 1 device as there is enough work guaranteed
# ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/friedman1_${hn}_${device_name_f1}_ocl_config_single.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --scenarioName friedman1_${device_name_f1}_weakscaling_parameters --level 7 --tuner_name line_search --repetitions 10 --isModLinear true --file_prefix results_diss/ --trans false

# ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/friedman1_${hn}_${device_name_f1}_ocl_config_single.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --scenarioName friedman1_${device_name_f1}_weakscaling_parameters --level 7 --tuner_name line_search --repetitions 10 --isModLinear true --file_prefix results_diss/ --trans true


# # compute performance of final parameters
# ./datadriven/examplesOCL/regressionGFlops --OpenCLConfigFile results_diss/friedman1_${device_name_f1}_weakscaling_parameters_mult_host_argon-gtx_tuner_line_search_t_float_10r_optimal.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --isModLinear true --trans false --level 7 --repetitions 10 > results_diss/friedman1_performance_mult_float_${device_name_f1}.log 2>&1

# ./datadriven/examplesOCL/regressionGFlops --OpenCLConfigFile results_diss/friedman1_${device_name_f1}_weakscaling_parameters_multTrans_host_argon-gtx_tuner_line_search_t_float_10r_optimal.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --isModLinear true --trans true --level 7 --repetitions 10 > results_diss/friedman1_performance_multTranspose_float_${device_name_f1}.log 2>&1

# # this excludes gtx1080ti and the Gold5120 (the Intel OpenCL platform does not support 64bit atomics)
# if [ ${hn} != "argon-gtx" ]; then
#     ./datadriven/examplesOCL/detectPlatform --precision double --file_name results_diss/friedman1_${hn}_${device_name_f1}_ocl_config_double.cfg --select="${AUTOTUNETMP_SELECT}" --remove_unselected

#     # tune for 1 device as there is enough work guaranteed
#     ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/friedman1_${hn}_${device_name_f1}_ocl_config_double.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --scenarioName friedman1_${device_name_f1}_weakscaling_parameters --level 7 --tuner_name line_search --repetitions 10 --isModLinear true --file_prefix results_diss/ --trans false

#     ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/friedman1_${hn}_${device_name_f1}_ocl_config_double.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --scenarioName friedman1_${device_name_f1}_weakscaling_parameters --level 7 --tuner_name line_search --repetitions 10 --isModLinear true --file_prefix results_diss/ --trans true

#     # compute performance of final parameters
#     ./datadriven/examplesOCL/regressionGFlops --OpenCLConfigFile results_diss/friedman1_${device_name_f1}_weakscaling_parameters_mult_host_argon-gtx_tuner_line_search_t_double_10r_optimal.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --isModLinear true --trans false --level 7 --repetitions 10 > results_diss/friedman1_performance_mult_double_${device_name_f1}.log 2>&1

#     ./datadriven/examplesOCL/regressionGFlops --OpenCLConfigFile results_diss/friedman1_${device_name_f1}_weakscaling_parameters_multTrans_host_argon-gtx_tuner_line_search_t_double_10r_optimal.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --isModLinear true --trans true --level 7 --repetitions 10 > results_diss/friedman1_performance_multTranspose_double_${device_name_f1}.log 2>&1
# fi

# if [ ${hn} = "argon-gtx" ]; then
#     # use argon-gtx as cpu platform
#     AUTOTUNETMP_SELECT="Intel(R) OpenCL/Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz"

#     ./datadriven/examplesOCL/detectPlatform --precision float --file_name results_diss/friedman1_${hn}_${device_name_f1_cpu}_ocl_config_single.cfg --select="${AUTOTUNETMP_SELECT}" --remove_unselected

#     ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/friedman1_${hn}_${device_name_f1_cpu}_ocl_config_single.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --scenarioName friedman1_${device_name_f1_cpu}_weakscaling_parameters --level 7 --tuner_name line_search --repetitions 10 --isModLinear true --file_prefix results_diss/ --trans false
#     ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/friedman1_${hn}_${device_name_f1_cpu}_ocl_config_single.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --scenarioName friedman1_${device_name_f1_cpu}_weakscaling_parameters --level 7 --tuner_name line_search --repetitions 10 --isModLinear true --file_prefix results_diss/ --trans true

#     # compute performance of final parameters
#     ./datadriven/examplesOCL/regressionGFlops --OpenCLConfigFile results_diss/friedman1_${device_name_f1_cpu}_weakscaling_parameters_mult_host_argon-gtx_tuner_line_search_t_float_10r_optimal.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --isModLinear true --trans false --level 7 --repetitions 10 > results_diss/friedman1_performance_mult_float_${device_name_f1_cpu}.log 2>&1

#     ./datadriven/examplesOCL/regressionGFlops --OpenCLConfigFile results_diss/friedman1_${device_name_f1_cpu}_weakscaling_parameters_multTrans_host_argon-gtx_tuner_line_search_t_float_10r_optimal.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --isModLinear true --trans true --level 7 --repetitions 10 > results_diss/friedman1_performance_multTranspose_float_${device_name_f1_cpu}.log 2>&1
# fi
