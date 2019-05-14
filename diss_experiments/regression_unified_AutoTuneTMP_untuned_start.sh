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
    AUTOTUNETMP_SELECT="AMD Accelerated Parallel Processing/gfx906+sram-ecc"
elif [ ${hn} = "argon-gtx" ]; then
    device_name_f1='gtx1080ti'
    AUTOTUNETMP_SELECT="NVIDIA CUDA/GeForce GTX 1080 Ti"
    device_name_f1_cpu='Gold5120'
fi
echo "hostname: ${hn}"
echo "device_name_f1: ${device_name_f1}"
echo "AUTOTUNETMP_SELECT: ${AUTOTUNETMP_SELECT}"
tuner_repetitions=1
eval_repetitions=5

############## friedman1 dataset ##############

./datadriven/examplesOCL/detectPlatform --precision float --file_name results_diss/unified/friedman1_${hn}_${device_name_f1}_ocl_config_single.cfg --select="${AUTOTUNETMP_SELECT}" --remove_unselected

# GPU device
./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/unified/friedman1_${hn}_${device_name_f1}_ocl_config_single.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --scenarioName friedman1_${device_name_f1}_untuned --level 7 --tuner_name line_search --repetitions ${tuner_repetitions} --repetitions_averaged ${eval_repetitions} --isModLinear true --file_prefix results_diss/unified/ --trans false --randomization_enabled false

./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/unified/friedman1_${hn}_${device_name_f1}_ocl_config_single.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --scenarioName friedman1_${device_name_f1}_untuned --level 7 --tuner_name line_search --repetitions ${tuner_repetitions} --repetitions_averaged ${eval_repetitions} --isModLinear true --file_prefix results_diss/unified/ --trans true --randomization_enabled false

# this excludes gtx1080ti and the Gold5120 (the Intel OpenCL platform does not support 64bit atomics)
if [ ${hn} != "argon-gtx" ]; then
    ./datadriven/examplesOCL/detectPlatform --precision double --file_name results_diss/unified/friedman1_${hn}_${device_name_f1}_ocl_config_double.cfg --select="${AUTOTUNETMP_SELECT}" --remove_unselected

    # tune for 1 device as there is enough work guaranteed
    ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/unified/friedman1_${hn}_${device_name_f1}_ocl_config_double.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --scenarioName friedman1_${device_name_f1}_untuned --level 7 --tuner_name line_search --repetitions ${tuner_repetitions} --repetitions_averaged ${eval_repetitions} --isModLinear true --file_prefix results_diss/unified/ --trans false --randomization_enabled false

    ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/unified/friedman1_${hn}_${device_name_f1}_ocl_config_double.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --scenarioName friedman1_${device_name_f1}_untuned --level 7 --tuner_name line_search --repetitions ${tuner_repetitions} --repetitions_averaged ${eval_repetitions} --isModLinear true --file_prefix results_diss/unified/ --trans true --randomization_enabled false
fi

if [ ${hn} = "argon-gtx" ]; then
    # use argon-gtx as cpu platform
    AUTOTUNETMP_SELECT="Intel(R) OpenCL/Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz"
    ./datadriven/examplesOCL/detectPlatform --precision float --file_name results_diss/unified/friedman1_${hn}_${device_name_f1_cpu}_ocl_config_single.cfg --select="${AUTOTUNETMP_SELECT}" --remove_unselected
    AUTOTUNETMP_SELECT="NVIDIA CUDA/GeForce GTX 1080 Ti"

    ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/unified/friedman1_${hn}_${device_name_f1_cpu}_ocl_config_single.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --scenarioName friedman1_${device_name_f1_cpu}_untuned --level 7 --tuner_name line_search --repetitions ${tuner_repetitions} --repetitions_averaged ${eval_repetitions} --isModLinear true --file_prefix results_diss/unified/ --trans false --randomization_enabled false
    ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/unified/friedman1_${hn}_${device_name_f1_cpu}_ocl_config_single.cfg --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --scenarioName friedman1_${device_name_f1_cpu}_untuned --level 7 --tuner_name line_search --repetitions ${tuner_repetitions} --repetitions_averaged ${eval_repetitions} --isModLinear true --file_prefix results_diss/unified/ --trans true --randomization_enabled false
fi

./datadriven/examplesOCL/detectPlatform --precision float --file_name results_diss/unified/DR5_${hn}_${device_name_f1}_ocl_config_single.cfg --select="${AUTOTUNETMP_SELECT}" --remove_unselected

############## DR5 dataset ##############

# tune for 1 device as there is enough work guaranteed
./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/unified/DR5_${hn}_${device_name_f1}_ocl_config_single.cfg --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --scenarioName DR5_${device_name_f1}_untuned --level 10 --use_support_refinement --support_refinement_min_support 500 --tuner_name line_search --repetitions ${tuner_repetitions} --repetitions_averaged ${eval_repetitions} --isModLinear true --file_prefix results_diss/unified/ --trans false --randomization_enabled false

./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/unified/DR5_${hn}_${device_name_f1}_ocl_config_single.cfg --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --scenarioName DR5_${device_name_f1}_untuned --level 10 --use_support_refinement --support_refinement_min_support 500 --tuner_name line_search --repetitions ${tuner_repetitions} --repetitions_averaged ${eval_repetitions} --isModLinear true --file_prefix results_diss/unified/ --trans true --randomization_enabled false

# this excludes gtx1080ti and the Gold5120 (the Intel OpenCL platform does not support 64bit atomics)
if [ ${hn} != "argon-gtx" ]; then
    ./datadriven/examplesOCL/detectPlatform --precision double --file_name results_diss/unified/DR5_${hn}_${device_name_f1}_ocl_config_double.cfg --select="${AUTOTUNETMP_SELECT}" --remove_unselected

    # tune for 1 device as there is enough work guaranteed
    ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/unified/DR5_${hn}_${device_name_f1}_ocl_config_double.cfg --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --scenarioName DR5_${device_name_f1}_untuned --level 10 --use_support_refinement --support_refinement_min_support 500 --tuner_name line_search --repetitions ${tuner_repetitions} --repetitions_averaged ${eval_repetitions} --isModLinear true --file_prefix results_diss/unified/ --trans false --randomization_enabled false

    ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/unified/DR5_${hn}_${device_name_f1}_ocl_config_double.cfg --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --scenarioName DR5_${device_name_f1}_untuned --level 10 --use_support_refinement --support_refinement_min_support 500 --tuner_name line_search --repetitions ${tuner_repetitions} --repetitions_averaged ${eval_repetitions} --isModLinear true --file_prefix results_diss/unified/ --trans true --randomization_enabled false
fi

if [ ${hn} = "argon-gtx" ]; then
    # use argon-gtx as cpu platform
    AUTOTUNETMP_SELECT="Intel(R) OpenCL/Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz"
    ./datadriven/examplesOCL/detectPlatform --precision float --file_name results_diss/unified/DR5_${hn}_${device_name_f1_cpu}_ocl_config_single.cfg --select="${AUTOTUNETMP_SELECT}" --remove_unselected
    AUTOTUNETMP_SELECT="NVIDIA CUDA/GeForce GTX 1080 Ti"

    ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/unified/DR5_${hn}_${device_name_f1_cpu}_ocl_config_single.cfg --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --scenarioName DR5_${device_name_f1_cpu}_untuned --level 10 --use_support_refinement --support_refinement_min_support 500 --tuner_name line_search --repetitions ${tuner_repetitions} --repetitions_averaged ${eval_repetitions} --isModLinear true --file_prefix results_diss/unified/ --trans false --randomization_enabled false
    ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/unified/DR5_${hn}_${device_name_f1_cpu}_ocl_config_single.cfg --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --scenarioName DR5_${device_name_f1_cpu}_untuned --level 10 --use_support_refinement --support_refinement_min_support 500 --tuner_name line_search --repetitions ${tuner_repetitions} --repetitions_averaged ${eval_repetitions} --isModLinear true --file_prefix results_diss/unified/ --trans true --randomization_enabled false
fi
