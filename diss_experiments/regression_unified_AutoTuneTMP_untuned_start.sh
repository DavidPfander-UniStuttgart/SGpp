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
elif [ ${hn} = "large" ]; then
    device_name_f1='E52670'
    AUTOTUNETMP_SELECT="Intel(R) CPU Runtime for OpenCL(TM) Applications/       Intel(R) Xeon(R) CPU E5-2670 0 @ 2.60GHz"
fi
echo "hostname: ${hn}"
echo "device_name_f1: ${device_name_f1}"
echo "AUTOTUNETMP_SELECT: ${AUTOTUNETMP_SELECT}"
tuner_repetitions=1
eval_repetitions=5

############## friedman1 dataset ##############

# single
./datadriven/examplesOCL/detectPlatform --precision float --file_name results_diss/unified/${hn}_${device_name_f1}_ocl_config_single.cfg --select="${AUTOTUNETMP_SELECT}" --remove_unselected

# double
if [ ${hn} != "argon-gtx" ]; then
    ./datadriven/examplesOCL/detectPlatform --precision double --file_name results_diss/unified/${hn}_${device_name_f1}_ocl_config_double.cfg --select="${AUTOTUNETMP_SELECT}" --remove_unselected
fi

# special case for Xeon on argon-gtx, single only
if [ ${hn} = "argon-gtx" ]; then
    # use argon-gtx as cpu platform
    AUTOTUNETMP_SELECT="Intel(R) OpenCL/Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz"
    ./datadriven/examplesOCL/detectPlatform --precision float --file_name results_diss/unified/${hn}_${device_name_f1_cpu}_ocl_config_single.cfg --select="${AUTOTUNETMP_SELECT}" --remove_unselected
    AUTOTUNETMP_SELECT="NVIDIA CUDA/GeForce GTX 1080 Ti"
fi

# for dataset_name in DR5 friedman1
# do
#     if [ ${dataset_name} = "DR5" ]; then
#         dataset_path=../datasets/DR5/DR5_nowarnings_less05_train.arff
#     elif [ ${dataset_name} = "friedman1" ]; then
#         dataset_path=../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff
#     fi
    
#     for is_trans in false true
#     do
#         for search_strategy in line_search neighborhood_search monte_carlo
#         do
#             # single
#             ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/unified/${hn}_${device_name_f1}_ocl_config_single.cfg --datasetFileName ${dataset_path} --scenarioName ${dataset_name}_${device_name_f1}_untuned --level 7 --tuner_name ${search_strategy} --repetitions ${tuner_repetitions} --repetitions_averaged ${eval_repetitions} --isModLinear true --file_prefix results_diss/unified/ --trans ${is_trans} --randomization_enabled false
#             # double
#             if [ ${hn} != "argon-gtx" ]; then
#                 ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/unified/${hn}_${device_name_f1}_ocl_config_double.cfg --datasetFileName ${dataset_path} --scenarioName ${dataset_name}_${device_name_f1}_untuned --level 7 --tuner_name ${search_strategy} --repetitions ${tuner_repetitions} --repetitions_averaged ${eval_repetitions} --isModLinear true --file_prefix results_diss/unified/ --trans ${is_trans} --randomization_enabled false
#             fi
#             # special case for Xeon Gold on argon-gtx
#             if [ ${hn} = "argon-gtx" ]; then
#                 ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/unified/${hn}_${device_name_f1_cpu}_ocl_config_single.cfg --datasetFileName ${dataset_path} --scenarioName ${dataset_name}_${device_name_f1_cpu}_untuned --level 7 --tuner_name ${search_strategy} --repetitions ${tuner_repetitions} --repetitions_averaged ${eval_repetitions} --isModLinear true --file_prefix results_diss/unified/ --trans ${is_trans} --randomization_enabled false
#             fi
#         done
#     done
# done

dataset_name=DR5
dataset_path=../datasets/DR5/DR5_nowarnings_less05_train.arff
for is_trans in false true
do
    for search_strategy in line_search neighborhood_search monte_carlo
    do
        # single
        ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/unified/${hn}_${device_name_f1}_ocl_config_single.cfg --datasetFileName ${dataset_path} --scenarioName ${dataset_name}_${device_name_f1}_untuned  --level 10 --use_support_refinement --support_refinement_min_support 500 --tuner_name ${search_strategy} --repetitions ${tuner_repetitions} --repetitions_averaged ${eval_repetitions} --isModLinear true --file_prefix results_diss/unified/ --trans ${is_trans} --randomization_enabled false
        # double
        if [ ${hn} != "argon-gtx" ]; then
            ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/unified/${hn}_${device_name_f1}_ocl_config_double.cfg --datasetFileName ${dataset_path} --scenarioName ${dataset_name}_${device_name_f1}_untuned --level 10 --use_support_refinement --support_refinement_min_support 500 --tuner_name ${search_strategy} --repetitions ${tuner_repetitions} --repetitions_averaged ${eval_repetitions} --isModLinear true --file_prefix results_diss/unified/ --trans ${is_trans} --randomization_enabled false
        fi
        # special case for Xeon Gold on argon-gtx
        if [ ${hn} = "argon-gtx" ]; then
            ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/unified/${hn}_${device_name_f1_cpu}_ocl_config_single.cfg --datasetFileName ${dataset_path} --scenarioName ${dataset_name}_${device_name_f1_cpu}_untuned --level 10 --use_support_refinement --support_refinement_min_support 500 --tuner_name ${search_strategy} --repetitions ${tuner_repetitions} --repetitions_averaged ${eval_repetitions} --isModLinear true --file_prefix results_diss/unified/ --trans ${is_trans} --randomization_enabled false
        fi
    done
done

dataset_name=friedman1
dataset_path=../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff
for is_trans in false true
do
    for search_strategy in line_search neighborhood_search monte_carlo
    do
        # single
        ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/unified/${hn}_${device_name_f1}_ocl_config_single.cfg --datasetFileName ${dataset_path} --scenarioName ${dataset_name}_${device_name_f1}_untuned --level 7 --tuner_name ${search_strategy} --repetitions ${tuner_repetitions} --repetitions_averaged ${eval_repetitions} --isModLinear true --file_prefix results_diss/unified/ --trans ${is_trans} --randomization_enabled false
        # double
        if [ ${hn} != "argon-gtx" ]; then
            ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/unified/${hn}_${device_name_f1}_ocl_config_double.cfg --datasetFileName ${dataset_path} --scenarioName ${dataset_name}_${device_name_f1}_untuned --level 7 --tuner_name ${search_strategy} --repetitions ${tuner_repetitions} --repetitions_averaged ${eval_repetitions} --isModLinear true --file_prefix results_diss/unified/ --trans ${is_trans} --randomization_enabled false
        fi
        # special case for Xeon Gold on argon-gtx
        if [ ${hn} = "argon-gtx" ]; then
            ./datadriven/examplesAutoTuneTMP/tune_unified_AutoTuneTMP_OCL --OpenCLConfigFile results_diss/unified/${hn}_${device_name_f1_cpu}_ocl_config_single.cfg --datasetFileName ${dataset_path} --scenarioName ${dataset_name}_${device_name_f1_cpu}_untuned --level 7 --tuner_name ${search_strategy} --repetitions ${tuner_repetitions} --repetitions_averaged ${eval_repetitions} --isModLinear true --file_prefix results_diss/unified/ --trans ${is_trans} --randomization_enabled false
        fi
    done
done
