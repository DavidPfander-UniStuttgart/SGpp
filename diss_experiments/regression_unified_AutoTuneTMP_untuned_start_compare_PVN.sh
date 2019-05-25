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
eval_repetitions=5

dataset_name=DR5
dataset_path=../datasets/DR5/DR5_nowarnings_less05_train.arff
for is_trans in false true
do
    for search_strategy in line_search neighborhood_search monte_carlo
    do

	if [ ${is_trans} = "true" ]; then
	    algorithm_name=multTrans
	else
	    algorithm_name=mult
	fi
	
        # single
	precision="float"
	./datadriven/examplesAutoTuneTMP/compare_pvn_unified --OpenCLConfigFile results_diss/unified/${dataset_name}_${device_name_f1}_untuned_${algorithm_name}_host_${hn}_tuner_${search_strategy}_t_${precision}_${eval_repetitions}av_0r_optimal.cfg --datasetFileName ${dataset_path} --level 10 --use_support_refinement --support_refinement_min_support 500 --scenarioName ${dataset_name}_${device_name_f1}_${precision} --repetitions_averaged ${eval_repetitions} --trans ${is_trans} --isModLinear true --file_prefix results_diss/unified/
	
        # double
	if [ ${hn} != "argon-gtx" ]; then

	    precision="double"
	    ./datadriven/examplesAutoTuneTMP/compare_pvn_unified --OpenCLConfigFile results_diss/unified/${dataset_name}_${device_name_f1}_untuned_${algorithm_name}_host_${hn}_tuner_${search_strategy}_t_${precision}_${eval_repetitions}av_0r_optimal.cfg --datasetFileName ${dataset_path} --level 10 --use_support_refinement --support_refinement_min_support 500 --scenarioName ${dataset_name}_${device_name_f1}_${precision} --repetitions_averaged ${eval_repetitions} --trans ${is_trans} --isModLinear true --file_prefix results_diss/unified/
	    
        fi
        # special case for Xeon Gold on argon-gtx
        if [ ${hn} = "argon-gtx" ]; then

	    precision="float"
	    ./datadriven/examplesAutoTuneTMP/compare_pvn_unified --OpenCLConfigFile results_diss/unified/${dataset_name}_${device_name_f1_cpu}_untuned_${algorithm_name}_host_${hn}_tuner_${search_strategy}_t_${precision}_${eval_repetitions}av_0r_optimal.cfg --datasetFileName ${dataset_path} --level 10 --use_support_refinement --support_refinement_min_support 500 --scenarioName ${dataset_name}_${device_name_f1_cpu}_${precision} --repetitions_averaged ${eval_repetitions} --trans ${is_trans} --isModLinear true --file_prefix results_diss/unified/
	    
        fi
    done
done

dataset_name=friedman1
dataset_path=../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff
for is_trans in false true
do
    for search_strategy in line_search neighborhood_search monte_carlo
    do

	if [ ${is_trans} = "true" ]; then
	    algorithm_name=multTrans
	else
	    algorithm_name=mult
	fi
	
        # single
	precision="float"
	./datadriven/examplesAutoTuneTMP/compare_pvn_unified --OpenCLConfigFile results_diss/unified/${dataset_name}_${device_name_f1}_untuned_${algorithm_name}_host_${hn}_tuner_${search_strategy}_t_${precision}_${eval_repetitions}av_0r_optimal.cfg --datasetFileName ${dataset_path} --level 7 --scenarioName ${dataset_name}_${device_name_f1}_${precision} --repetitions_averaged ${eval_repetitions} --trans ${is_trans} --isModLinear true --file_prefix results_diss/unified/
	
        # double
	if [ ${hn} != "argon-gtx" ]; then

	    precision="double"
	    ./datadriven/examplesAutoTuneTMP/compare_pvn_unified --OpenCLConfigFile results_diss/unified/${dataset_name}_${device_name_f1}_untuned_${algorithm_name}_host_${hn}_tuner_${search_strategy}_t_${precision}_${eval_repetitions}av_0r_optimal.cfg --datasetFileName ${dataset_path} --level 7 --scenarioName ${dataset_name}_${device_name_f1}_${precision} --repetitions_averaged ${eval_repetitions} --trans ${is_trans} --isModLinear true --file_prefix results_diss/unified/
	    
        fi
        # special case for Xeon Gold on argon-gtx
        if [ ${hn} = "argon-gtx" ]; then

	    precision="float"
	    ./datadriven/examplesAutoTuneTMP/compare_pvn_unified --OpenCLConfigFile results_diss/unified/${dataset_name}_${device_name_f1_cpu}_untuned_${algorithm_name}_host_${hn}_tuner_${search_strategy}_t_${precision}_${eval_repetitions}av_0r_optimal.cfg --datasetFileName ${dataset_path} --level 7 --scenarioName ${dataset_name}_${device_name_f1_cpu}_${precision} --repetitions_averaged ${eval_repetitions} --trans ${is_trans} --isModLinear true --file_prefix results_diss/unified/
	    
        fi
	
    done
done
