#!/bin/bash
set -x

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

echo "hostname: ${hn}"
echo "device_name: ${device_name}"
echo "AUTOTUNETMP_SELECT: ${AUTOTUNETMP_SELECT}"
eval_repetitions=5

for dataset_name in DR5 friedman1
do
    if [ ${dataset_name} = "DR5" ]; then
        dataset_path='../datasets/DR5/DR5_nowarnings_less05_train.arff'
        grid_config='--level 10 --use_support_refinement --support_refinement_min_support 500'
    else
        dataset_path='../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff'
        grid_config='--level 7'
    fi
    for is_trans in false true
    do
        for search_strategy in line_search
        do

            if [ ${is_trans} = "true" ]; then
                algorithm_name='multTrans'
            else
                algorithm_name='mult'
            fi
            ./datadriven/examplesAutoTuneTMP/compare_pvn_subspace --additionalConfig results_diss/subspace/${dataset_name}_${device_name}_untuned_Subspace_${algorithm_name}_tuner_${search_strategy}_t_${eval_repetitions}av_0r_optimal.cfg --datasetFileName ${dataset_path} ${grid_config} --scenarioName ${dataset_name}_${device_name} --tuner_name ${search_strategy} --repetitions_averaged ${eval_repetitions} --trans ${is_trans} --isModLinear true --file_prefix results_diss/subspace/
        done
    done
done
