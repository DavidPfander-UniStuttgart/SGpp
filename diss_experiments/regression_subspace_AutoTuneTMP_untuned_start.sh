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

tuner_repetitions=1
eval_repetitions=5

echo "hostname: ${hn}"
echo "device_name_f1: ${device_name}"

source source_autotunetmp_generic.sh

###### tune all cases with multiple tuners for the autotunetmp evaluation ######

# DR5
./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --file_prefix results_diss/subspace/ --scenarioName DR5_${device_name}_untuned --level 10 --tuner_name line_search --repetitions ${tuner_repetitions} --use_support_refinement --support_refinement_min_support 500 --trans true --repetitions_averaged ${eval_repetitions} --randomization_enabled false
./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --file_prefix results_diss/subspace/ --scenarioName DR5_${device_name}_untuned --level 10 --tuner_name neighborhood_search --repetitions ${tuner_repetitions} --use_support_refinement --support_refinement_min_support 500 --trans true --repetitions_averaged ${eval_repetitions} --randomization_enabled false
./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --file_prefix results_diss/subspace/ --scenarioName DR5_${device_name}_untuned --level 10 --tuner_name monte_carlo --repetitions ${tuner_repetitions} --use_support_refinement --support_refinement_min_support 500 --trans true --repetitions_averaged ${eval_repetitions} --randomization_enabled false

./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --file_prefix results_diss/subspace/ --scenarioName DR5_${device_name}_untuned --level 10 --tuner_name line_search --repetitions ${tuner_repetitions} --use_support_refinement --support_refinement_min_support 500 --trans false --repetitions_averaged ${eval_repetitions} --randomization_enabled false
./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --file_prefix results_diss/subspace/ --scenarioName DR5_${device_name}_untuned --level 10 --tuner_name neighborhood_search --repetitions ${tuner_repetitions} --use_support_refinement --support_refinement_min_support 500 --trans false --repetitions_averaged ${eval_repetitions} --randomization_enabled false
./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --file_prefix results_diss/subspace/ --scenarioName DR5_${device_name}_untuned --level 10 --tuner_name monte_carlo --repetitions ${tuner_repetitions} --use_support_refinement --support_refinement_min_support 500 --trans false --repetitions_averaged ${eval_repetitions} --randomization_enabled false

# friedman1
./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --file_prefix results_diss/subspace/ --scenarioName friedman1_${device_name}_untuned --level 7 --tuner_name line_search --repetitions ${tuner_repetitions} --trans true --repetitions_averaged ${eval_repetitions} --randomization_enabled false
./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --file_prefix results_diss/subspace/ --scenarioName friedman1_${device_name}_untuned --level 7 --tuner_name neighborhood_search --repetitions ${tuner_repetitions} --trans true --repetitions_averaged ${eval_repetitions} --randomization_enabled false
./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --file_prefix results_diss/subspace/ --scenarioName friedman1_${device_name}_untuned --level 7 --tuner_name monte_carlo --repetitions ${tuner_repetitions} --trans true --repetitions_averaged ${eval_repetitions} --randomization_enabled false

./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --file_prefix results_diss/subspace/ --scenarioName friedman1_${device_name}_untuned --level 7 --tuner_name line_search --repetitions ${tuner_repetitions} --trans false --repetitions_averaged ${eval_repetitions} --randomization_enabled false
./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --file_prefix results_diss/subspace/ --scenarioName friedman1_${device_name}_untuned --level 7 --tuner_name neighborhood_search --repetitions ${tuner_repetitions} --trans false --repetitions_averaged ${eval_repetitions} --randomization_enabled false
./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --file_prefix results_diss/subspace/ --scenarioName friedman1_${device_name}_untuned --level 7 --tuner_name monte_carlo --repetitions ${tuner_repetitions} --trans false --repetitions_averaged ${eval_repetitions} --randomization_enabled false

