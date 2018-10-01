#!/bin/bash

source source_autotunetmp_generic.sh

./datadriven/examplesAutoTuneTMP/tune_mult_AutoTuneTMP --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --level 6 --scenarioName tune_mult_AutoTuneTMP --tuner_name bruteforce --repetitions 5
./datadriven/examplesAutoTuneTMP/tune_mult_AutoTuneTMP --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --level 6 --scenarioName tune_mult_AutoTuneTMP --tuner_name line_search --repetitions 5
./datadriven/examplesAutoTuneTMP/tune_mult_AutoTuneTMP --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --level 6 --scenarioName tune_mult_AutoTuneTMP --tuner_name neighborhood_search --repetitions 5
./datadriven/examplesAutoTuneTMP/tune_mult_AutoTuneTMP --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --level 6 --scenarioName tune_mult_AutoTuneTMP --tuner_name monte_carlo --repetitions 5

mv tune_mult_AutoTuneTMP_host_${HOSTNAME}_* ${ALL_DATA_REPO_ROOT_PATH}/sparse_grids/regression/
