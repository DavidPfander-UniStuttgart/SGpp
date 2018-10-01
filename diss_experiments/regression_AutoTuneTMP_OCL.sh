#!/bin/bash

set -x
set -e

source source_autotunetmp_generic.sh

./datadriven/examplesOCL/detectPlatform --precision float --file_name config_single.cfg

./datadriven/examplesAutoTuneTMP/tune_mult_AutoTuneTMP_OCL --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --level 6 --scenarioName tune_mult_AutoTuneTMP_OCL --tuner_name bruteforce --repetitions 5 --OpenCLConfigFile config_single.cfg
./datadriven/examplesAutoTuneTMP/tune_mult_AutoTuneTMP_OCL --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --level 6 --scenarioName tune_mult_AutoTuneTMP_OCL --tuner_name line_search --repetitions 5 --OpenCLConfigFile config_single.cfg
./datadriven/examplesAutoTuneTMP/tune_mult_AutoTuneTMP_OCL --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --level 6 --scenarioName tune_mult_AutoTuneTMP_OCL --tuner_name neighborhood_search --repetitions 5 --OpenCLConfigFile config_single.cfg

mv tune_mult_AutoTuneTMP_OCL_host_${HOSTNAME}_* ${ALL_DATA_REPO_ROOT_PATH}/sparse_grids/regression/
