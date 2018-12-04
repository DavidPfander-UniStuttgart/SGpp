#!/bin/bash

set -x
set -e

source source_autotunetmp_generic.sh

./datadriven/examplesOCL/detectPlatform --precision float --file_name ${HOSTNAME}_h_config_single.cfg --select="${AUTOTUNETMP_SELECT}" --remove_unselected

./datadriven/examplesAutoTuneTMP/tune_mult_AutoTuneTMP_OCL --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --level 8 --scenarioName AutoTuneTMP_regression --tuner_name bruteforce --repetitions 5 --OpenCLConfigFile ${HOSTNAME}_h_config_single.cfg
./datadriven/examplesAutoTuneTMP/tune_mult_AutoTuneTMP_OCL --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --level 8 --scenarioName AutoTuneTMP_regression --tuner_name line_search --repetitions 5 --OpenCLConfigFile ${HOSTNAME}_h_config_single.cfg
./datadriven/examplesAutoTuneTMP/tune_mult_AutoTuneTMP_OCL --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --level 8 --scenarioName AutoTuneTMP_regression --tuner_name neighborhood_search --repetitions 5 --OpenCLConfigFile ${HOSTNAME}_h_config_single.cfg

mv AutoTuneTMP_regression_mult_host_${HOSTNAME}_* ${ALL_DATA_REPO_ROOT_PATH}/sparse_grids/regression/

./datadriven/examplesAutoTuneTMP/tune_mult_AutoTuneTMP_OCL --trans true --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --level 8 --scenarioName AutoTuneTMP_regression --tuner_name bruteforce --repetitions 5 --OpenCLConfigFile ${HOSTNAME}_h_config_single.cfg
./datadriven/examplesAutoTuneTMP/tune_mult_AutoTuneTMP_OCL --trans true --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --level 8 --scenarioName AutoTuneTMP_regression --tuner_name line_search --repetitions 1 --OpenCLConfigFile ${HOSTNAME}_h_config_single.cfg
./datadriven/examplesAutoTuneTMP/tune_mult_AutoTuneTMP_OCL --trans true --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --level 8 --scenarioName AutoTuneTMP_regression --tuner_name neighborhood_search --repetitions 5 --OpenCLConfigFile ${HOSTNAME}_h_config_single.cfg

mv AutoTuneTMP_regression_multTrans_host_${HOSTNAME}_* ${ALL_DATA_REPO_ROOT_PATH}/sparse_grids/regression/

mv ${HOSTNAME}_h_config_single.cfg ${ALL_DATA_REPO_ROOT_PATH}/sparse_grids/regression/
