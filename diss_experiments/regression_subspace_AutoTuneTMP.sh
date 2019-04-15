#!/bin/bash

source source_autotunetmp_generic.sh

# # DR5
# # ./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --file_prefix results_diss/ --scenarioName DR5 --level 10 --tuner_name bruteforce --repetitions 5 --use_support_refinement --support_refinement_min_support 500 --trans true --repetitions_averaged 5

# ./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --file_prefix results_diss/ --scenarioName DR5 --level 10 --tuner_name line_search --repetitions 5 --use_support_refinement --support_refinement_min_support 500 --trans true --repetitions_averaged 5
# ./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --file_prefix results_diss/ --scenarioName DR5 --level 10 --tuner_name neighborhood_search --repetitions 5 --use_support_refinement --support_refinement_min_support 500 --trans true --repetitions_averaged 5
# ./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --file_prefix results_diss/ --scenarioName DR5 --level 10 --tuner_name monte_carlo --repetitions 5 --use_support_refinement --support_refinement_min_support 500 --trans true --repetitions_averaged 5

# # ./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --file_prefix results_diss/ --scenarioName DR5 --level 10 --tuner_name bruteforce --repetitions 5 --use_support_refinement --support_refinement_min_support 500 --trans false --repetitions_averaged 5
# ./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --file_prefix results_diss/ --scenarioName DR5 --level 10 --tuner_name line_search --repetitions 5 --use_support_refinement --support_refinement_min_support 500 --trans false --repetitions_averaged 5
# ./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --file_prefix results_diss/ --scenarioName DR5 --level 10 --tuner_name neighborhood_search --repetitions 5 --use_support_refinement --support_refinement_min_support 500 --trans false --repetitions_averaged 5
# ./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --file_prefix results_diss/ --scenarioName DR5 --level 10 --tuner_name monte_carlo --repetitions 5 --use_support_refinement --support_refinement_min_support 500 --trans false --repetitions_averaged 5

# friedman1
# ./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --file_prefix results_diss/ --scenarioName friedman1 --level 10 --tuner_name bruteforce --repetitions 5 --trans true --repetitions_averaged 5
./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --file_prefix results_diss/ --scenarioName friedman1 --level 7 --tuner_name line_search --repetitions 5 --trans true --repetitions_averaged 5
./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --file_prefix results_diss/ --scenarioName friedman1 --level 7 --tuner_name neighborhood_search --repetitions 5 --trans true --repetitions_averaged 5
./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --file_prefix results_diss/ --scenarioName friedman1 --level 7 --tuner_name monte_carlo --repetitions 5 --trans true --repetitions_averaged 5

# ./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --file_prefix results_diss/ --scenarioName friedman1 --level 10 --tuner_name bruteforce --repetitions 5 --trans false --repetitions_averaged 5
./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --file_prefix results_diss/ --scenarioName friedman1 --level 7 --tuner_name line_search --repetitions 5 --trans false --repetitions_averaged 5
./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --file_prefix results_diss/ --scenarioName friedman1 --level 7 --tuner_name neighborhood_search --repetitions 5 --trans false --repetitions_averaged 5
./datadriven/examplesAutoTuneTMP/tune_Subspace --datasetFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --file_prefix results_diss/ --scenarioName friedman1 --level 7 --tuner_name monte_carlo --repetitions 5 --trans false --repetitions_averaged 5

