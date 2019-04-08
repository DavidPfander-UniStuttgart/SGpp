#!/bin/bash

#DR5 experiments
./datadriven/examplesOCL/regression_cmd --trainingFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --testFileName ../datasets/DR5/DR5_nowarnings_less05_test.arff --learnerMode LEARNTEST --grid.level 10 --lambda 1E-5 --grid.type ModLinear --verbose true --operation.type SUBSPACELINEAR --operation.subType COMBINED --solverFinal.maxIterations 300 --solverFinal.eps 0 --isRegression true --adaptConfig.support_refinement_min_support 500 --adaptConfig.use_support_refinement > results_diss/DR5_regression_Gold5120_subspacelinear_modlinear.log 2>&1

./datadriven/examplesOCL/regression_cmd --trainingFileName ../datasets/DR5/DR5_nowarnings_less05_train.arff --testFileName ../datasets/DR5/DR5_nowarnings_less05_test.arff --learnerMode LEARNTEST --grid.level 10 --lambda 1E-5 --grid.type Linear --verbose true --operation.type SUBSPACELINEAR --operation.subType COMBINED --solverFinal.maxIterations 300 --solverFinal.eps 0 --isRegression true --adaptConfig.support_refinement_min_support 500 --adaptConfig.use_support_refinement > results_diss/DR5_regression_Gold5120_subspacelinear_linear.log 2>&1

#friedman1 experiments
./datadriven/examplesOCL/regression_cmd --trainingFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --learnerMode LEARN --grid.level 7 --lambda 1E-5 --grid.type ModLinear --verbose true --operation.type SUBSPACELINEAR --operation.subType COMBINED --solverFinal.maxIterations 100 --solverFinal.eps 0 --isRegression true > results_diss/friedman1_regression_Gold5120_subspacelinear_modlinear.log 2>&1

./datadriven/examplesOCL/regression_cmd --trainingFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --learnerMode LEARN --grid.level 7 --lambda 1E-5 --grid.type Linear --verbose true --operation.type SUBSPACELINEAR --operation.subType COMBINED --solverFinal.maxIterations 100 --solverFinal.eps 0 --isRegression true > results_diss/friedman1_regression_Gold5120_subspacelinear_linear.log 2>&1

# comparison with streaming
