#!/bin/bash
set -x

hn=`hostname`
if [ "${hn}" = "pcsgs07" ]; then
    device_name='w8100'

elif [ ${hn} = "argon-tesla1" ]; then
    device_name='XeonSilver4116'

elif [ ${hn} = "argon-tesla2" ]; then
    device_name='XeonSilver4116'

elif [ ${hn} = "argon-epyc" ]; then
    device_name='Epyc7551P'

elif [ ${hn} = "argon-gtx" ]; then
    device_name='XeonGold5120'
fi

# friedman1
echo "friedman1, host: ${hn}, device: ${device_name}"
cmd="./datadriven/examplesOCL/regression_cmd --trainingFileName ../datasets/friedman/weakscaling_regression/friedman1_10d_200000.arff --learnerMode LEARN --grid.level 7 --lambda 1E-5 --grid.type ModLinear --verbose true --operation.type SUBSPACE --operation.subType AUTOTUNETMP --solverFinal.maxIterations 100 --solverFinal.eps 0 --isRegression true"
echo ${cmd}
echo "${cmd}" > results_diss/friedman1_subspace_${hn}_${device_name}.log
./${cmd} >> results_diss/friedman1_subspace_${hn}_${device_name}.log

# DR5
echo "DR5, host: ${hn}, device: ${device_name}"
cmd="./datadriven/examplesOCL/regression_cmd --trainingFileName ../../DissertationCodeGTX/datasets/DR5/DR5_nowarnings_less05_train.arff --testFileName ../../DissertationCodeGTX/datasets/DR5/DR5_nowarnings_less05_test.arff --learnerMode LEARNTEST --grid.level 10 --lambda 1E-5 --grid.type ModLinear --verbose true --operation.type SUBSPACE --operation.subType AUTOTUNETMP --solverFinal.maxIterations 300 --solverFinal.eps 0 --isRegression true --adaptConfig.support_refinement_min_support 500 --adaptConfig.use_support_refinement"
echo "${cmd}" > results_diss/DR5_subspace_${hn}_${device_name}.log
./${cmd} >> results_diss/DR5_subspace_${hn}_${device_name}.log
