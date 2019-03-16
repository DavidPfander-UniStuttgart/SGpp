#!/usr/bin/python3

import subprocess
import shlex
import re
import numpy as np
import os
import re

# lambda_value = 1E-5
lambda_values = [1E-6] # 1E-5,  , 1E-7
grid_type = "ModLinear"
# level = 10 # other testerd: 10, 12
levels = [10]
dataset_name = "DR5"
dataset_train = "../datasets/DR5/DR5_nowarnings_less05_train.arff"
dataset_test = "../datasets/DR5/DR5_nowarnings_less05_test.arff"
ocl_config = "OCL_configs/config_ocl_double_Vega7.cfg"
ops_type = "STREAMING"
ops_subtype = "OCLUNIFIED"
refinementSteps = 0
refinePoints = 200
finalIterations = 400 # other test: 150

useSupportRefinement = True
# supportRefinementMinSupport = 20000
supportRefinementMinSupports = [1000] # 500, 750, 

refineIterations = finalIterations
regression = "true"
useSupportRefinement_str = ""
if useSupportRefinement:
    useSupportRefinement_str = "--adaptConfig.use_support_refinement"

supportRefinementOCLConfig = ocl_config

cmd_format = "./datadriven/examplesOCL/regression_cmd --trainingFileName {dataset_train} --testFileName {dataset_test} --learnerMode LEARNTEST --grid.level {level} --lambda {lambda_value} --grid.type {grid_type} --verbose true --additionalConfig {ocl_config} --operation.type {ops_type} --operation.subType {ops_subtype} --solverFinal.maxIterations {finalIterations} --solverFinal.eps 0 --isRegression true --solverRefine.eps 0 --solverRefine.maxIterations {refineIterations} --adaptConfig.noPoints {refinePoints} --adaptConfig.numRefinements {refinementSteps} {useSupportRefinement_str} --adaptConfig.support_refinement_min_support {supportRefinementMinSupport}  --adaptConfig.support_refinement_ocl_config {supportRefinementOCLConfig}"

results_file_name = "results_diss/" + dataset_name + "_result_regression_tuning.csv"
f_results = None
if os.path.exists(results_file_name):
    f_results = open(results_file_name, "a")
else:
    f_results = open(results_file_name, "a")
f_results.write("level, iterations, supportRefinementMinSupport, lambda_value, solver_dur, total_dur, mse\n")

log_file_name = "results_diss/" + dataset_name + "_result_regression_tuning.log"
f_log = None
if os.path.exists(log_file_name):
    f_log = open(log_file_name, "a")
else:
    f_log = open(log_file_name, "w")

for level in levels:
    for supportRefinementMinSupport in supportRefinementMinSupports:
        for lambda_value in lambda_values:
            cmd_str = cmd_format.format(cmd_format, dataset_train=dataset_train, dataset_test=dataset_test, level=level, lambda_value=lambda_value, grid_type=grid_type, ocl_config=ocl_config, ops_type=ops_type, ops_subtype=ops_subtype, finalIterations=finalIterations, refineIterations=refineIterations, refinePoints=refinePoints, refinementSteps=refinementSteps, useSupportRefinement_str=useSupportRefinement_str, supportRefinementMinSupport=supportRefinementMinSupport, supportRefinementOCLConfig=supportRefinementOCLConfig)
            # print(cmd_str)

            p = subprocess.Popen(shlex.split(cmd_str), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = p.communicate()
            o = out.decode('ascii')
            e = err.decode('ascii')
            f_log.write("-------- attempt supportRefinementMinSupport: " + str(supportRefinementMinSupport) + " lambda_value: " + str(lambda_value) + " ---------\n")
            f_log.write(o)
            f_log.write(e)
            f_log.flush()

            solver_dur = float(re.search(r"Training took: (.*?) seconds", o).group(1))
            total_dur = float(re.search(r"total_duration: (.*?)\n", o).group(1))
            mse = float(re.search(r"mse: (.*?)\n", o).group(1))

            print("solver_dur: ", solver_dur)
            print("total_dur: ", total_dur)
            print("mse:", mse)
            f_results.write(str(level) + ", " + str(finalIterations) + ", " + str(supportRefinementMinSupport) + ", " + str(lambda_value) + ", " + str(solver_dur) + ", " + str(total_dur) + ", " + str(mse) + "\n")
            f_results.flush()

f_results.close()
