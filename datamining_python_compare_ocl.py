#!/usr/bin/python

import csv
import time
import subprocess
import shlex
import re

import argparse
parser = argparse.ArgumentParser(description='Run comparison stuff on OCL device.')
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument('--device_name', required=True)
args = parser.parse_args()

f = open('python_cpp_compare_ocl_10d.csv', 'wb')
fieldnames = ['dataset_size', 'duration', 'gflops', 'final_residuum']
writer = csv.DictWriter(f, fieldnames=fieldnames)
writer.writeheader()
for numElements in [1000, 10000, 20000, 40000, 60000, 80000, 100000]:
    print "work on dataset with " + str(numElements) + " elements"
    print "using device: " + str(args.device_name)
    fileName = "friedman1_10d_" + str(numElements) + ".arff"
    command = "./datadriven/examplesOCL/learner --trainingFileName datasets/friedman/python_c_compare/" + fileName + " --lambda 1E-2 --verbose true --learnerMode LEARN --grid.level 8 --grid.type Linear --solverFinal.eps 1E-12 --solverFinal.maxIterations 1000 --operation.type STREAMING --operation.subType OCLMP --additionalConfig config_ocl_double_" + str(args.device_name) +"_regression.cfg"
    print command
    timer_start = time.time()
    p = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    output = p.communicate()
    print "stdout:"
    print output[0]
    pattern = r"Training took\: (.*?) seconds"
    c = re.compile(pattern)
    m = c.search(output[0])
    last_duration = float(m.group(1))
    print "last_duration:", last_duration

    pattern = r"Current GFlop\/s\: (.*?)\n"
    c = re.compile(pattern)
    m = c.search(output[0])
    gflops = float(m.group(1))
    print "gflops:", gflops

    pattern = r"Final residuum: (.*?)\n"
    c = re.compile(pattern)
    m = c.search(output[0])
    final_residuum = float(m.group(1))
    print "final_residuum:", final_residuum

    # print "stderr:"
    # print output[1]
    timer_stop = time.time()
    last_duration = timer_stop - timer_start
    # surplusses = learnLeastSquares(fileName, level, dim, lambdaParameter)
    writer.writerow({'dataset_size': str(numElements), 'duration': str(last_duration), 'gflops': str(gflops), 'final_residuum': str(final_residuum)})
