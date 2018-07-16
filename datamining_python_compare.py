#!/usr/bin/python

import csv
import time
import subprocess
import shlex

f = open('python_cpp_compare.csv', 'wb')
fieldnames = ['dataset_size', 'duration']
writer = csv.DictWriter(f, fieldnames=fieldnames)
writer.writeheader()
for numElements in [10000, 20000, 40000, 60000, 80000, 100000]:
    print "work on dataset with " + str(numElements) + " elements"
    fileName = "friedman2_4d_" + str(numElements) + ".arff"
    command = "./datadriven/examplesOCL/learner --trainingFileName datasets/friedman/python_c_compare/" + fileName + " --lambda 1E-2 --verbose true --learnerMode LEARN --grid.level 8 --grid.type Linear --solverFinal.eps 1E-12 --solverFinal.maxIterations 1000 --operation.type STREAMING --operation.subType DEFAULT"
    print command
    timer_start = time.time()
    p = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    output = p.communicate()
    print "stdout:"
    print output[0]
    print "stderr:"
    print output[1]
    timer_stop = time.time()
    last_duration = timer_stop - timer_start
    # surplusses = learnLeastSquares(fileName, level, dim, lambdaParameter)
    writer.writerow({'dataset_size': str(numElements), 'duration': str(last_duration)})
