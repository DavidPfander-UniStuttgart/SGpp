#!/bin/bash

config=$1
echo "config: $config"
eval_grid_level=$2
echo "eval_grid_level: $eval_grid_level"
deviceName=$3
echo "deviceName: $deviceName"

# scenarioName=$1
# echo "scenarioName: $scenarioName"
# datasetFileName=$2
# echo "datasetFileName: $datasetFileName"
# level=$3
# echo "level: $level"
# lambda=$4
# echo "lambda: $lambda"
# threshold=$5
# echo "threshold: $threshold"
# k=$6
# echo "k: k"

# # ripley dataset
# scenarioName="ripley_$deviceName"
# echo "scenarioName: $scenarioName"
# datasetFileName=datasets/ripley/ripleyGarcke.train.arff
# echo "datasetFileName: $datasetFileName"
# level="4"
# echo "level: $level"
# lambda="1E-4"
# echo "lambda: $lambda"
# threshold="0.2"
# echo "threshold: $threshold"
# k="5"
# echo "k: k"
# ./datadriven/examplesOCL/clustering_cmd --datasetFileName $datasetFileName --level $level --lambda $lambda --threshold $threshold --k $k --config $config --write_graphs $scenarioName --density_eval_full_grid_level $eval_grid_level

# ./create_clustering_graphs.py --eval_grid_level $eval_grid_level --scenario_name $scenarioName --dataset_name $datasetFileName --dont_display_grid --dont_display_knn
# ./create_clustering_graphs.py --eval_grid_level $eval_grid_level --scenario_name $scenarioName --dataset_name $datasetFileName --dont_display_knn
# ./create_clustering_graphs.py --eval_grid_level $eval_grid_level --scenario_name $scenarioName --dataset_name $datasetFileName --use_unpruned_graph
# ./create_clustering_graphs.py --eval_grid_level $eval_grid_level --scenario_name $scenarioName --dataset_name $datasetFileName
# ./create_clustering_graphs.py --eval_grid_level $eval_grid_level --scenario_name $scenarioName --dataset_name $datasetFileName --use_unpruned_graph --dont_display_grid
# ./create_clustering_graphs.py --eval_grid_level $eval_grid_level --scenario_name $scenarioName --dataset_name $datasetFileName --dont_display_grid

# plotable clusters dataset
scenarioName="gaussian_plotable_$deviceName"
echo "scenarioName: $scenarioName"
datasetFileName="datasets/scikit/gaussian_d2_c3_s100.arff"
echo "datasetFileName: $datasetFileName"
level="4"
echo "level: $level"
lambda="1E-4"
echo "lambda: $lambda"
threshold="0.2"
echo "threshold: $threshold"
k="5"
echo "k: k"
./datadriven/examplesOCL/clustering_cmd --datasetFileName $datasetFileName --level $level --lambda $lambda --threshold $threshold --k $k --config $config --write_graphs $scenarioName --density_eval_full_grid_level $eval_grid_level

./create_clustering_graphs.py --eval_grid_level $eval_grid_level --scenario_name $scenarioName --dataset_name $datasetFileName --dont_display_grid --dont_display_knn
./create_clustering_graphs.py --eval_grid_level $eval_grid_level --scenario_name $scenarioName --dataset_name $datasetFileName --dont_display_knn
./create_clustering_graphs.py --eval_grid_level $eval_grid_level --scenario_name $scenarioName --dataset_name $datasetFileName --use_unpruned_graph
./create_clustering_graphs.py --eval_grid_level $eval_grid_level --scenario_name $scenarioName --dataset_name $datasetFileName
./create_clustering_graphs.py --eval_grid_level $eval_grid_level --scenario_name $scenarioName --dataset_name $datasetFileName --use_unpruned_graph --dont_display_grid
./create_clustering_graphs.py --eval_grid_level $eval_grid_level --scenario_name $scenarioName --dataset_name $datasetFileName --dont_display_grid
