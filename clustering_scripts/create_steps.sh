#!/bin/bash

config=$1
echo "config: $config"
eval_grid_level=$2
echo "eval_grid_level: $eval_grid_level"
deviceName=$3
echo "deviceName: $deviceName"

if [ $# -eq 0 ]; then
    echo "No arguments supplied"
    exit
fi

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

# ./clustering_scripts/create_clustering_graphs.py --eval_grid_level $eval_grid_level --file_prefix $scenarioName --dataset_name $datasetFileName --dont_display_grid --dont_display_knn
# ./clustering_scripts/create_clustering_graphs.py --eval_grid_level $eval_grid_level --file_prefix $scenarioName --dataset_name $datasetFileName --dont_display_knn
# ./clustering_scripts/create_clustering_graphs.py --eval_grid_level $eval_grid_level --file_prefix $scenarioName --dataset_name $datasetFileName --use_unpruned_graph
# ./clustering_scripts/create_clustering_graphs.py --eval_grid_level $eval_grid_level --file_prefix $scenarioName --dataset_name $datasetFileName
# ./clustering_scripts/create_clustering_graphs.py --eval_grid_level $eval_grid_level --file_prefix $scenarioName --dataset_name $datasetFileName --use_unpruned_graph --dont_display_grid
# ./clustering_scripts/create_clustering_graphs.py --eval_grid_level $eval_grid_level --file_prefix $scenarioName --dataset_name $datasetFileName --dont_display_grid

# plotable clusters dataset
knn_algorithm="naive_ocl"
lsh_tables=50
lsh_hashes=15
lsh_w=1.5
results_folder="results_WPDM18/"
scenarioName="gaussian_plotable_${deviceName}_knn_${knn_algorithm}"
echo "scenarioName: $scenarioName"
datasetFileName="datasets/gaussian_c3_size200_dim2.arff"
echo "datasetFileName: $datasetFileName"
level="3"
echo "level: $level"
lambda="2.5E-2" # 2.5E-2
echo "lambda: $lambda"
# threshold="1.1" # was 10 ref points
threshold="1.15"
echo "threshold: $threshold"
k="5"
echo "k: k"

echo "./datadriven/examplesOCL/clustering_cmd --datasetFileName $datasetFileName --level $level --lambda $lambda --threshold $threshold --k $k --config $config --file_prefix ${results_folder}$scenarioName --density_eval_full_grid_level $eval_grid_level --refinement_steps 4 --refinement_points 10 --coarsen_points 999 --coarsen_threshold 0.01 --write_density_grid true --write_evaluated_density_full_grid true --write_knn_graph true --write_pruned_knn_graph true --write_cluster_map true --knn_algorithm $knn_algorithm --lsh_tables $lsh_tables --lsh_hashes $lsh_hashes --lsh_w $lsh_w"

./datadriven/examplesOCL/clustering_cmd --datasetFileName $datasetFileName --level $level --lambda $lambda --threshold $threshold --k $k --config $config --file_prefix ${results_folder}$scenarioName --density_eval_full_grid_level $eval_grid_level --refinement_steps 4 --refinement_points 7 --coarsen_points 999 --coarsen_threshold 0.01 --write_density_grid true --write_evaluated_density_full_grid true --write_knn_graph true --write_pruned_knn_graph true --write_cluster_map true --knn_algorithm $knn_algorithm --lsh_tables $lsh_tables --lsh_hashes $lsh_hashes --lsh_w $lsh_w

# echo "./clustering_scripts/create_clustering_graphs.py --knn_algorithm $knn_algorithm --eval_grid_level $eval_grid_level --file_prefix $scenarioName --dataset_name $datasetFileName --dont_display_grid --dont_display_knn --dont_display_data"

./clustering_scripts/create_clustering_graphs.py --knn_algorithm $knn_algorithm --eval_grid_level $eval_grid_level --results_folder=$results_folder --file_prefix $scenarioName --dataset_name $datasetFileName --dont_display_grid --dont_display_knn --dont_display_data
./clustering_scripts/create_clustering_graphs.py --knn_algorithm $knn_algorithm --eval_grid_level $eval_grid_level --results_folder=$results_folder --file_prefix $scenarioName --dataset_name $datasetFileName --dont_display_grid --dont_display_knn
# echo "./clustering_scripts/create_clustering_graphs.py --knn_algorithm $knn_algorithm --eval_grid_level $eval_grid_level --results_folder=$results_folder  --file_prefix $scenarioName --dataset_name $datasetFileName --dont_display_knn"
./clustering_scripts/create_clustering_graphs.py --knn_algorithm $knn_algorithm --eval_grid_level $eval_grid_level --results_folder=$results_folder  --file_prefix $scenarioName --dataset_name $datasetFileName --dont_display_knn
./clustering_scripts/create_clustering_graphs.py --knn_algorithm $knn_algorithm --eval_grid_level $eval_grid_level --results_folder=$results_folder --file_prefix $scenarioName --dataset_name $datasetFileName --use_unpruned_graph
./clustering_scripts/create_clustering_graphs.py --knn_algorithm $knn_algorithm --eval_grid_level $eval_grid_level --results_folder=$results_folder --file_prefix $scenarioName --dataset_name $datasetFileName
./clustering_scripts/create_clustering_graphs.py --knn_algorithm $knn_algorithm --eval_grid_level $eval_grid_level --results_folder=$results_folder --file_prefix $scenarioName --dataset_name $datasetFileName --use_unpruned_graph --dont_display_grid
./clustering_scripts/create_clustering_graphs.py --knn_algorithm $knn_algorithm --eval_grid_level $eval_grid_level --results_folder=$results_folder --file_prefix $scenarioName --dataset_name $datasetFileName --dont_display_grid
./clustering_scripts/create_clustering_graphs.py --knn_algorithm $knn_algorithm --eval_grid_level $eval_grid_level --results_folder=$results_folder --file_prefix $scenarioName --dataset_name $datasetFileName --dont_display_grid --dont_display_knn --dont_display_density
./clustering_scripts/create_clustering_graphs.py --knn_algorithm $knn_algorithm --eval_grid_level $eval_grid_level --results_folder=$results_folder --file_prefix $scenarioName --dataset_name $datasetFileName --dont_display_data --dont_display_knn

# # plotable clusters dataset
# scenarioName="gaussian_plotable_large_$deviceName"
# echo "scenarioName: $scenarioName"
# datasetFileName="datasets/gaussian_c3_size20000_dim2.arff"
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

# ./clustering_scripts/create_clustering_graphs.py --eval_grid_level $eval_grid_level --file_prefix $scenarioName --dataset_name $datasetFileName --dont_display_grid --dont_display_knn --dont_display_data
# ./clustering_scripts/create_clustering_graphs.py --eval_grid_level $eval_grid_level --file_prefix $scenarioName --dataset_name $datasetFileName --dont_display_grid --dont_display_knn
# ./clustering_scripts/create_clustering_graphs.py --eval_grid_level $eval_grid_level --file_prefix $scenarioName --dataset_name $datasetFileName --dont_display_knn
# ./clustering_scripts/create_clustering_graphs.py --eval_grid_level $eval_grid_level --file_prefix $scenarioName --dataset_name $datasetFileName --use_unpruned_graph
# ./clustering_scripts/create_clustering_graphs.py --eval_grid_level $eval_grid_level --file_prefix $scenarioName --dataset_name $datasetFileName
# ./clustering_scripts/create_clustering_graphs.py --eval_grid_level $eval_grid_level --file_prefix $scenarioName --dataset_name $datasetFileName --use_unpruned_graph --dont_display_grid
# ./clustering_scripts/create_clustering_graphs.py --eval_grid_level $eval_grid_level --file_prefix $scenarioName --dataset_name $datasetFileName --dont_display_grid
