#!/bin/bash

device="QuadroGP100"

# dim5id5
# datadriven refinement with 0! surplus step
./datadriven/examplesOCL/clustering_cmd --datasetFileName datasets_diss/gaussian_c100_size100000_dim5id5_noise.arff --level 15 --epsilon 1E-2 --threshold 55.0 --lambda 1.0E-5 --k 6 --config OCL_configs/config_ocl_float_${device}.cfg --write_cluster_map --file_prefix ref_testing --refinement_steps 0 --refinement_points 1000 --coarsen_points 0 --coarsen_threshold 1E-5 --max_iterations 200 --use_datadriven_refinement --datadriven_refinement_min_support 800 > results_diss/clustering_adaptive_datadriven_dim5id5.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment datasets_diss/gaussian_c100_size100000_dim5id5_noise_class.arff --cluster_assignment ref_testing_cluster_map.csv >> results_diss/clustering_adaptive_datadriven_dim5id5.log 2>&1

# dim10id5
./datadriven/examplesOCL/clustering_cmd --datasetFileName datasets_diss/gaussian_c100_size100000_dim10id5_noise.arff --level 15 --epsilon 1E-2 --threshold 13000.0 --lambda 3.0E-8 --k 6 --config OCL_configs/config_ocl_float_${device}.cfg --write_cluster_map --file_prefix ref_testing  --refinement_steps 0 --refinement_points 1000 --coarsen_points 0 --coarsen_threshold 1E-7 --max_iterations 200 --use_datadriven_refinement --datadriven_refinement_min_support 800 --write_density_grid > results_diss/clustering_adaptive_datadriven_dim10id5.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment datasets_diss/gaussian_c100_size100000_dim10id5_noise_class.arff --cluster_assignment ref_testing_cluster_map.csv >> results_diss/clustering_adaptive_datadriven_dim10id5.log 2>&1

# dim20id5
./datadriven/examplesOCL/clustering_cmd --datasetFileName datasets_diss/gaussian_c100_size100000_dim20id5_noise.arff --level 15 --epsilon 1E-2 --threshold 7.6E8 --lambda 7.8E-13 --k 6 --config OCL_configs/config_ocl_float_${device}.cfg --write_cluster_map --file_prefix ref_testing  --refinement_steps 0 --refinement_points 1000 --coarsen_points 0 --coarsen_threshold 1E-7 --max_iterations 200 --use_datadriven_refinement --datadriven_refinement_min_support 800 --write_density_grid > results_diss/clustering_adaptive_datadriven_dim20id5.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment datasets_diss/gaussian_c100_size100000_dim20id5_noise_class.arff --cluster_assignment ref_testing_cluster_map.csv >> results_diss/clustering_adaptive_datadriven_dim20id5.log 2>&1

# comparison runs

# dim5id5 with spatial refinement
./datadriven/examplesOCL/clustering_cmd --datasetFileName datasets_diss/gaussian_c100_size100000_dim5id5_noise.arff --level 7 --epsilon 1E-2 --threshold 55.0 --lambda 1.0E-5 --k 6 --config OCL_configs/config_ocl_float_QuadroGP100.cfg --write_cluster_map --file_prefix ref_testing --refinement_steps 5 --refinement_points 800 --write_density_grid > results_diss/clustering_adaptive_surplus_dim5id5.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment datasets_diss/gaussian_c100_size100000_dim5id5_noise_class.arff --cluster_assignment ref_testing_cluster_map.csv >> results_diss/clustering_adaptive_surplus_dim5id5.log 2>&1

# dim5id5 with pruning only
./datadriven/examplesOCL/clustering_cmd --datasetFileName datasets_diss/gaussian_c100_size100000_dim5id5_noise.arff --level 11 --epsilon 1E-2 --threshold 54.0 --lambda 1.0E-5 --k 6 --config OCL_configs/config_ocl_float_QuadroGP100.cfg --write_cluster_map --file_prefix ref_testing --use_b_coarsening --b_coarsening_threshold 7.5E-4 --refinement_steps 0 --refinement_points 1000 --coarsen_points 0 --coarsen_threshold 5.0E-5 > results_diss/clustering_adaptive_pruning_dim5id5.log

./clustering_scripts/check_ARI.py --reference_cluster_assignment datasets_diss/gaussian_c100_size100000_dim5id5_noise_class.arff --cluster_assignment ref_testing_cluster_map.csv >> results_diss/clustering_adaptive_pruning_dim5id5.log 2>&1

# dim10id5 with pruning only
./datadriven/examplesOCL/clustering_cmd --datasetFileName datasets_diss/gaussian_c100_size100000_dim10id5_noise.arff --level 9 --epsilon 1E-2 --threshold 13000.0 --lambda 3.0E-8 --k 6 --config OCL_configs/config_ocl_float_QuadroGP100.cfg --write_cluster_map --file_prefix ref_testing --refinement_steps 0 --refinement_points 800 --max_iterations 200 --use_b_coarsening --b_coarsening_threshold 1E-4 --write_density_grid > results_diss/clustering_adaptive_pruning_dim10id5.log

./clustering_scripts/check_ARI.py --reference_cluster_assignment datasets_diss/gaussian_c100_size100000_dim10id5_noise_class.arff --cluster_assignment ref_testing_cluster_map.csv >> results_diss/clustering_adaptive_pruning_dim10id5.log 2>&1
