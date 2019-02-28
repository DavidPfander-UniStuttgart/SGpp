#!/bin/bash

device="gtx1080ti"

# #1M-10C
# ./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size1000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda 3.16E-7 --threshold 2222 --level 6 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing > results_diss/gaussian_c10_size1000000_dim10_noise_optimal.log 2>&1

# ./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size1000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c10_size1000000_dim10_noise_optimal.log 2>&1

# #1M-100C
# ./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda 1.78E-8 --threshold 1389 --level 7 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing > results_diss/gaussian_c100_size1000000_dim10_noise_optimal.log 2>&1

# ./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c100_size1000000_dim10_noise_optimal.log 2>&1

# #10M-10C
# ./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size10000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda 5.62E-6 --threshold 1944 --level 7 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing > results_diss/gaussian_c10_size10000000_dim10_noise_optimal.log 2>&1

# ./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size10000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c10_size10000000_dim10_noise_optimal.log 2>&1

# #10M-100C
# ./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size10000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda 1.00E-7 --threshold 1944 --level 7 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing > results_diss/gaussian_c100_size10000000_dim10_noise_optimal.log 2>&1

# ./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size10000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c100_size10000000_dim10_noise_optimal.log 2>&1

# #1M-100C, special run for improved runtime
# ./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda 1.0E-7 --threshold 1111.11 --level 7 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing > results_diss/gaussian_c100_size1000000_dim10_noise_optimal_faster.log 2>&1

# ./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c100_size1000000_dim10_noise_optimal_faster.log 2>&1

########## experiments with limited number of iterations (50) #####
#1M-10C
./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size1000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda 3.16E-7 --threshold 2222 --level 6 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing --max_iterations 50 --use_b_coarsening --b_coarsening_threshold 1.0E-5 > results_diss/gaussian_c10_size1000000_dim10_noise_optimal_limited.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size1000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c10_size1000000_dim10_noise_optimal_limited.log 2>&1

#1M-100C
./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda 1.78E-8 --threshold 1389 --level 7 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing --max_iterations 50 --use_b_coarsening --b_coarsening_threshold 1.0E-5 > results_diss/gaussian_c100_size1000000_dim10_noise_optimal_limited.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c100_size1000000_dim10_noise_optimal_limited.log 2>&1

#1M-100C, special run for improved runtime
./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda 1.0E-7 --threshold 1111.11 --level 7 --epsilon 1E-2 --k 6 --max_iterations 50 --use_b_coarsening --b_coarsening_threshold 1.0E-5 --write_cluster_map --file_prefix data_ref_testing > results_diss/gaussian_c100_size1000000_dim10_noise_optimal_faster_limited.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c100_size1000000_dim10_noise_optimal_faster_limited.log 2>&1

#10M-10C
./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size10000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda 5.62E-6 --threshold 1944 --level 7 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing --max_iterations 50 --use_b_coarsening --b_coarsening_threshold 1.0E-5 > results_diss/gaussian_c10_size10000000_dim10_noise_optimal_limited.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size10000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c10_size10000000_dim10_noise_optimal_limited.log 2>&1

#10M-100C
./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size10000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda 1.00E-7 --threshold 1944 --level 7 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing --max_iterations 50 --use_b_coarsening --b_coarsening_threshold 1.0E-5 > results_diss/gaussian_c100_size10000000_dim10_noise_optimal_limited.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size10000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c100_size10000000_dim10_noise_optimal_limited.log 2>&1


# ########### experiments with b-based coarsening enabled #################
# #1M-10C
# ./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size1000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda 3.16E-7 --threshold 2222 --level 6 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing --max_iterations 50 --use_b_coarsening --b_coarsening_threshold 1.0E-5 > results_diss/gaussian_c10_size1000000_dim10_noise_optimal_coarsen.log 2>&1

# ./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size1000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c10_size1000000_dim10_noise_optimal_coarsen.log 2>&1

# #1M-100C
# ./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda 1.78E-8 --threshold 1389 --level 7 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing --max_iterations 50 --use_b_coarsening --b_coarsening_threshold 1.0E-5 > results_diss/gaussian_c100_size1000000_dim10_noise_optimal_coarsen.log 2>&1

# ./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c100_size1000000_dim10_noise_optimal_coarsen.log 2>&1

# #1M-100C, special run for improved runtime
# ./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda 1.0E-7 --threshold 1111.11 --level 7 --epsilon 1E-2 --k 6 --use_b_coarsening --b_coarsening_threshold 1.0E-5 --write_cluster_map --file_prefix data_ref_testing > results_diss/gaussian_c100_size1000000_dim10_noise_optimal_faster_coarsen.log 2>&1

# ./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c100_size1000000_dim10_noise_optimal_faster_coarsen.log 2>&1

# #10M-10C
# ./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size10000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda 5.62E-6 --threshold 1944 --level 7 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing --max_iterations 50 --use_b_coarsening --b_coarsening_threshold 1.0E-5 > results_diss/gaussian_c10_size10000000_dim10_noise_optimal_coarsen.log 2>&1

# ./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size10000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c10_size10000000_dim10_noise_optimal_coarsen.log 2>&1

# #10M-100C
# ./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size10000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda 1.00E-7 --threshold 1944 --level 7 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing --max_iterations 50 --use_b_coarsening --b_coarsening_threshold 1.0E-5 > results_diss/gaussian_c100_size10000000_dim10_noise_optimal_coarsen.log 2>&1

# ./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size10000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c100_size10000000_dim10_noise_optimal_coarsen.log 2>&1

# ########## experiments with b-based coarsening enabled and limited number of iterations (50) #####
# #1M-10C
# ./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size1000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda 3.16E-7 --threshold 2222 --level 6 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing --max_iterations 50 --use_b_coarsening --b_coarsening_threshold 1.0E-5 > results_diss/gaussian_c10_size1000000_dim10_noise_optimal_coarsen_limited.log 2>&1

# ./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size1000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c10_size1000000_dim10_noise_optimal_coarsen_limited.log 2>&1

# #1M-100C
# ./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda 1.78E-8 --threshold 1389 --level 7 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing --max_iterations 50 --use_b_coarsening --b_coarsening_threshold 1.0E-5 > results_diss/gaussian_c100_size1000000_dim10_noise_optimal_coarsen_limited.log 2>&1

# ./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c100_size1000000_dim10_noise_optimal_coarsen_limited.log 2>&1

# #1M-100C, special run for improved runtime
# ./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda 1.0E-7 --threshold 1111.11 --level 7 --epsilon 1E-2 --k 6 --max_iterations 50 --use_b_coarsening --b_coarsening_threshold 1.0E-5 --write_cluster_map --file_prefix data_ref_testing > results_diss/gaussian_c100_size1000000_dim10_noise_optimal_faster_coarsen_limited.log 2>&1

# ./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c100_size1000000_dim10_noise_optimal_faster_coarsen_limited.log 2>&1

# #10M-10C
# ./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size10000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda 5.62E-6 --threshold 1944 --level 7 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing --max_iterations 50 --use_b_coarsening --b_coarsening_threshold 1.0E-5 > results_diss/gaussian_c10_size10000000_dim10_noise_optimal_coarsen_limited.log 2>&1

# ./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size10000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c10_size10000000_dim10_noise_optimal_coarsen_limited.log 2>&1

# #10M-100C
# ./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size10000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda 1.00E-7 --threshold 1944 --level 7 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing --max_iterations 50 --use_b_coarsening --b_coarsening_threshold 1.0E-5 > results_diss/gaussian_c100_size10000000_dim10_noise_optimal_coarsen_limited.log 2>&1

# ./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size10000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c100_size10000000_dim10_noise_optimal_coarsen_limited.log 2>&1
