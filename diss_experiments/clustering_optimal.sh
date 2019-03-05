#!/bin/bash
set -x

device="gtx1080ti"

coarsen_threshold="1.0E-5"
max_iterations="50"

############## 1M-10C ##############
lambda="1.778279410038923e-08"
threshold="4074.07407407"
# direct
./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size1000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda ${lambda} --threshold ${threshold} --level 6 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing > results_diss/gaussian_c10_size1000000_dim10_noise_optimal.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size1000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c10_size1000000_dim10_noise_optimal.log 2>&1

# limited
./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size1000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda ${lambda} --threshold ${threshold} --level 6 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing --max_iterations ${max_iterations} > results_diss/gaussian_c10_size1000000_dim10_noise_optimal_limited.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size1000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c10_size1000000_dim10_noise_optimal_limited.log 2>&1

# coarsen
./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size1000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda ${lambda} --threshold ${threshold} --level 6 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing --use_b_coarsening --b_coarsening_threshold ${coarsen_threshold} > results_diss/gaussian_c10_size1000000_dim10_noise_optimal_coarsen.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size1000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c10_size1000000_dim10_noise_optimal_coarsen.log 2>&1

# coarsen + limited
./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size1000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda ${lambda} --threshold ${threshold} --level 6 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing --max_iterations ${max_iterations} --use_b_coarsening --b_coarsening_threshold ${coarsen_threshold} > results_diss/gaussian_c10_size1000000_dim10_noise_optimal_coarsen_limited.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size1000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c10_size1000000_dim10_noise_optimal_coarsen_limited.log 2>&1

################# 1M-100C ###################
lambda="1.778279410038923e-08"
threshold="1666.66666667"
# direct
./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda ${lambda} --threshold ${threshold} --level 7 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing > results_diss/gaussian_c100_size1000000_dim10_noise_optimal.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c100_size1000000_dim10_noise_optimal.log 2>&1

# limited
./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda ${lambda} --threshold ${threshold} --level 7 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing --max_iterations ${max_iterations} > results_diss/gaussian_c100_size1000000_dim10_noise_optimal_limited.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c100_size1000000_dim10_noise_optimal_limited.log 2>&1

# coarsen
./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda ${lambda} --threshold ${threshold} --level 7 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing --b_coarsening_threshold ${coarsen_threshold} > results_diss/gaussian_c100_size1000000_dim10_noise_optimal_coarsen.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c100_size1000000_dim10_noise_optimal_coarsen.log 2>&1

# limited + coarsen
./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda ${lambda} --threshold ${threshold} --level 7 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing --max_iterations ${max_iterations} --use_b_coarsening --b_coarsening_threshold ${coarsen_threshold} > results_diss/gaussian_c100_size1000000_dim10_noise_optimal_coarsen_limited.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c100_size1000000_dim10_noise_optimal_coarsen_limited.log 2>&1

######################## 1M-100C, special run for improved runtime ###################
lambda="1.778279410038923e-08"
threshold="1358.02469136"
# direct
./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda ${lambda} --threshold ${threshold} --level 7 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing > results_diss/gaussian_c100_size1000000_dim10_noise_optimal_faster.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c100_size1000000_dim10_noise_optimal_faster.log 2>&1

# limited
./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda ${lambda} --threshold ${threshold} --level 7 --epsilon 1E-2 --k 6 --max_iterations ${max_iterations} --write_cluster_map --file_prefix data_ref_testing > results_diss/gaussian_c100_size1000000_dim10_noise_optimal_faster_limited.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c100_size1000000_dim10_noise_optimal_faster_limited.log 2>&1

# coarsen
./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda ${lambda} --threshold ${threshold} --level 7 --epsilon 1E-2 --k 6 --use_b_coarsening --b_coarsening_threshold ${coarsen_threshold} --write_cluster_map --file_prefix data_ref_testing > results_diss/gaussian_c100_size1000000_dim10_noise_optimal_faster_coarsen.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c100_size1000000_dim10_noise_optimal_faster_coarsen.log 2>&1

# limited + coarsen
./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda ${lambda} --threshold ${threshold} --level 7 --epsilon 1E-2 --k 6 --max_iterations ${max_iterations} --use_b_coarsening --b_coarsening_threshold ${coarsen_threshold} --write_cluster_map --file_prefix data_ref_testing > results_diss/gaussian_c100_size1000000_dim10_noise_optimal_faster_coarsen_limited.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size1000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c100_size1000000_dim10_noise_optimal_faster_coarsen_limited.log 2>&1

################## 10M-10C ################
lambda="1e-06"
threshold="5185.18518519"
# direct
./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size10000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda ${lambda} --threshold ${threshold} --level 7 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing > results_diss/gaussian_c10_size10000000_dim10_noise_optimal.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size10000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c10_size10000000_dim10_noise_optimal.log 2>&1

# limited
./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size10000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda ${lambda} --threshold ${threshold} --level 7 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing --max_iterations ${max_iterations} > results_diss/gaussian_c10_size10000000_dim10_noise_optimal_limited.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size10000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c10_size10000000_dim10_noise_optimal_limited.log 2>&1

# coarsen
./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size10000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda ${lambda} --threshold ${threshold} --level 7 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing --use_b_coarsening --b_coarsening_threshold ${coarsen_threshold} > results_diss/gaussian_c10_size10000000_dim10_noise_optimal_coarsen.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size10000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c10_size10000000_dim10_noise_optimal_coarsen.log 2>&1

# limited + coarsen
./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size10000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda ${lambda} --threshold ${threshold} --level 7 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing --max_iterations ${max_iterations} --use_b_coarsening --b_coarsening_threshold ${coarsen_threshold} > results_diss/gaussian_c10_size10000000_dim10_noise_optimal_coarsen_limited.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c10_size10000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c10_size10000000_dim10_noise_optimal_coarsen_limited.log 2>&1

########## 10M-100C #####
lambda="1.778279410038923e-08"
threshold="3641.97530864"
# direct
./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size10000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda ${lambda} --threshold ${threshold} --level 7 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing > results_diss/gaussian_c100_size10000000_dim10_noise_optimal.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size10000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c100_size10000000_dim10_noise_optimal.log 2>&1

# limited
./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size10000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda ${lambda} --threshold ${threshold} --level 7 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing --max_iterations ${max_iterations}  > results_diss/gaussian_c100_size10000000_dim10_noise_optimal_limited.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size10000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c100_size10000000_dim10_noise_optimal_limited.log 2>&1

# coarsen
./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size10000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda ${lambda} --threshold ${threshold} --level 7 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing --use_b_coarsening --b_coarsening_threshold ${coarsen_threshold} > results_diss/gaussian_c100_size10000000_dim10_noise_optimal_coarsen.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size10000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c100_size10000000_dim10_noise_optimal_coarsen.log 2>&1

# limited + coarsen
./datadriven/examplesOCL/clustering_cmd --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size10000000_dim10_noise --config OCL_configs/config_ocl_float_${device}.cfg --lambda ${lambda} --threshold ${threshold} --level 7 --epsilon 1E-2 --k 6 --write_cluster_map --file_prefix data_ref_testing --max_iterations ${max_iterations} --use_b_coarsening --b_coarsening_threshold ${coarsen_threshold} > results_diss/gaussian_c100_size10000000_dim10_noise_optimal_coarsen_limited.log 2>&1

./clustering_scripts/check_ARI.py --reference_cluster_assignment ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c100_size10000000_dim10_noise_class.arff --cluster_assignment data_ref_testing_cluster_map.csv >> results_diss/gaussian_c100_size10000000_dim10_noise_optimal_coarsen_limited.log 2>&1
