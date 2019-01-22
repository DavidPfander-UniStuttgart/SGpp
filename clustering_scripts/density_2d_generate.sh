#!/bin/bash

level=4

ocl_config='config_ocl_float_gtx1080ti.cfg'

./datadriven/examplesOCL/clustering_cmd --datasetFileName results_WPDM18/tiny_2d.arff --level ${level} --threshold 1.0 --lambda 10.0 --density_eval_full_grid_level ${level} --write_all --file_prefix results_WPDM18/lambda_experiments_tiny_2d_10_0 --config OCL_configs/${ocl_config} --write_evaluated_density_full_grid

./datadriven/examplesOCL/clustering_cmd --datasetFileName results_WPDM18/tiny_2d.arff --level ${level} --threshold 1.0 --lambda 1.0 --density_eval_full_grid_level ${level} --write_all --file_prefix results_WPDM18/lambda_experiments_tiny_2d_1_0 --config OCL_configs/${ocl_config} --write_evaluated_density_full_grid

./datadriven/examplesOCL/clustering_cmd --datasetFileName results_WPDM18/tiny_2d.arff --level ${level} --threshold 1.0 --lambda 0.2 --density_eval_full_grid_level ${level} --write_all --file_prefix results_WPDM18/lambda_experiments_tiny_2d_0_2 --config OCL_configs/${ocl_config} --write_evaluated_density_full_grid

./datadriven/examplesOCL/clustering_cmd --datasetFileName results_WPDM18/tiny_2d.arff --level ${level} --threshold 1.0 --lambda 0.5 --density_eval_full_grid_level ${level} --write_all --file_prefix results_WPDM18/lambda_experiments_tiny_2d_0_5 --config OCL_configs/${ocl_config} --write_evaluated_density_full_grid

./datadriven/examplesOCL/clustering_cmd --datasetFileName results_WPDM18/tiny_2d.arff --level ${level} --threshold 1.0 --lambda 0.1 --density_eval_full_grid_level ${level} --write_all --file_prefix results_WPDM18/lambda_experiments_tiny_2d_0_1 --config OCL_configs/${ocl_config} --write_evaluated_density_full_grid

./datadriven/examplesOCL/clustering_cmd --datasetFileName results_WPDM18/tiny_2d.arff --level ${level} --threshold 1.0 --lambda 0.05 --density_eval_full_grid_level ${level} --write_all --file_prefix results_WPDM18/lambda_experiments_tiny_2d_0_05 --config OCL_configs/${ocl_config} --write_evaluated_density_full_grid

./datadriven/examplesOCL/clustering_cmd --datasetFileName results_WPDM18/tiny_2d.arff --level ${level} --threshold 1.0 --lambda 0.01 --density_eval_full_grid_level ${level} --write_all --file_prefix results_WPDM18/lambda_experiments_tiny_2d_0_01 --config OCL_configs/${ocl_config} --write_evaluated_density_full_grid

# ./datadriven/examplesOCL/clustering_cmd --datasetFileName results_WPDM18/tiny_2d.arff --level ${level} --threshold 1.0 --lambda 0.05 --density_eval_full_grid_level ${level} --write_all --file_prefix results_WPDM18/lambda_experiments_tiny_2d_0_05 --config OCL_configs/config_ocl_float_i76700k.cfg --write_evaluated_density_full_grid
