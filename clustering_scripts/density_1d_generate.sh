#!/bin/bash

level=4


./datadriven/examplesOCL/clustering_cmd --datasetFileName results_WPDM18/tiny.arff --level ${level} --threshold 1.0 --lambda 1.0 --density_eval_full_grid_level ${level} --write_all --file_prefix results_WPDM18/lambda_experiments_tiny_1_0 --config OCL_configs/config_ocl_float_i76700k.cfg --write_evaluated_density_full_grid

./datadriven/examplesOCL/clustering_cmd --datasetFileName results_WPDM18/tiny.arff --level ${level} --threshold 1.0 --lambda 0.2 --density_eval_full_grid_level ${level} --write_all --file_prefix results_WPDM18/lambda_experiments_tiny_0_2 --config OCL_configs/config_ocl_float_i76700k.cfg --write_evaluated_density_full_grid

./datadriven/examplesOCL/clustering_cmd --datasetFileName results_WPDM18/tiny.arff --level ${level} --threshold 1.0 --lambda 0.05 --density_eval_full_grid_level ${level} --write_all --file_prefix results_WPDM18/lambda_experiments_tiny_0_05 --config OCL_configs/config_ocl_float_i76700k.cfg --write_evaluated_density_full_grid
