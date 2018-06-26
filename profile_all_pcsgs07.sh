#!/bin/bash
# double profiles
/opt/CodeXL_2.5-25/rcprof -o profiling_results/prof_double_friedman2_4d_500000.csv -w . -p ./datadriven/examplesOCL/clustering_cmd --datasetFileName datasets/friedman/friedman2_4d_500000.arff --config config_ocl_double_w8100.cfg --level 9 --lambda 2E-2 --k 5 --threshold 0.8 --density_eval_full_grid_level 8 --refinement_steps 0 --refinement_points 10 --coarsen_points 10 --coarsen_threshold 0.05
/opt/CodeXL_2.5-25/rcprof -o profiling_results/prof_double_DR5.csv -w . -p ./datadriven/examplesOCL/clustering_cmd --datasetFileName datasets/DR5/DR5_nowarnings_less05_train.arff --config config_ocl_double_w8100.cfg --level 8 --lambda 2E-2 --k 5 --threshold 0.8 --density_eval_full_grid_level 8 --refinement_steps 0 --refinement_points 10 --coarsen_points 10 --coarsen_threshold 0.05
/opt/CodeXL_2.5-25/rcprof -o profiling_results/prof_double_friedman1_150000.csv -w . -p ./datadriven/examplesOCL/clustering_cmd --datasetFileName datasets/friedman/friedman1_10d_150000.arff --config config_ocl_double_w8100.cfg --level 6 --lambda 2E-2 --k 5 --threshold 0.8 --density_eval_full_grid_level 8 --refinement_steps 0 --refinement_points 10 --coarsen_points 10 --coarsen_threshold 0.05

# float profiles
/opt/CodeXL_2.5-25/rcprof -o profiling_results/prof_float_friedman2_4d_500000.csv -w . -p ./datadriven/examplesOCL/clustering_cmd --datasetFileName datasets/friedman/friedman2_4d_500000.arff --config config_ocl_float_w8100.cfg --level 9 --lambda 2E-2 --k 5 --threshold 0.8 --density_eval_full_grid_level 8 --refinement_steps 0 --refinement_points 10 --coarsen_points 10 --coarsen_threshold 0.05
/opt/CodeXL_2.5-25/rcprof -o profiling_results/prof_float_DR5.csv -w . -p ./datadriven/examplesOCL/clustering_cmd --datasetFileName datasets/DR5/DR5_nowarnings_less05_train.arff --config config_ocl_float_w8100.cfg --level 8 --lambda 2E-2 --k 5 --threshold 0.8 --density_eval_full_grid_level 8 --refinement_steps 0 --refinement_points 10 --coarsen_points 10 --coarsen_threshold 0.05
/opt/CodeXL_2.5-25/rcprof -o profiling_results/prof_float_friedman1_150000.csv -w . -p ./datadriven/examplesOCL/clustering_cmd --datasetFileName datasets/friedman/friedman1_10d_150000.arff --config config_ocl_float_w8100.cfg --level 6 --lambda 2E-2 --k 5 --threshold 0.8 --density_eval_full_grid_level 8 --refinement_steps 0 --refinement_points 10 --coarsen_points 10 --coarsen_threshold 0.05

echo "GFLOPS from here on without profiler"

# double profiles
./datadriven/examplesOCL/clustering_cmd --datasetFileName datasets/friedman/friedman2_4d_500000.arff --config config_ocl_double_w8100.cfg --level 9 --lambda 2E-2 --k 5 --threshold 0.8 --density_eval_full_grid_level 8 --refinement_steps 0 --refinement_points 10 --coarsen_points 10 --coarsen_threshold 0.05
./datadriven/examplesOCL/clustering_cmd --datasetFileName datasets/DR5/DR5_nowarnings_less05_train.arff --config config_ocl_double_w8100.cfg --level 8 --lambda 2E-2 --k 5 --threshold 0.8 --density_eval_full_grid_level 8 --refinement_steps 0 --refinement_points 10 --coarsen_points 10 --coarsen_threshold 0.05
./datadriven/examplesOCL/clustering_cmd --datasetFileName datasets/friedman/friedman1_10d_150000.arff --config config_ocl_double_w8100.cfg --level 6 --lambda 2E-2 --k 5 --threshold 0.8 --density_eval_full_grid_level 8 --refinement_steps 0 --refinement_points 10 --coarsen_points 10 --coarsen_threshold 0.05

# float profiles
./datadriven/examplesOCL/clustering_cmd --datasetFileName datasets/friedman/friedman2_4d_500000.arff --config config_ocl_float_w8100.cfg --level 9 --lambda 2E-2 --k 5 --threshold 0.8 --density_eval_full_grid_level 8 --refinement_steps 0 --refinement_points 10 --coarsen_points 10 --coarsen_threshold 0.05
./datadriven/examplesOCL/clustering_cmd --datasetFileName datasets/DR5/DR5_nowarnings_less05_train.arff --config config_ocl_float_w8100.cfg --level 8 --lambda 2E-2 --k 5 --threshold 0.8 --density_eval_full_grid_level 8 --refinement_steps 0 --refinement_points 10 --coarsen_points 10 --coarsen_threshold 0.05
./datadriven/examplesOCL/clustering_cmd --datasetFileName datasets/friedman/friedman1_10d_150000.arff --config config_ocl_float_w8100.cfg --level 6 --lambda 2E-2 --k 5 --threshold 0.8 --density_eval_full_grid_level 8 --refinement_steps 0 --refinement_points 10 --coarsen_points 10 --coarsen_threshold 0.05
