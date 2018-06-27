#!/bin/bash
source source_autotunetmp_argon.sh
./run_all_experiments.py config_ocl_float_P100.cfg P100
./run_all_experiments.py config_ocl_double_P100.cfg P100
