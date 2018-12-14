#!/bin/bash
source source_autotunetmp_argon.sh
./run_all_experiments.py config_ocl_float_Gold5120.cfg Gold5120
./run_all_experiments.py config_ocl_double_Gold5120.cfg Gold5120
