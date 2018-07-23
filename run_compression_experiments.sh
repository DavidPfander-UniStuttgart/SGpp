#!/bin/bash

deviceName=$1
echo "deviceName: $deviceName"

if [ $# -eq 0 ]; then
    echo "No arguments supplied"
    exit
fi

source source_autotunetmp_`hostname`.sh
./run_all_experiments_high_dim.py --device_name $deviceName --config_file config_ocl_float_${deviceName}_compression.cfg --precision float --with-compression --fixed-grid-points
# ./run_all_experiments_high_dim.py --device_name $deviceName --config_file config_ocl_float_${deviceName}_compression_fewer_registers.cfg --precision float --with-compression --fixed-grid-points --use-fewer-registers
./run_all_experiments_high_dim.py --device_name ${deviceName} --config_file config_ocl_float_${deviceName}_no_compression.cfg --precision float --fixed-grid-points
./run_all_experiments_high_dim.py --device_name ${deviceName} --config_file config_ocl_float_${deviceName}_compression_32.cfg --precision float --with-compression --fixed-grid-points --use-32-bits-compression

# ./run_all_experiments_high_dim.py --device_name $deviceName --config_file config_ocl_float_${deviceName}_compression.cfg --precision float --with-compression
# # ./run_all_experiments_high_dim.py --device_name $deviceName --config_file config_ocl_float_${deviceName}_compression_fewer_registers.cfg --precision float --with-compression --use-fewer-registers
# ./run_all_experiments_high_dim.py --device_name ${deviceName} --config_file config_ocl_float_${deviceName}_no_compression.cfg --precision float
# ./run_all_experiments_high_dim.py --device_name ${deviceName} --config_file config_ocl_float_${deviceName}_compression_32.cfg --precision float --with-compression --use-32-bits-compression
