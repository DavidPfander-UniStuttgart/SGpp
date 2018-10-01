# to be sourced from shell to setup the environment
module load cuda-9.0

BASE_PATH=/scratch/pfandedd/DissertationCode/
export PATH=${BASE_PATH}/MatrixMultiplicationHPX/AutoTuneTMP/gcc_install/bin:$PATH
export LD_LIBRARY_PATH=$PWD/lib/sgpp:${BASE_PATH}/MatrixMultiplicationHPX/lshknn/build:${BASE_PATH}/MatrixMultiplicationHPX/AutoTuneTMP/boost_install/lib:$LD_LIBRARY_PATH
# export SGPP_BASE_INCLUDE_DIR=/home/pfandedd/git/SGPP_debug/base/src
export AUTOTUNETMP_INCLUDE_DIR=${BASE_PATH}/MatrixMultiplicationHPX/AutoTuneTMP/include
export VC_INCLUDE_DIR=${BASE_PATH}/MatrixMultiplicationHPX/AutoTuneTMP/Vc_install/include
export BOOST_INCLUDE_DIR=${BASE_PATH}/MatrixMultiplicationHPX/AutoTuneTMP/boost_install/include
export LSHKNN_CMAKE_CUDA_COMPILER=/usr/local.nfs/sw/cuda/cuda-9.0/bin/nvcc
export LSHKNN_CXX_COMPILER=g++-5
export VALGRIND_LIB=/home/pfandedd/usr_local/lib/valgrind/
