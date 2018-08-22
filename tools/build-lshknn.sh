#!/bin/bash
set -x

# - to be run from project root directory
# - don't source any other file before running this scripts
# - make sure gcc <=6 is loaded
# - make sure cuda 9.0 (or possibly newer) is available

git clone git@gitlab-sgs.informatik.uni-stuttgart.de:breyerml/HighPerformanceApproxKNNAlgorithm.git lshknn

cd lshknn
git checkout build_shared_library
./build-cmake.sh
echo "cmake version: `cmake --version`"
export PATH=$PWD/cmake/bin:$PATH
echo "cmake version: `cmake --version`"
mkdir -p build
cd build
cmake -DCMAKE_CXX_COMPILER=${LSHKNN_CXX_COMPILER} -DCMAKE_CUDA_HOST_COMPILER=${LSHKNN_CXX_COMPILER} -DCMAKE_CUDA_COMPILER=${LSHKNN_CMAKE_CUDA_COMPILER}  ../
make -j
cd ../..
