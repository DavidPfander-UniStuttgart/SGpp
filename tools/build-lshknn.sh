#!/bin/bash
set -x

# - to be run from project root directory
# - only source proper sourceme-*.sh files (e.g. for the argon machines)
# - make sure gcc <=6 is loaded
# - make sure cuda 9.0 (or possibly newer) is available

if [ ! -d "lshknn" ]; then
    git clone git@gitlab-sgs.informatik.uni-stuttgart.de:breyerml/HighPerformanceApproxKNNAlgorithm.git lshknn
    cd lshknn
    git checkout build_shared_library
else
    cd lshknn
    git pull
fi

./build-cmake.sh
export PATH=$PWD/cmake/bin:$PATH
echo "cmake version: `cmake --version`"
mkdir -p build
cd build
cmake -DCMAKE_CXX_COMPILER=${LSHKNN_CXX_COMPILER} -DCMAKE_CUDA_HOST_COMPILER=${LSHKNN_CXX_COMPILER} -DCMAKE_CUDA_COMPILER=${LSHKNN_CMAKE_CUDA_COMPILER} -DCMAKE_BUILD_TYPE=RelWithDebInfo -DWITH_CUDA=${LSHKNN_WITH_CUDA} -DWITH_OPENCL=${LSHKNN_WITH_OPENCL}  ../
make -j VERBOSE=1
cd ../..
