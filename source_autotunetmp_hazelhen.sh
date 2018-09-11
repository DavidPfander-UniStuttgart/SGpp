module swap PrgEnv-cray PrgEnv-gnu
module load tools/boost/1.66.0
export CRAYPE_LINK_TYPE=dynamic

# to be sourced from shell to setup the environment
export BUILD_FOR_HAZEL=
export HAZEL_BASE_PATH=/zhome/academic/HLRS/ipv/ipvpfand

# export PATH=/home/pfandedd/git/AutoTuneTMP/gcc_install/bin:$PATH
export PATH=${HAZEL_BASE_PATH}/cmake/bin/:${HAZEL_BASE_PATH}/scons/:$PATH
export LD_LIBRARY_PATH=${HAZEL_BASE_PATH}/git/OpenCL-ICD-Loader/build/lib:${HAZEL_BASE_PATH}/git/SGpp/lshknn/build:${HAZEL_BASE_PATH}/git/SGpp/lib/sgpp:$LD_LIBRARY_PATH #${HAZEL_BASE_PATH}/AutoTuneTMP/boost_install/lib:$LD_LIBRARY_PATH
# export SGPP_BASE_INCLUDE_DIR=${HAZEL_BASE_PATH}/SGPP_debug/base/src
# export AUTOTUNETMP_INCLUDE_DIR=${HAZEL_BASE_PATH}/AutoTuneTMP/AutoTuneTMP_install_debug/include
# export VC_INCLUDE_DIR=${HAZEL_BASE_PATH}/AutoTuneTMP/Vc_install/include
# export BOOST_INCLUDE_DIR=${HAZEL_BASE_PATH}/AutoTuneTMP/boost_install/include
export OPENCL_ICD_DIR=${HAZEL_BASE_PATH}/intel_opencl/intel/opencl-1.2-6.4.0.37/etc/
# export LSHKNN_CMAKE_CUDA_COMPILER=/usr/local.nfs/sw/cuda/cuda-9.0/bin/nvcc
export LSHKNN_CXX_COMPILER=g++
export OCL_LIBRARY_PATH=${HAZEL_BASE_PATH}/git/OpenCL-ICD-Loader/build/lib
export OCL_INCLUDE_PATH=${HAZEL_BASE_PATH}/git/OpenCL-ICD-Loader/inc
