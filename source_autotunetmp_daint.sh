# module swap craype-haswell craype-accel-nvidia60
module swap PrgEnv-cray PrgEnv-gnu
# required to compile lshknn
# module swap gcc gcc/6.2.0
module swap gcc gcc/7.3.0
module load daint-gpu
module load cudatoolkit/9.0.103_3.7-6.0.4.1_2.1__g72b395b
# module swap cudatoolkit cudatoolkit/9.0.103_3.7-6.0.4.1_2.1__g72b395b

# module load tools/boost/1.66.0
export CRAYPE_LINK_TYPE=dynamic

# to be sourced from shell to setup the environment
export BUILD_FOR_DAINT=
export HAZEL_BASE_PATH=/users/pfandedd

# export PATH=/home/pfandedd/git/AutoTuneTMP/gcc_install/bin:$PATH
export PATH=${HAZEL_BASE_PATH}/git/AutoTuneTMP/cmake/bin/:${HAZEL_BASE_PATH}/git/SGpp/scons/:$PATH
export LD_LIBRARY_PATH=${HAZEL_BASE_PATH}/git/OpenCL-ICD-Loader/build/lib:${HAZEL_BASE_PATH}/git/SGpp/lshknn/build:${HAZEL_BASE_PATH}/git/SGpp/lib/sgpp:${HAZEL_BASE_PATH}/git/AutoTuneTMP/boost_install/lib:$LD_LIBRARY_PATH:$LD_LIBRARY_PATH
# export SGPP_BASE_INCLUDE_DIR=${HAZEL_BASE_PATH}/SGPP_debug/base/src
# export AUTOTUNETMP_INCLUDE_DIR=${HAZEL_BASE_PATH}/AutoTuneTMP/AutoTuneTMP_install_debug/include
# export VC_INCLUDE_DIR=${HAZEL_BASE_PATH}/AutoTuneTMP/Vc_install/include
# export BOOST_INCLUDE_DIR=${HAZEL_BASE_PATH}/AutoTuneTMP/boost_install/include
# export OPENCL_ICD_DIR=${HAZEL_BASE_PATH}/intel_opencl/intel/opencl-1.2-6.4.0.37/etc/
export LSHKNN_CMAKE_CUDA_COMPILER=nvcc
export LSHKNN_CXX_COMPILER=g++
export OCL_LIBRARY_PATH=${HAZEL_BASE_PATH}/git/OpenCL-ICD-Loader/build/lib
# export OCL_LIBRARY_PATH=/opt/nvidia/cudatoolkit9.0/9.0.103_3.7-6.0.4.1_2.1__g72b395b/lib64/
export OCL_INCLUDE_PATH=${HAZEL_BASE_PATH}/git/OpenCL-Headers/
# export OCL_INCLUDE_PATH=/opt/nvidia/cudatoolkit9.0/9.0.103_3.7-6.0.4.1_2.1__g72b395b/include
