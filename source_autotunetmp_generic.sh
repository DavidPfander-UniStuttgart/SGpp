#!/bin/bash
# only to be source'd!

if [[ -z ${ALL_CODE_SOURCE_ME_SOURCED} ]]; then
    source ../source-me.sh
fi

# to be sourced from shell to setup the environment
export LD_LIBRARY_PATH=${SGPP_REPO_PATH}/lib/sgpp:${LSHKNN_REPO_PATH}/build:${AUTOTUNETMP_REPO_PATH}/boost_install/lib:$LD_LIBRARY_PATH
# export AUTOTUNETMP_INCLUDE_DIR=${AUTOTUNETMP_REPO_PATH}/AutoTuneTMP_install/include
# export VC_INCLUDE_DIR=${AUTOTUNETMP_REPO_PATH}/Vc_install/include
# export BOOST_INCLUDE_DIR=${AUTOTUNETMP_REPO_PATH}/boost_install/include
