#!/bin/bash
# only to be source'd!

if [[ -z ${ALL_CODE_SOURCE_ME_SOURCED} ]]; then
    source ../source-me.sh
fi


if [[ -z ${SGPP_SOURCE_ME_SOURCED} ]]; then
    # to be sourced from shell to setup the environment
    export LD_LIBRARY_PATH=${SGPP_REPO_PATH}/lib/sgpp:${LSHKNN_REPO_PATH}/build:${AUTOTUNETMP_REPO_PATH}/boost_install/lib:$LD_LIBRARY_PATH

    # extra paths for SGpp's AutoTuneTMP experiments (needed for JIT compilation)
    export SGPP_BASE_INCLUDE_DIR=${SGPP_REPO_PATH}/base/src
    export BOOST_INCLUDE_DIR=${BOOST_REPO_PATH}/include
    export AUTOTUNETMP_INCLUDE_DIR=${AUTOTUNETMP_REPO_PATH}/AutoTuneTMP_install/include
    export VC_INCLUDE_DIR=${AUTOTUNETMP_REPO_PATH}/Vc_install/include
    export SGPP_SOURCE_ME_SOURCED=1
fi
