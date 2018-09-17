#!/bin/bash
set -x

# - to be run from project root directory
# - only source proper sourceme-*.sh files (e.g. for the argon machines)
# - make sure gcc <=6 is loaded
# - make sure cuda 9.0 (or possibly newer) is available

if [ ! -d "scons" ]; then
    mkdir scons
    cd scons
    wget http://prdownloads.sourceforge.net/scons/scons-local-3.0.1.tar.gz
    tar xf scons-local-3.0.1.tar.gz
    cd ..
fi
