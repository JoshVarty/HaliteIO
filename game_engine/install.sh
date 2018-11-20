#!/usr/bin/env bash

set -e

cmake -DCMAKE_PREFIX_PATH=/home/josh/git/HaliteIO/game_engine/libtorch -DCUDA_HOST_COMPILER=/usr/bin/gcc-8 
make
