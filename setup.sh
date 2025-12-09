#!/bin/bash
BUILD_DIR=$(pwd)/build
rm -rf $BUILD_DIR
mkdir $BUILD_DIR
cmake -B $BUILD_DIR -G Ninja -DCMAKE_BUILD_TYPE=Release
