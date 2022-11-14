#!/bin/bash

# Parameter format
if [ $# -ne 3 ]
then
  echo "Wrong parameter format."
  echo "Usage:"
  echo "         bash $0 OM_PATH RESULT_PATH GEN_NUM"
  echo "Example:"
  echo "         bash $0 ../data/models/DCGAN.om ./results 10"

  exit 255
fi


# Rebuild build folder
rm core
rm -r build
mkdir -p build
# Enter build floder
cd build || exit

# Cmake & make
if ! cmake ..;
then
  echo "[ERROR] Cmake failed."
  exit
fi
if ! (make);
then
  echo "[ERROR] Make failed."
  exit
fi
echo "[INFO] Build successfully."

# Enter previous floder
cd - || exit
# Rebuild results folder
rm -r results
mkdir -p results

# run
OM_PATH=$1
RESULT_PATH=$2
GEN_NUM=$3
./build/DCGAN "${OM_PATH}" "${RESULT_PATH}" "${GEN_NUM}"