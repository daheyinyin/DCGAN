#!/bin/bash

# Parameter format
if [ $# -ne 2 ]
then
  echo "Wrong parameter format."
  echo "Usage:"
  echo "         bash $0 RESULT_PATH GEN_NUM"
  echo "Example:"
  echo "         bash $0 ./results 10"

  exit 255
fi

# Rebuild results folder
rm -r results
mkdir -p results

# Run main.py
RESULT_PATH=$1
GEN_NUM=$2
python3 main.py "${RESULT_PATH}" "${GEN_NUM}"