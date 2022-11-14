#!/bin/bash

# Parameter format
if [ $# -ne 2 ]
then
  echo "Wrong parameter format."
  echo "Usage:"
  echo "         bash $0 INPUT_AIR_PATH OUTPUT_OM_PATH_NAME"
  echo "Example:"
  echo "         bash $0 ./dcgan_16_20220106.air ../models/DCGAN"

  exit 255
fi

# DCGAN model from .air to .om
AIR_PATH=$1
OM_PATH=$2
atc --input_format=NCHW \
--framework=1 \
--model="${AIR_PATH}" \
--output="${OM_PATH}" \
--soc_version=Ascend310

# Delete unnecessary files
rm fusion_result.json
rm -r kernel_meta/

# Modify file permissions
chmod +r+w "${OM_PATH}.om"
