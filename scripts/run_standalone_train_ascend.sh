#!/bin/bash

if [ $# != 3 ]
then
    echo "Usage: bash run_standalone_train_ascend.sh [DEVICE_ID] [DATA_URL] [TRAIN_URL]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

ID=$1
echo $ID
PATH1=$(get_real_path $2)
echo $PATH1
PATH2=$(get_real_path $3)
echo $PATH2

if [ ! -d $PATH1 ]
then
    echo "error: DATA_URL=$PATH1 is not a directory"
exit 1
fi

if [ ! -d $PATH2 ]
then
    echo "error: TRAIN_URL=$PATH2 is not a directory"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=$ID
export RANK_ID=0
export RANK_SIZE=1

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp ../*.py ./train
cp *.sh ./train
cp -r ../src ./train
cd ./train || exit
echo "start training for device $ID"
env > env.log
nohup python -u train.py --device_target=Ascend --device_id=$ID --data_url=$PATH1 --train_url=$PATH2 > train_log 2>&1 &
cd ..
