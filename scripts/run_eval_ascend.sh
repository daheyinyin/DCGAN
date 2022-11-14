#!/bin/bash

if [ $# != 3 ]
then
    echo "Usage: sh run_eval.sh [IMG_URL] [CKPT_URL] [DEVICE_ID]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
PATH2=$(get_real_path $2)

if [ ! -d $PATH1 ]
then
    echo "error: IMG_URL=$PATH1 is not a directory"
exit 1
fi

if [ ! -f $PATH2 ]
then
    echo "error: CKPT_URL=$PATH2 is not a file"
exit 1
fi

ulimit -c unlimited
export DEVICE_NUM=1
export RANK_SIZE=$DEVICE_NUM
export DEVICE_ID=$3
export RANK_ID=$DEVICE_ID


if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cp ../*.py ./eval
cp *.sh ./eval
cp -r ../src ./eval
cd ./eval || exit
env > env.log
echo "start evaluation for device $DEVICE_ID"
nohup python -u eval.py --device_id=$DEVICE_ID --img_url=$PATH1 --ckpt_url=$PATH2 --device_target=Ascend> eval_log 2>&1 &
cd ..

