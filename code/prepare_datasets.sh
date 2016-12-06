#!/usr/bin/env bash
ROOT_DATA_DIR=/mnt/data
cd $ROOT_DATA_DIR
unzip train_all.zip
unzip test_all.zip
mv  train_all train
mv test_all test
awk -F" " '{split($1,a,"/");print a[4]" "$2}' trainingset_all.txt > train_data.txt
awk -F" " '{split($1,a,"/");print a[4]" "$2}' testset_all.txt > test_data.txt
