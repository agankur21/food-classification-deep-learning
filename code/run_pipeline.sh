#!/usr/bin/env bash
##############################
ROOT_DATA_DIR=/mnt/data
TRAIN_DATA_DIR=${ROOT_DATA_DIR}/train
TEST_DATA_DIR=${ROOT_DATA_DIR}/test
CAFFE_TOOLS_DIR=/home/ubuntu/caffe/build/tools
CAFFE_MODELS_DIR=/home/ubuntu/caffe/models
###############################

#Creating LMDB files:
$CAFFE_TOOLS_DIR/convert_imageset -resize_height=227 -resize_width=227 -shuffle=true ${TEST_DATA_DIR}/ ${ROOT_DATA_DIR}/test_data.txt ${ROOT_DATA_DIR}/test_lmdb
$CAFFE_TOOLS_DIR/convert_imageset -resize_height=227 -resize_width=227 -shuffle=true ${TRAIN_DATA_DIR}/ ${ROOT_DATA_DIR}/train_data.txt ${ROOT_DATA_DIR}/train_lmdb

#Compute Image mean :
${CAFFE_TOOLS_DIR}/compute_image_mean -backend=lmdb ${ROOT_DATA_DIR}/train_lmdb ${ROOT_DATA_DIR}/mean_all.binaryproto

#Train model
${CAFFE_TOOLS_DIR}/caffe train --solver ${CAFFE_MODELS_DIR}/bvlc_reference_caffenet/solver_spatial.prototxt --weights $CAFFE_MODELS_DIR/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel 2>&1 | tee ${ROOT_DATA_DIR}/spatial_cnn.log
