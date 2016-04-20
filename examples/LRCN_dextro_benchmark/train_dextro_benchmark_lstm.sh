#!/bin/bash

#TOOLS=../../build/tools
TOOLS=../../cmake_build/install/bin

export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH=.
export GLOG_logtostderr=1

  $TOOLS/caffe \
  train \
  -solver dextro_benchmark_lstm_solver.prototxt \
  -weights ../LRCN_activity_recognition/single_frame_all_layers_hyb_RGB_iter_5000.caffemodel \
  |& tee lstm_dextro_benchmark_2016_04_19.log

#  |& tee lstm_dextro_benchmark_2016_03_31.log

echo "Done."
