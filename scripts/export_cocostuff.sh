#!/bin/bash
export CUDA_VISIBLE_DEVICES=''

logdir="cocostuff_ws55"

TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

checkpoint_path="./train_log/"$logdir"/"
output_path=$checkpoint_path"pb"
prefix=$(head -1 $checkpoint_path"checkpoint" | grep -oh "model-[0-9]*")

fn_ig=$output_path"/infernce_graph.pb"
fn_fg=$output_path"/frozen_graph.pb"
fn_ir=$output_path"/icnet_IR"

python predict.py \
    --export-graph $fn_ig \
    --load $checkpoint_path"checkpoint" \
    --config \
    DATA.NAME='cocostuff' \
    BACKBONE.FILTER_SCALE=2.0 \
    INFERENCE.SHIFT_PREDICTION=1

python $TF_LIB/python/tools/freeze_graph.py \
  --input_graph $fn_ig \
  --input_binary \
  --input_checkpoint $checkpoint_path$prefix \
  --output_node_names 'segmentation_output' \
  --output_graph $fn_fg
#
# python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py \
#     --input_model $fn_fg \
#     --model_name $fn_ir \
#     --data_type FP32 \
#     --log_level INFO \
#     --input 'data' \
#     --input_shape '[1, 1024, 2048, 3]' \
#     --output 'segmentation_output' \
#     --extensions '/opt/intel/openvino/deployment_tools/model_optimizer/extensions' \

# python /opt/intel/openvino/deployment_tools/tools/calibration_tool/calibrate.py \
#   -c ./openvino/icnet-cityscapes-calibration.yml \
#   -m $output_path \
#   -ic 10 \
#   --threshold 3.0
#
# python /opt/intel/openvino/deployment_tools/tools/accuracy_checker_tool/accuracy_check.py \
#   -c ./openvino/icnet-cityscapes.yml \
#   -m $output_path \
#
