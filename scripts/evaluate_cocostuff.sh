#/bin/bash
export CUDA_VISIBLE_DEVICES='0,1'
./predict.py \
    --evaluate ./eval_log/cocostuff_val.npz \
    --load ./train_log/cocostuff_ws55/checkpoint \
    --config \
    DATA.NAME='cocostuff' \
    DATA.COCOSTUFF.BASEDIR='~/dataset/cocostuff' \
    BACKBONE.FILTER_SCALE=2.0 \
    INFERENCE.SHIFT_PREDICTION=1 \
    TRAIN.NUM_GPUS=2 \
    # --evalfromjson
    # to rerun mAP evaluation from previously evaluated results, uncomment the above line.
