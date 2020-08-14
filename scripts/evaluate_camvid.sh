#/bin/bash
export CUDA_VISIBLE_DEVICES='0,1'
./predict.py \
    --evaluate ./eval_log/camvid_val.npz \
    --flops \
    --load ./train_log/camvid_ws00/checkpoint \
    --config \
    DATA.NAME='camvid' \
    DATA.COCOSTUFF.BASEDIR='~/dataset/CamVid' \
    BACKBONE.FILTER_SCALE=2.0 \
    PREPROC.EVAL_BATCH_SIZE=5 \
    TRAIN.NUM_GPUS=1 \
    # --evalfromjson
    # to rerun mAP evaluation from previously evaluated results, uncomment the above line.

