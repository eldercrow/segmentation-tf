#/bin/bash
export CUDA_VISIBLE_DEVICES='0,1'
./predict.py \
    --evaluate ./eval_log/cocostuff_val.npz \
    --load ./train_log/cocostuff_icnet_r2/checkpoint \
    --config \
    DATA.NAME='cocostuff' \
    TRAIN.NUM_GPUS=2 \
    # --evalfromjson
    # to rerun mAP evaluation from previously evaluated results, uncomment the above line.
