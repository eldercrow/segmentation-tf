#/bin/bash
export CUDA_VISIBLE_DEVICES='0,1'
./predict.py \
    --evaluate ./eval_log/cityscapes_val.npz \
    --load ./train_log/cityscapes_ssdnet/checkpoint \
    --config \
    DATA.NAME='cityscapes' \
    TRAIN.NUM_GPUS=1
    # --evalfromjson
    # to rerun mAP evaluation from previously evaluated results, uncomment the above line.
