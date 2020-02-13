#/bin/bash
export CUDA_VISIBLE_DEVICES='0,1'
./predict.py \
    --evaluate ./eval_log/cityscapes_val.npz \
    --load ./train_log/cityscapes_icnet/checkpoint \
    --config \
    DATA.NAME='cityscapes' \
    DATA.CITYSCAPES.BASEDIR='~/dataset/cityscapes' \
    TRAIN.NUM_GPUS=2
    # --evalfromjson
    # to rerun mAP evaluation from previously evaluated results, uncomment the above line.
