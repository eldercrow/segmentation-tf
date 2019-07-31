#/bin/bash
export CUDA_VISIBLE_DEVICES='0,1'

./train.py \
    --logdir 'train_log/cityscapes_icnet' \
    --config \
    PREPROC.BATCH_SIZE=12 \
    PREPROC.NUM_WORKERS=4 \
    DATA.NAME='cityscapes' \
    TRAIN.NUM_EPOCH_PARTITIONS=1 \
    TRAIN.MAX_LR=1e-02 \
    TRAIN.MIN_LR=1e-04 \
    TRAIN.WEIGHT_DECAY=1e-04 \
    TRAIN.NUM_CYCLES=2 \
    BACKBONE.WEIGHTS='./pretrained/test.npz' \
    # --load './temp.npz' \
    # --resume
