#/bin/bash
export CUDA_VISIBLE_DEVICES='0,1'

./train.py \
    --logdir 'train_log/cocostuff_icnet_r2' \
    --config \
    PREPROC.BATCH_SIZE=16 \
    PREPROC.NUM_WORKERS=6 \
    DATA.NAME='cocostuff' \
    TRAIN.NUM_EPOCH_PARTITIONS=1 \
    TRAIN.MAX_LR=1e-02 \
    TRAIN.MIN_LR=5e-05 \
    TRAIN.WEIGHT_DECAY=1e-04 \
    TRAIN.EPOCHS_PER_CYCLE=240 \
    TRAIN.NUM_CYCLES=1 \
    BACKBONE.WEIGHTS='./pretrained/icnet_cocostuff_r1.npz' \
    # --resume
    # --load './temp.npz' \
    # --resume
