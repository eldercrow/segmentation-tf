#/bin/bash
export CUDA_VISIBLE_DEVICES='0,1'

./train.py \
    --logdir 'train_log/cityscapes_icnet' \
    --config \
    PREPROC.BATCH_SIZE=12 \
    PREPROC.NUM_WORKERS=6 \
    DATA.NAME='cityscapes' \
    DATA.CITYSCAPES.BASEDIR='~/dataset/cityscapes' \
    TRAIN.NUM_EPOCH_PARTITIONS=1 \
    TRAIN.MAX_LR=1e-03 \
    TRAIN.MIN_LR=1e-04 \
    TRAIN.WEIGHT_DECAY=1e-04 \
    TRAIN.NUM_CYCLES=1 \
    TRAIN.EPOCHS_PER_CYCLE=60 \
    BACKBONE.WEIGHTS='./pretrained/icnet_tensorpack.npz' \
    # --load './temp.npz' \
    # --resume
