#/bin/bash
export CUDA_VISIBLE_DEVICES='1'

./train.py \
    --logdir 'train_log/cityscapes_ssdnet' \
    --config \
    PREPROC.BATCH_SIZE=12 \
    PREPROC.NUM_WORKERS=2 \
    DATA.NAME='cityscapes' \
    TRAIN.NUM_EPOCH_PARTITIONS=1 \
    TRAIN.MAX_LR=1e-02 \
    TRAIN.MIN_LR=5e-05 \
    TRAIN.NUM_CYCLES=2 \
    BACKBONE.WEIGHTS='./pretrained/ssdnetv2_imagenet.npz' \
    # --load './train_log/cityscapes_ssdnet/checkpoint' \
    # --resume
