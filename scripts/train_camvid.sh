#/bin/bash
export CUDA_VISIBLE_DEVICES='0,1'

./train.py \
    --logdir './train_log/camvid_ws00' \
    --config \
    PREPROC.BATCH_SIZE=12 \
    PREPROC.NUM_WORKERS=6 \
    DATA.NAME='camvid' \
    DATA.COCOSTUFF.BASEDIR='~/dataset/CamVid' \
    TRAIN.NUM_EPOCH_PARTITIONS=1 \
    TRAIN.MAX_LR=1e-02 \
    TRAIN.MIN_LR=5e-05 \
    TRAIN.WEIGHT_DECAY=1e-04 \
    TRAIN.EPOCHS_PER_CYCLE=240 \
    TRAIN.NUM_CYCLES=1 \
    BACKBONE.WEIGHTS='./pretrained/icnet_cocostuff_nocls.npz' \
    BACKBONE.FILTER_SCALE=2.0 \
    APPLY_PRUNING=False \
    # PRUNING.TARGET_SPARSITY=0.31 \
    # --resume
    # --load './temp.npz' \
    # --resume
