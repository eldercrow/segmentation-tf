#/bin/bash
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5'

./train.py \
    --logdir './train_log/cocostuff_ws30' \
    --config \
    PREPROC.BATCH_SIZE=12 \
    PREPROC.NUM_WORKERS=18 \
    DATA.NAME='cocostuff' \
    DATA.COCOSTUFF.BASEDIR='~/dataset/cocostuff' \
    TRAIN.NUM_EPOCH_PARTITIONS=1 \
    TRAIN.MAX_LR=1e-03 \
    TRAIN.MIN_LR=5e-05 \
    TRAIN.WEIGHT_DECAY=1e-04 \
    TRAIN.EPOCHS_PER_CYCLE=100 \
    TRAIN.NUM_CYCLES=1 \
    BACKBONE.WEIGHTS='./pretrained/icnet_cocostuff_ws00.npz' \
    BACKBONE.FILTER_SCALE=2.0 \
    APPLY_PRUNING=True \
    PRUNING.TARGET_SPARSITY=0.31 \
    # --resume
    # --load './temp.npz' \
    # --resume
