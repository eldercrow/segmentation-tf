#/bin/bash
export CUDA_VISIBLE_DEVICES='2,3,4,5'

name='camvid_ws60'
logdir='./train_log/'$name

# ./train.py \
#     --logdir $logdir \
#     --config \
#     PREPROC.BATCH_SIZE=12 \
#     PREPROC.NUM_WORKERS=12 \
#     DATA.NAME='camvid' \
#     DATA.COCOSTUFF.BASEDIR='~/dataset/CamVid' \
#     TRAIN.NUM_EPOCH_PARTITIONS=1 \
#     TRAIN.MAX_LR=1e-03 \
#     TRAIN.MIN_LR=5e-05 \
#     TRAIN.WEIGHT_DECAY=1e-04 \
#     TRAIN.EPOCHS_PER_CYCLE=120 \
#     TRAIN.NUM_CYCLES=1 \
#     BACKBONE.WEIGHTS='./train_log/camvid_ws00/camvid_ws00.npz' \
#     BACKBONE.FILTER_SCALE=2.0 \
#     APPLY_PRUNING=1 \
#     PRUNING.TARGET_SPARSITY=0.61 \
#
# python scripts/dump_model_params.py \
#   $logdir'/checkpoint' \
#   $logdir'/'$name'.npz' \

logdir_merged=$logdir'_merged'

./train.py \
    --logdir $logdir_merged \
    --savefirst \
    --config \
    PREPROC.BATCH_SIZE=12 \
    PREPROC.NUM_WORKERS=18 \
    DATA.NAME='camvid' \
    DATA.COCOSTUFF.BASEDIR='~/dataset/CamVid' \
    TRAIN.NUM_EPOCH_PARTITIONS=1 \
    TRAIN.MAX_LR=1e-02 \
    TRAIN.MIN_LR=5e-05 \
    TRAIN.WEIGHT_DECAY=1e-04 \
    TRAIN.EPOCHS_PER_CYCLE=100 \
    TRAIN.NUM_CYCLES=1 \
    BACKBONE.WEIGHTS=$logdir'/'$name'.npz' \
    BACKBONE.FILTER_SCALE=2.0
