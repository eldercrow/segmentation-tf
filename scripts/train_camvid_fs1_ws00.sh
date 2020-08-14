#/bin/bash
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5'

name='camvid_fs1_ws00'
logdir='./train_log/'$name

./train.py \
    --logdir $logdir \
    --config \
    PREPROC.BATCH_SIZE=12 \
    PREPROC.NUM_WORKERS=18 \
    DATA.NAME='camvid' \
    DATA.COCOSTUFF.BASEDIR='~/dataset/CamVid' \
    TRAIN.NUM_EPOCH_PARTITIONS=1 \
    TRAIN.MAX_LR=1e-02 \
    TRAIN.MIN_LR=5e-05 \
    TRAIN.WEIGHT_DECAY=1e-04 \
    TRAIN.EPOCHS_PER_CYCLE=240 \
    TRAIN.NUM_CYCLES=1 \
    BACKBONE.WEIGHTS='./pretrained/icnet_cityscapes_nocls.npz' \
    BACKBONE.FILTER_SCALE=1.0 \
    APPLY_PRUNING=False \
#     # PRUNING.TARGET_SPARSITY=0.31 \
#     # --resume
#     # --load './temp.npz' \
#     # --resume

python scripts/dump_model_params.py \
  $logdir'/checkpoint' \
  $logdir'/'$name'.npz' \
