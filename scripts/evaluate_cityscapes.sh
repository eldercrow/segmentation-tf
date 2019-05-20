#/bin/bash
export CUDA_VISIBLE_DEVICES='1'
./train.py \
    --evaluate ./eval_log/cityscapes_val.npz \
    --load ./train_log/cityscapes_ssdnet/checkpoint \
    --config \
    DATA.NAME='cityscapes' \
    # --evalfromjson
    # to rerun mAP evaluation from previously evaluated results, uncomment the above line.
