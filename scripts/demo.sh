#/bin/bash
export CUDA_VISIBLE_DEVICES=''
./predict.py \
    --predict '~/github/ICNet-tensorflow/data/input/cityscapes1.png' \
    --load './train_log/cityscapes_ssdnet/checkpoint' \
    --config \
    DATA.NAME='cityscapes'

# ./train.py \
#     --predict './data/demo/Low Crawl Demo.mp4' \
#     --load './train_log/pvtdbpsroir3/checkpoint' \
#     --pred-video \
#     --config \
#     MODE_MASK=False MODE_FPN=False MODE_LANDMARK=True OPENVINO=False \
#     DATA.NAME='pvtdb'
