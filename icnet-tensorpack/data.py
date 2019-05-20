# -*- coding: utf-8 -*-
# File: data.py

import cv2
import numpy as np
import copy
import itertools

from copy import deepcopy

from tensorpack.utils.argtools import memoized, log_once
from tensorpack.dataflow import (
    imgaug, TestDataSpeed, BatchData,
    PrefetchDataZMQ, MultiProcessMapDataZMQ, MultiThreadMapData, MapData,
    MapDataComponent, DataFromList)
from tensorpack.utils import logger
# import tensorpack.utils.viz as tpviz

from dataset.dataset_utils import load_many_from_db
from common import (
    SSDCropRandomShape, SSDResize, SSDColorJitter, DataFromListOfDict)
from config import config as cfg

import matplotlib.pyplot as plt


class MalformedData(BaseException):
    pass


def get_train_dataflow():
    """
    Return a training dataflow. Each datapoint consists of the following:

    input image: (h, w, 3),
    semantic label image: (h, w, 1)
    """
    # imgs is a list, where each element is a dict containing 'fn_img', and 'fn_label'
    imgs = load_many_from_db(cfg.DATA.NAME, add_gt=True, is_train=True)
    # imgs = COCODetection.load_many(
    #     cfg.DATA.BASEDIR, cfg.DATA.TRAIN, add_gt=True, add_mask=cfg.MODE_MASK)
    """
    To train on your own data, change this to your loader.
    Produce "imgs" as a list of dict, in the dict the following keys are needed for training:
    height, width: integer
    file_name: str
    boxes: kx4 floats
    class: k integers
    difficult: k booleans. Use k False if you don't know what it means.
    segmentation: k lists of numpy arrays (one for each box).
        Each list of numpy array corresponds to the mask for one instance.
        Each numpy array in the list is a polygon of shape Nx2,
        because one mask can be represented by N polygons.

        If your segmentation annotations are originally masks rather than polygons,
        either convert it, or the augmentation code below will need to be
        changed or skipped accordingly.
    """

    # Valid training images should have at least one fg box.
    # But this filter shall not be applied for testing.
    num = len(imgs)
    # log invalid training

    ds = DataFromList(imgs, shuffle=True)

    mean_bgr = np.array(cfg.PREPROC.PIXEL_MEAN[::-1])

    aug = imgaug.AugmentorList([ \
            SSDCropRandomShape(mean_rgbgr=mean_bgr),
            SSDResize(cfg.PREPROC.INPUT_SHAPE_TRAIN),
            imgaug.Flip(horiz=True),
            SSDColorJitter(mean_rgbgr=mean_bgr)
            ])
    aug_label = imgaug.AugmentorList([ \
            SSDCropRandomShape(mean_rgbgr=[255,]),
            SSDResize(cfg.PREPROC.INPUT_SHAPE_TRAIN, interp=cv2.INTER_NEAREST),
            imgaug.Flip(horiz=True)
            ])

    # idx_aug_flip = 3

    def preprocess(img):
        fn_img, fn_label = img['fn_img'], img['fn_label']
        # load head (and landmark) data as well
        im = cv2.imread(fn_img, cv2.IMREAD_COLOR)
        label = cv2.imread(fn_label, cv2.IMREAD_GRAYSCALE)
        label = np.expand_dims(label, 2)
        assert (im is not None) and (label is not None), fn_img
        im = im.astype('float32')
        # label = label.astype('int32')
        # augmentation
        im, params = aug.augment_return_params(im)
        # TODO: adjust params
        params_label = deepcopy(params[:-1])
        params_label[0].mean_rgbgr = [255,]
        params_label[1].interp = cv2.INTER_NEAREST
        label = aug_label.augment_with_params(label, params_label)
        label = label.astype('int32')

        ret = [im, label]
        return ret

    if cfg.TRAINER == 'horovod':
        ds = MultiThreadMapData(ds, 5, preprocess)
        # MPI does not like fork()
    else:
        # ds = MapData(ds, preprocess) # for debugging
        ds = MultiProcessMapDataZMQ(ds, cfg.PREPROC.NUM_WORKERS, preprocess)
    ds = BatchData(ds, cfg.PREPROC.BATCH_SIZE)
    return ds


def get_eval_dataflow(batch_size=0, shard=0, num_shards=1):
    '''
    '''
    imgs = load_many_from_db(cfg.DATA.NAME, add_gt=True, is_train=False)

    if num_shards > 1:
        num_imgs = len(imgs)
        img_per_shard = num_imgs // num_shards
        s, e = shard * img_per_shard, min(num_imgs, (shard + 1) * img_per_shard)
        imgs = imgs[s:e]

    assert len(imgs) % batch_size == 0, \
            'len(img) must be multiples of batch_size, {}, {}'.format(len(imgs), batch_size)
    # imgs = COCODetection.load_many(cfg.DATA.BASEDIR, cfg.DATA.VAL, add_gt=False)
    # no filter for training
    # ds = DataFromList(imgs, shuffle=False)
    ds = DataFromListOfDict(imgs, ['fn_img', 'id'])

    if batch_size <= 0:
        batch_size = cfg.PREPROC.EVAL_BATCH_SIZE
    assert batch_size > 0, 'Batch size should be greater than 0'

    def f(fname):
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        assert im is not None, fname
        im = cv2.resize(im, (cfg.PREPROC.INPUT_SHAPE_EVAL[1], cfg.PREPROC.INPUT_SHAPE_EVAL[0]))
        return im
    ds = MapDataComponent(ds, f, 0)
    # def f(img):
    #     fn_img, fn_label, idx = img['fn_img'], img['fn_label'], img['id']
    #     # load head (and landmark) data as well
    #     im = cv2.imread(fn_img, cv2.IMREAD_COLOR)
    #     label = cv2.imread(fn_label)
    #     assert (im is not None) and (label is not None), fn_img
    #     im = im.astype('float32')
    #     label = label.astype('int32')
    #     return [im, label, idx]
    # ds = MapData(ds, f)
    ds = BatchData(ds, batch_size, use_list=True)
    return ds


if __name__ == '__main__':
    import os
    from tensorpack.dataflow import PrintData
    cfg.DATA.BASEDIR = os.path.expanduser('~/data/coco')
    ds = get_train_dataflow()
    ds = PrintData(ds, 100)
    TestDataSpeed(ds, 50000).start()
    ds.reset_state()
    for k in ds.get_data():
        pass
