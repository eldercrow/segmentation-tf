# -*- coding: utf-8 -*-
# File: data.py

import cv2
import numpy as np
import copy
import itertools

from scipy.io import loadmat
from copy import deepcopy

from tensorpack.utils.argtools import memoized, log_once
from tensorpack.dataflow import (
    imgaug, TestDataSpeed, BatchData,
    MultiProcessMapDataZMQ, MultiThreadMapData, MapData,
    MapDataComponent, DataFromList)
from tensorpack.utils import logger
# import tensorpack.utils.viz as tpviz

from dataset.dataset_utils import load_many_from_db
from common import (
    SSDCropRandomShape, SSDResize, SSDColorJitter, CropPadTransform, DataFromListOfDict)
from config import config as cfg

# import matplotlib.pyplot as plt


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
    mean_label = cfg.PREPROC.LABEL_MEAN

    if cfg.DATA.NAME in ('cityscapes', 'camvid'):
        aspect_exp = 1.1
    elif cfg.DATA.NAME == 'cocostuff':
        aspect_exp = 1.1 #2.0
    else:
        logger.warn('Dataset name not known.')
        assert False

    aug = imgaug.AugmentorList([ \
            SSDCropRandomShape(cfg.PREPROC.INPUT_SHAPE_TRAIN, aspect_exp=aspect_exp, mean_rgbgr=mean_bgr),
            SSDResize(cfg.PREPROC.INPUT_SHAPE_TRAIN),
            imgaug.Flip(horiz=True),
            SSDColorJitter(mean_rgbgr=mean_bgr)
            ])
    aug_label = imgaug.AugmentorList([ \
            SSDCropRandomShape(cfg.PREPROC.INPUT_SHAPE_TRAIN, aspect_exp=aspect_exp, mean_rgbgr=[mean_label,]),
            SSDResize(cfg.PREPROC.INPUT_SHAPE_TRAIN, interp=cv2.INTER_NEAREST),
            imgaug.Flip(horiz=True)
            ])

    def preprocess(img):
        fn_img, fn_label = img['fn_img'], img['fn_label']
        # load head (and landmark) data as well
        im = cv2.imread(fn_img, cv2.IMREAD_COLOR)
        if fn_label.endswith('.mat'): # cocostuff
            label = loadmat(fn_label)['S'].astype(int)
            label = label.astype(np.uint8)
            # label = (label - 1).astype(np.uint8) # -1 becomes 255
        else:
            label = cv2.imread(fn_label, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        label = np.expand_dims(label, 2)
        assert (im is not None) and (label is not None), fn_img
        im = im.astype('float32')
        # label = label.astype('int32')
        # augmentation
        tfms = aug.get_transform(im)
        im = tfms.apply_image(im)
        # im, params = aug.augment_return_params(im)
        # TODO: better way to adjust label?
        tfms_label = deepcopy(tfms.tfms[:-1])
        tfms_label[0].mean_rgbgr = [255,]
        tfms_label[1]._transform.interp = cv2.INTER_NEAREST
        tfms_label = imgaug.TransformList(tfms_label)
        label = tfms_label.apply_image(label)
        # label = aug_label.augment_with_params(label, params_label)
        label = label.astype('int32')

        ret = [im, label]
        return ret

    if cfg.TRAINER == 'horovod':
        ds = MultiThreadMapData(ds, 5, preprocess)
        # MPI does not like fork()
    else:
        if cfg.PREPROC.NUM_WORKERS == 1:
            ds = MapData(ds, preprocess) # for debugging
        else:
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

    hh, ww = cfg.PREPROC.INPUT_SHAPE_EVAL
    mean_bgr = np.array(cfg.PREPROC.PIXEL_MEAN[::-1])
    aug = CropPadTransform(0, 0, ww, hh, mean_bgr)
    def f(fname):
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        assert im is not None, fname
        scale = min(ww / float(im.shape[1]), hh / float(im.shape[0]))
        im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
        im = aug.apply_image(im)
        im = cv2.resize(im, (ww, hh))
        return im
    ds = MapDataComponent(ds, f, 0)
    ds = BatchData(ds, batch_size, use_list=False)
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
