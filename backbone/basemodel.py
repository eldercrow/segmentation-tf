# -*- coding: utf-8 -*-
# File: basemodel.py

from contextlib import contextmanager
import tensorflow as tf
import numpy as np
from tensorpack.tfutils.argscope import argscope
from tensorpack.models import BatchNorm

from config import config as cfg


def maybe_freeze_affine(getter, *args, **kwargs):
    # custom getter to freeze affine params inside bn
    name = args[0] if len(args) else kwargs.get('name')
    if name.endswith('/gamma') or name.endswith('/beta'):
        if cfg.BACKBONE.FREEZE_AFFINE:
            kwargs['trainable'] = False
    return getter(*args, **kwargs)


def maybe_reverse_pad(topleft, bottomright):
    if cfg.BACKBONE.TF_PAD_MODE:
        return [topleft, bottomright]
    return [bottomright, topleft]


@contextmanager
def maybe_syncbn_scope():
    if cfg.BACKBONE.NORM == 'SyncBN':
        with argscope(BatchNorm, training=None, sync_statistics='nccl'):
            yield
    else:
        yield


def image_preprocess(image, bgr=True):
    with tf.name_scope('image_preprocess'):
        if image.dtype.base_dtype != tf.float32:
            image = tf.cast(image, tf.float32)

        mean = np.array(cfg.PREPROC.PIXEL_MEAN, dtype=np.float32)
        std = np.array(cfg.PREPROC.PIXEL_STD, dtype=np.float32)
        if bgr:
            mean = mean[::-1]
            std = std[::-1]
        # for openvino fuse op
        image_std = tf.expand_dims(tf.constant(std, dtype=tf.float32), 0)
        image_mean = tf.expand_dims(tf.constant(mean / std, dtype=tf.float32), 0)
        image = (image / image_std) - image_mean
        return image


def get_bn(zero_init=False):
    if zero_init:
        return lambda x, name=None: BatchNorm('bn', x, gamma_init=tf.zeros_initializer())
    else:
        return lambda x, name=None: BatchNorm('bn', x)
