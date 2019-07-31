# -*- coding: utf-8 -*-
# File: basemodel.py

# import argparse
# import cv2
# import os
import numpy as np
import tensorflow as tf

from contextlib import contextmanager

from tensorpack import *
from tensorpack.dataflow import imgaug
from tensorpack.tfutils import argscope, get_model_loader, model_utils
from tensorpack.tfutils.argscope import argscope #, get_arg_scope
# from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
# from tensorpack.tfutils.varreplace import custom_getter_scope
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.utils import logger
from tensorpack.models import (
    Conv2D, Deconv2D, MaxPooling, BatchNorm, BNReLU, LinearWrap, GlobalAvgPooling)
from tensorpack.models.regularize import Dropout


mom = tf.get_variable('bn_momentum',
                      (),
                      dtype=tf.float32,
                      trainable=False,
                      initializer=tf.constant_initializer(0.9))


@contextmanager
def ssdnet_argscope():
    with argscope([Conv2D, MaxPooling, BatchNorm, GlobalAvgPooling], data_format='NHWC'), \
            argscope([Conv2D, FullyConnected], use_bias=False), \
            argscope([BatchNorm], momentum=mom): #get_bn_momentum()):
        yield


@layer_register(log_shape=True)
def AccuracyBoost(x):
    '''
    Accuracy boost block for bottleneck layers.
    '''
    nch = x.get_shape().as_list()[-1]
    g = GlobalAvgPooling('gpool', x)
    g = tf.reshape(g, [-1, 1, 1, nch])
    # g = DWConv(g, 1, active=None)
    wp = tf.nn.sigmoid(BatchNorm('p/bn', g))
    wn = tf.nn.sigmoid(BatchNorm('n/bn', -g))
    return tf.multiply(x, wp + wn, name='res')


@layer_register(log_shape=True)
def DWConv(x, kernel, padding='SAME', stride=1, w_init=None, active=True, data_format='NHWC'):
    '''
    Depthwise conv + BN + (optional) ReLU.
    We do not use channel multiplier here (fixed as 1).
    '''
    assert data_format in ('NHWC', 'channels_last')
    channel = x.get_shape().as_list()[-1]
    if not isinstance(kernel, (list, tuple)):
        kernel = [kernel, kernel]
    filter_shape = [kernel[0], kernel[1], channel, 1]

    if w_init is None:
        w_init = tf.variance_scaling_initializer(2.0)
    W = tf.get_variable('W', filter_shape, initializer=w_init)
    out = tf.nn.depthwise_conv2d(x, W, [1, stride, stride, 1], padding=padding, data_format=data_format)

    if active is None:
        return out

    out = BNReLU(out) if active else BatchNorm('bn', out)
    return out


@layer_register(log_shape=True)
def LinearBottleneck(x, ich, och, kernel,
                     padding='SAME',
                     stride=1,
                     active=None,
                     t=3,
                     use_ab=False,
                     w_init=None):
    '''
    mobilenetv2 linear bottlenet.
    '''
    if active is None:
        active = True if kernel > 3 else False

    out = Conv2D('conv_e', x, int(ich*t), 1, activation=BNReLU)
    if use_ab:
        out = AccuracyBoost('ab', out)
    out = DWConv('conv_d', out, kernel, padding, stride, w_init, active)
    out = Conv2D('conv_p', out, och, 1, activation=None)
    with tf.variable_scope('conv_p'):
        out = BatchNorm('bn', out)
    return out


@layer_register(log_shape=True)
def DownsampleBottleneck(x, ich, och, kernel,
                         padding='SAME',
                         stride=2,
                         active=None,
                         t=3,
                         use_ab=False,
                         w_init=None):
    '''
    downsample linear bottlenet.
    '''
    if active is None:
        active = True if kernel > 3 else False

    out_e = Conv2D('conv_e', x, ich*t, 1, activation=BNReLU)
    if use_ab:
        out_e = AccuracyBoost('ab', out_e)
    out_d = DWConv('conv_d', out_e, kernel, padding, stride, w_init, active)
    out_m = DWConv('conv_m', out_e, kernel, padding, stride, w_init, active)
    out = tf.concat([out_d, out_m], axis=-1)
    out = Conv2D('conv_p', out, och, 1, activation=None)
    with tf.variable_scope('conv_p'):
        out = BatchNorm('bn', out)
    return out


@layer_register(log_shape=True)
def inception(x, ch, k, stride, t=3, swap_block=False, active=None, use_ab=False):
    '''
    ssdnet inception layer.
    '''
    ich = ch // 2
    if stride == 1:
        oi = LinearBottleneck('conv1', x, ch, ch, k, stride=stride, t=t, active=active, use_ab=use_ab)
    else:
        oi = DownsampleBottleneck('conv1', x, ich, ch, 4, stride=stride, t=t, active=active, use_ab=use_ab)
    oi = tf.split(oi, 2, axis=-1)
    o1 = oi[0]
    o2 = oi[1] + LinearBottleneck('conv2', oi[1], ich, ich, k, t=t, active=active, use_ab=use_ab)

    if not swap_block:
        out = tf.concat([o1, o2], -1)
    else:
        out = tf.concat([o2, o1], -1)
    if stride == 1:
        out = tf.add(out, x)
    return out


def ssdnet_backbone(image, **kwargs):
    #
    with ssdnet_argscope():
        l = image #tf.transpose(image, perm=[0, 2, 3, 1])
        with argscope([BatchNorm], training=False):
            # conv1
            l = Conv2D('conv1', l, 24, 4, strides=2, activation=None, padding='SAME')
            with tf.variable_scope('conv1'):
                l = BNReLU(tf.concat([l, -l], axis=-1))
            l = MaxPooling('pool1', l, 2)

        l = tf.stop_gradient(l)

        with argscope([BatchNorm], training=None):
            # conv2
            l = LinearBottleneck('conv2', l, 48, 24, 3, t=1, use_ab=True)
            l = l + LinearBottleneck('conv3', l, 24, 24, 5, t=2, use_ab=True)

            ch_all = [48, 72, 96]
            iters = [2, 4, 4]
            mults = [3, 4, 6]

            hlist = []
            for ii, (ch, it, mu) in enumerate(zip(ch_all, iters, mults)):
                use_ab = (ii < 2)
                for jj in range(it):
                    name = 'inc{}/{}'.format(ii, jj)
                    stride = 2 if jj == 0 else 1
                    k = 3 if jj < (it // 2) else 5
                    swap_block = True if jj % 2 == 1 else False
                    l = inception(name, l, ch, k, stride, t=mu, swap_block=swap_block, use_ab=use_ab)
                hlist.append(l)

        return hlist
