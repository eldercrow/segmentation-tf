# -*- coding: utf-8 -*-
# File: basemodel.py
import numpy as np
import tensorflow as tf
from contextlib import contextmanager

from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope, under_variable_scope
from tensorpack.tfutils.varreplace import custom_getter_scope
from tensorpack.models import (
    Conv2D, FullyConnected, MaxPooling, BatchNorm, BNReLU, LinearWrap)
from tensorpack import layer_register


@auto_reuse_variable_scope
def get_bn_momentum():
    mom = tf.get_variable('bn_momentum',
                          (),
                          dtype=tf.float32,
                          trainable=False,
                          initializer=tf.constant_initializer(0.9))
    tf.summary.scalar('bn_momentum-summary', mom)
    return mom


@contextmanager
def ssdnet_argscope():
    with argscope([Conv2D, MaxPooling, BatchNorm, DWConv], data_format='NHWC'), \
            argscope([Conv2D, FullyConnected], use_bias=False), \
            argscope([BatchNorm], momentum=get_bn_momentum(), training=None):
        yield


@layer_register(log_shape=True)
def DWConv(x, kernel, padding='SAME', stride=1, w_init=None, active=True, dilate=1, data_format='NHWC'):
    '''
    Depthwise conv + BN + (optional) ReLU.
    We do not use channel multiplier here (fixed as 1).
    '''
    assert data_format in ('NHWC', 'channels_last')
    channel = x.get_shape().as_list()[-1]
    if not isinstance(kernel, (list, tuple)):
        kernel = [kernel, kernel]
    if not isinstance(dilate, (list, tuple)):
        dilate = [dilate, dilate]
    filter_shape = [kernel[0], kernel[1], channel, 1]

    if w_init is None:
        w_init = tf.variance_scaling_initializer(2.0)
    W = tf.get_variable('W', filter_shape, initializer=w_init)
    out = tf.nn.depthwise_conv2d(x, W,
                                 [1, stride, stride, 1],
                                 padding=padding,
                                 rate=dilate,
                                 data_format=data_format)

    if active is None:
        return out

    out = BNReLU('bn', out) if active else BatchNorm('bn', out)
    return out


@layer_register(log_shape=True)
def LinearBottleneck(x, ich, och, kernel,
                     padding='SAME',
                     stride=1,
                     active=None,
                     t=3,
                     w_init=None):
    '''
    mobilenetv2 linear bottlenet.
    '''
    if active is None:
        active = True if kernel > 3 else False

    out = Conv2D('conv_e', x, int(ich*t), 1, activation=BNReLU)
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
                         w_init=None):
    '''
    downsample linear bottlenet.
    '''
    if active is None:
        active = True if kernel > 3 else False

    out_e = Conv2D('conv_e', x, ich*t, 1, activation=BNReLU)
    out_d = DWConv('conv_d', out_e, kernel, padding, stride, w_init, active)
    out_m = DWConv('conv_m', out_e, kernel, padding, stride, w_init, active)
    out = tf.concat([out_d, out_m], axis=-1)
    out = Conv2D('conv_p', out, och, 1, activation=None)
    with tf.variable_scope('conv_p'):
        out = BatchNorm('bn', out)
    return out


@layer_register(log_shape=True)
def inception(x, ch, stride, t=3, swap_block=False, w_init=None, active=None):
    '''
    ssdnet inception layer.
    '''
    k = 4 if stride == 2 else 3
    ich = ch // 2
    if stride == 1:
        oi = LinearBottleneck('conv1', x, ch, ch, k, stride=stride, t=t, w_init=w_init, active=active)
    else:
        oi = DownsampleBottleneck('conv1', x, ich, ch, k, stride=stride, t=t, w_init=w_init, active=active)
    oi = tf.split(oi, 2, axis=-1)
    o1 = oi[0]
    o2 = oi[1] + LinearBottleneck('conv2', oi[1], ich, ich, 5, t=t, w_init=w_init, active=active)

    if not swap_block:
        out = tf.concat([o1, o2], -1)
    else:
        out = tf.concat([o2, o1], -1)
    if stride == 1:
        out = tf.add(out, x)
    return out


def ssdnet_backbone(image, **kwargs):
    #
    with ssdnet_argscope() as scope:
        l = image #tf.transpose(image, perm=[0, 2, 3, 1])
        with argscope([BatchNorm], training=False):
            # conv1
            l = Conv2D('conv1', l, 32, 4, strides=2, activation=None, padding='SAME')
            with tf.variable_scope('conv1'):
                l = BNReLU(tf.concat([l, -l], axis=-1))
            l = MaxPooling('pool1', l, 2)
            l = tf.stop_gradient(l)
        # l = l + LinearBottleneck('conv3', l, 32, 32, 3, t=2)

        with argscope([BatchNorm], training=None):
            # conv2
            l = LinearBottleneck('conv2', l, 64, 32, 5, t=1)

            ch_all = [64, 96, 128]
            iters = [2, 4, 4]
            mults = [3, 4, 6]

            hlist = []
            for ii, (ch, it, mu) in enumerate(zip(ch_all, iters, mults)):
                for jj in range(it):
                    name = 'inc{}/{}'.format(ii, jj)
                    stride = 2 if jj == 0 else 1
                    swap_block = True if jj % 2 == 1 else False
                    l = inception(name, l, ch, stride, t=mu, swap_block=swap_block)
                hlist.append(l)

    return hlist
    #     # hyperfeatures
    #     h0, h1, h2 = hlist
    #     h0 = LinearBottleneck('downsample', h0, ch_all[0], ch_all[0], 4, stride=2, t=3)
    #     # h2 = LinearBottleneck('subpixel', h2, 192, 192, 3)
    #     h2 = tf.depth_to_space(h2, 2, name='upsample')
    #
    #     # nch = 192
    #     features = tf.concat([h0, h1, h2], -1, name='backbone_feature')
    # return features
