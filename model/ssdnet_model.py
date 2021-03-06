#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py
#
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary

from tensorpack.tfutils import optimizer
from backbone.basemodel import image_preprocess
from backbone.ssdnet import ssdnet_backbone, ssdnet_argscope
# from backbone.ssdnetv2 import ssdnet_backbone, ssdnet_argscope
from aspp.losses import aspp_losses
from aspp.ssdnet_aspp import aspp_features
from aspp.inference import aspp_inference

from config import config as cfg


class SSDNetModel(ModelDesc):
    '''
    '''
    def __init__(self, batch_size, input_shape):
        self._batch_size = batch_size
        self._input_h, self._input_w = input_shape

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.005, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        # opt = tf.train.RMSPropOptimizer(lr, epsilon=0.1)
        return opt

    def get_inference_tensor_names(self):
        """
        Returns two lists of tensor names to be used to create an inference callable.

        Returns:
            [str]: input names
            [str]: output names
        """
        out = ['segmentation_output']
        return ['data'], out

    def inputs(self): #, input_h, input_h, n_gt):
        ret = [ \
                tf.TensorSpec((self._batch_size, self._input_h, self._input_w, 3), tf.float32, 'data'),
                tf.TensorSpec((self._batch_size, self._input_h, self._input_w, 1), tf.int32, 'gt_labels'),
              ] # all > 0
        return ret

    def build_graph(self, *inputs):
        is_training = get_current_tower_context().is_training
        images, gt_labels = inputs

        # preprocessing
        images = image_preprocess(images, bgr=True)
        # backbone feature
        hlist = ssdnet_backbone(images)
        # get segmentation output
        feat = aspp_features(hlist)

        # loss
        if is_training:
            cls_loss = aspp_losses(feat, gt_labels, cfg.DATA.NUM_CLASS)
            # wd_pattern = '.*(?:W|gamma)'
            wd_pattern = '.*W'
            wd_cost = regularize_cost(wd_pattern, \
                                      l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), \
                                      name='wd_cost')

            costs = [cls_loss, wd_cost]
            total_cost = tf.add_n(costs, 'total_cost')

            add_moving_summary(total_cost, wd_cost)
            return total_cost
        else: # is_training
            inference_res = aspp_inference(feat, images)
