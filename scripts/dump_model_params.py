#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dump-model-params.py

import numpy as np
import six
import argparse
import os, sys
import tensorflow as tf

from tensorpack.tfutils import varmanip
from tensorpack.tfutils.common import get_op_tensor_name
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util as tensor_util

def _process_icnet_names(name):
    '''
    '''
    k = '{}'.format(name)
    k = k.replace('batch_normalization/beta', 'bn/beta')
    k = k.replace('batch_normalization/gamma', 'bn/gamma')
    k = k.replace('/moving_mean', '/mean/EMA')
    k = k.replace('/moving_variance', '/variance/EMA')
    k = k.replace('/weights', '/W')
    k = k.replace('/biases', '/b')
    return k


def _merge_sparsity_mask(params):
    '''
    Merge weights and masks
    '''
    r_params = {}
    for k, v in params.items():
        rk = k.replace('/mask:0', 'W:0')
        if rk not in r_params:
            r_params[rk] = v.copy()
        else:
            r_params[rk] *= v
    return r_params


def _measure_sparsity(params):
    '''
    '''
    num_w = 0
    num_zw = 0

    for k, v in params.items():
        if k.endswith('W:0'):
            num_w += v.size
            num_zw += np.sum(v == 0)
    sparsity = float(num_zw) / float(num_w)
    return sparsity


def _temp_fix_cocostuff(params):
    for k, v in params.items():
        if k in ('conv6_cls/W:0', 'sub24_out/W:0', 'sub4_out/W:0') and v.shape[-1] == 182:
            nshape = list(v.shape)
            nshape[-1] = 183
            nw = np.zeros(nshape, np.float32)
            nw[:, :, :, 1:] = v
            params[k] = nw
        if k in ('conv6_cls/b:0', 'sub24_out/b:0', 'sub4_out/b:0') and v.shape[-1] == 182:
            nshape = list(v.shape)
            nshape[-1] = 183
            nw = -10.0 * np.ones(nshape, np.float32)
            nw[1:] = v
            params[k] = nw
    return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Keep only TRAINABLE and MODEL variables in a checkpoint.')
    # parser.add_argument('--meta', help='metagraph file', required=True)
    parser.add_argument(dest='input', help='input model file, has to be a TF checkpoint')
    parser.add_argument(dest='output', help='output model file, can be npz or TF checkpoint')
    args = parser.parse_args()

    # this script does not need GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # tf.train.import_meta_graph(args.meta, clear_devices=True)

    # loading...
    if args.input.endswith('.npz'):
        dic = np.load(args.input)
    elif args.input.endswith('.pb'):
        with tf.Session() as sess:
            with gfile.FastGFile(args.input, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')
                graph_nodes = [n for n in graph_def.node]
        dic = {n.name: tensor_util.MakeNdarray(n.attr['value'].tensor) for n in graph_nodes if n.op=='Const'}
    else:
        dic = varmanip.load_chkpt_vars(args.input)
    dic = {get_op_tensor_name(k)[1]: v for k, v in six.iteritems(dic)}

    dic = _merge_sparsity_mask(dic)

    dic_to_dump = {}
    postfixes = ['W:0', 'b:0', 'beta:0', 'gamma:0', 'EMA:0',
            'weights:0', 'kernels', 'biases:0', 'moving_mean:0', 'moving_variance:0']
    for k, v in dic.items():
        found = False
        for p in postfixes:
            if p in k:
                found = True
                break
        if found:
            kk = _process_icnet_names(k)
            # if ('conv6_cls' in k) or ('sub4_out' in k) or ('sub24_out' in k):
            #     continue
            # else:
            #     dic_to_dump[k] = v
            dic_to_dump[kk] = v
            # print(k)
    sparsity = _measure_sparsity(dic_to_dump)
    dic_to_dump = _temp_fix_cocostuff(dic_to_dump)
    varmanip.save_chkpt_vars(dic_to_dump, args.output)
    print('Net sparsity = {}'.format(sparsity))
    #
    # import ipdb
    # ipdb.set_trace()
    # # varmanip.save_chkpt_vars(dic, args.output)
    #
    # # save variables that are GLOBAL, and either TRAINABLE or MODEL
    # var_trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # var_model = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
    # var_to_dump = var_trainable + var_model
    # assert len(set(var_to_dump)) == len(var_to_dump), "TRAINABLE and MODEL variables have duplication!"
    # globvarname = [k.name for k in tf.global_variables()]
    # var_to_dump = set([k.name for k in var_to_dump if k.name in globvarname])
    #
    # for name in var_to_dump:
    #     assert name in dic, "Variable {} not found in the model!".format(name)
    #
    # dic_to_dump = {k: v for k, v in six.iteritems(dic) if k in var_to_dump}
    # varmanip.save_chkpt_vars(dic_to_dump, args.output)
