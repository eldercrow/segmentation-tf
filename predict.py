#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import itertools
import numpy as np
import os, sys
import shutil
import tensorflow as tf
import cv2
import six
import tqdm

assert six.PY3, "This example requires Python 3!"

import tensorpack.utils.viz as tpviz
from tensorpack.predict import MultiTowerOfflinePredictor, OfflinePredictor, PredictConfig
from tensorpack.tfutils import get_model_loader, get_tf_version_tuple, model_utils
from tensorpack.tfutils.tower import PredictTowerContext
from tensorpack.utils import fs, logger
# from tensorpack.graph_builder import InputDesc
from tensorpack import PlaceholderInput, InputDesc

from model.icnet_model import ICNetModel as PredModel
# from dataset import DatasetRegistry, register_coco
# from config import config as cfg
from config import finalize_configs, config as cfg
from data import get_eval_dataflow
from eval import ( \
        pred_image, pred_dataflow, \
        multithread_pred_dataflow, print_evaluation_scores)
from viz import draw_predictions


def do_evaluate(pred_config, output_file, batch_size):
    '''
    Multi-gpu evaluation, if available
    '''
    num_tower = max(cfg.TRAIN.NUM_GPUS, 1)
    graph_funcs = MultiTowerOfflinePredictor(
        pred_config, list(range(num_tower))).get_predictors()
    dataflows = [get_eval_dataflow(batch_size, shard=k, num_shards=num_tower) for k in range(num_tower)]
    all_results = multithread_pred_dataflow(dataflows, graph_funcs)
    # df = get_eval_dataflow()
    # all_results = pred_dataflow(df, lambda img: detect_batch(img, pred_func))
    logger.info('Dumping evaluation results')
    np.savez(output_file, **all_results)
    return print_evaluation_scores(output_file)


def do_predict(pred_func, input_file):
    '''
    parse a single image
    '''
    img = cv2.imread(os.path.expanduser(input_file), cv2.IMREAD_COLOR)
    results = pred_image(img, pred_func)
    draw_predictions(img, results, results.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='trained model to load.', default='')
    parser.add_argument('--logdir', help='log directory', default='eval_log/aspp')
    parser.add_argument('--predict', help="Run prediction on a given image. "
                                          "This argument is the path to the input image file")
    parser.add_argument('--evaluate', help="Run evaluation on a given dataset. "
                                           "This argument is the path to the output npz evaluation file")
    parser.add_argument('--evalfromjson', action='store_true', help='Run evaluation using pre-evaluated .json file')
    parser.add_argument('--export-graph', type=str, default='',
                        help='if given, export an inference graph with the given name')
    parser.add_argument('--flops', action='store_true', help='Compute flops, default size=(1, 3, 544, 960).')
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in config.py",
                        nargs='+')

    args = parser.parse_args()
    try:
        cfg.update_args(args.config)
    except:
        pass

    if args.flops:
        finalize_configs(is_training=False)

        MODEL = PredModel(1, cfg.PREPROC.INPUT_SHAPE_EVAL)

        in_shape = [1, cfg.PREPROC.INPUT_SHAPE_EVAL[0], cfg.PREPROC.INPUT_SHAPE_EVAL[1], 3]
        pred_config = PredictConfig(model=MODEL,
                                    input_names=MODEL.get_inference_tensor_names()[0],
                                    output_names=MODEL.get_inference_tensor_names()[1])
        pred_config.inputs_desc[0] = InputDesc(type=tf.float32, shape=in_shape, name='data')
        inputs = PlaceholderInput()
        inputs.setup(pred_config.inputs_desc)
        with PredictTowerContext(''):
            MODEL.build_graph(*inputs.get_input_tensors())
        model_utils.describe_trainable_vars()
        tf.profiler.profile(
                tf.get_default_graph(),
                cmd='op',
                options=tf.profiler.ProfileOptionBuilder.float_operation())
        sys.exit(0)

    assert args.load

    # is_training = False
    finalize_configs(is_training=False)

    # define model
    input_shape = cfg.PREPROC.INPUT_SHAPE_EVAL
    batch_size = cfg.PREPROC.BATCH_SIZE
    if args.evaluate:
        batch_size = cfg.PREPROC.EVAL_BATCH_SIZE
    if args.predict:
        batch_size = 1
    MODEL = PredModel(batch_size, input_shape)

    pred_config = PredictConfig( \
            model=MODEL, \
            session_init=get_model_loader(args.load), \
            input_names=MODEL.get_inference_tensor_names()[0], \
            output_names=MODEL.get_inference_tensor_names()[1])
    if args.evaluate:
        assert args.evaluate.endswith('.json') or args.evaluate.endswith('.npz'), args.evaluate
        if args.evalfromjson:
            ret = print_evaluation_scores(args.evaluate)
        else:
            ret = do_evaluate(pred_config, args.evaluate, batch_size)
        print('mIoU = {:.3f}'.format(ret['miou']))
    else:
        pred = OfflinePredictor(pred_config)
        if args.export_graph:
            from tensorflow.python.framework import graph_io
            export_path, export_name = os.path.split(args.export_graph)
            graph_io.write_graph(pred.sess.graph, export_path, export_name+'txt', as_text=True)
            graph_io.write_graph(pred.sess.graph, export_path, export_name, as_text=False)
        elif args.predict:
            do_predict(pred, args.predict)
        # if args.pred_video:
        #     predict_video(pred, args.predict)
        # else:
        #     do_predict(pred, args.predict)
