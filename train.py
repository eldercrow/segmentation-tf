#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py

import os, sys
import argparse
import cv2
import imageio
import shutil
import itertools
import tqdm
import numpy as np
import json
import six
from functools import partial

from tensorpack.utils.utils import get_tqdm_kwargs

import tensorflow as tf
try:
    import horovod.tensorflow as hvd
except ImportError:
    pass

assert six.PY3, "R-ASPP requires Python 3!"

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils import optimizer, model_utils
#from tensorpack.tfutils.tower import PredictTowerContext
import tensorpack.utils.viz as tpviz

from data import get_train_dataflow #, get_eval_dataflow
# from viz import draw_predictions
# from eval import ( \
#         detect_one_image, detect_batch, pred_dataflow, \
#         multithread_pred_dataflow, print_evaluation_scores)
from config import finalize_configs, config as cfg

from model.icnet_model import ICNetModel as TrainModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load a model for evaluation or training. Can overwrite BACKBONE.WEIGHTS')
    parser.add_argument('--resume', action='store_true', help='resume training if set.')
    parser.add_argument('--logdir', help='log directory', default='train_log/aspp')
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in config.py",
                        nargs='+')

    args = parser.parse_args()
    try:
        cfg.update_args(args.config)
    except:
        pass

    # is_training = not (args.evaluate or args.predict or args.export_graph)
    finalize_configs(is_training=True)

    # define model
    input_shape = cfg.PREPROC.INPUT_SHAPE_TRAIN
    batch_size = cfg.PREPROC.BATCH_SIZE
    # if args.evaluate:
    #     batch_size = cfg.PREPROC.EVAL_BATCH_SIZE
    # if args.predict:
    #     batch_size = 1
    MODEL = TrainModel(batch_size, input_shape)

    is_horovod = cfg.TRAINER == 'horovod'
    if is_horovod:
        hvd.init()
        logger.info("Horovod Rank={}, Size={}".format(hvd.rank(), hvd.size()))

    if not is_horovod or hvd.rank() == 0:
        logger.set_logger_dir(args.logdir)

    # get dataflow to get the number of images in the training dataset
    df = get_train_dataflow()
    num_images = df.size()

    # factor = 8. / cfg.TRAIN.NUM_GPUS
    num_gpus = cfg.TRAIN.NUM_GPUS
    step_size = num_images // (cfg.TRAIN.NUM_EPOCH_PARTITIONS * num_gpus)
    cyclic_epoch = cfg.TRAIN.NUM_EPOCH_PARTITIONS * cfg.TRAIN.EPOCHS_PER_CYCLE
    total_epoch = cyclic_epoch * cfg.TRAIN.NUM_CYCLES
    max_lr = cfg.TRAIN.MAX_LR
    min_lr = cfg.TRAIN.MIN_LR

    # lr function
    def _compute_lr(e, x, max_lr, min_lr, cepoch):
        # we won't use x, but this is the function template anyway
        w = 0.5 * (1. + np.cos((e % cepoch) / float(cepoch - 1) * np.pi)) # from 1 to 0
        lr = min_lr + (max_lr - min_lr) * w # from max to min
        return lr

    # lr function
    def _compute_mom(e, x, min_m, max_m, cepoch):
        # we won't use x, but this is the function template anyway
        w = 0.5 * (1. + np.cos((e % cepoch) / float(cepoch - 1) * np.pi)) # from 1 to 0
        mom = max_m - (max_m - min_m) * w # from min to max
        return mom

    callbacks = [
        PeriodicCallback(
            ModelSaver(max_to_keep=10, keep_checkpoint_every_n_hours=1),
            every_k_epochs=cfg.TRAIN.SAVE_EPOCH_STEP),
        HyperParamSetterWithFunc(
            'learning_rate', partial(_compute_lr, max_lr=max_lr, min_lr=min_lr, cepoch=cyclic_epoch)),
        HyperParamSetterWithFunc(
            'bn_momentum', partial(_compute_mom, min_m=0.9, max_m=0.999, cepoch=cyclic_epoch)),
        # EvalCallback(*MODEL.get_inference_tensor_names()),
        # PeakMemoryTracker(),
        EstimatedTimeLeft(),
        SessionRunTimeout(60000).set_chief_only(True),   # 1 minute timeout
    ]

    if not is_horovod:
        try:
            callbacks.append(GPUUtilizationTracker())
        except:
            pass

    if args.resume and not args.load:
        args.load = os.path.join(args.logdir, 'checkpoint')

    if args.load:
        session_init = get_model_loader(args.load)
    else:
        session_init = get_model_loader(cfg.BACKBONE.WEIGHTS) if cfg.BACKBONE.WEIGHTS else None

    TrainCfg = AutoResumeTrainConfig if args.resume else TrainConfig
    traincfg = TrainCfg(
        model=MODEL,
        data=QueueInput(df),
        callbacks=callbacks,
        steps_per_epoch=step_size,
        max_epoch=total_epoch,
        session_init=session_init,
    )
    if is_horovod:
        # horovod mode has the best speed for this model
        trainer = HorovodTrainer()
    else:
        # nccl mode has better speed than cpu mode
        trainer = SyncMultiGPUTrainerReplicated(cfg.TRAIN.NUM_GPUS, mode='nccl')
    launch_train_with_config(traincfg, trainer)
