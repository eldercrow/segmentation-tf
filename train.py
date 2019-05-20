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
from tensorpack.tfutils.tower import PredictTowerContext
import tensorpack.utils.viz as tpviz

from data import get_train_dataflow, get_eval_dataflow
from viz import draw_predictions
#     draw_annotation, draw_proposal_recall,
#     draw_predictions, draw_final_outputs, draw_final_outputs_cv)
from eval import detect_one_image, detect_batch, eval_dataset, print_evaluation_scores
#     eval_dataset, detect_one_image, detect_batch,
#     print_evaluation_scores, create_detection_result)
from config import finalize_configs, config as cfg

from model.ssdnet_model import SSDNetModel as TrainModel
# from model.avanet_model import AVANetModel as TrainModel
# from model.mobilenet_model import MobilenetModel as TrainModel


def offline_evaluate(pred_func, output_file):
    df = get_eval_dataflow()
    all_results = eval_dataset(df, lambda img: detect_batch(img, pred_func))
    # all_results = eval_dataset(
    #     df, lambda img: detect_one_image(img, pred_func))
    logger.info('Dumping evaluation results')
    np.savez(output_file, **all_results)
    # with tqdm.tqdm(total=len(all_results), **get_tqdm_kwargs()) as pbar:
    #     for ii, res in enumerate(all_results):
    #         fn, ext = os.path.splitext(output_file)
    #         fn_out = os.path.join(fn + '_{:03d}'.format(ii) + ext)
    #         with open(fn_out, 'w') as f:
    #             json.dump(res, f)
    #         pbar.update(1)
    # with open(output_file, 'w') as f:
    #     json.dump(all_results, f)
    return print_evaluation_scores(output_file)
#
#
def predict(pred_func, input_file):
    img = cv2.imread(os.path.expanduser(input_file), cv2.IMREAD_COLOR)
    results = detect_one_image(img, pred_func)
    draw_predictions(img, results, results.shape)
    # final, _ = draw_final_outputs_cv(img, results)
    # viz = np.concatenate((img, final), axis=1)
    # tpviz.interactive_imshow(final)
#
#
# def predict_video(pred_func, video_name, vis=False, res_prefix=''):
#     reader = imageio.get_reader(video_name)
#     metadata = reader.get_meta_data()
#
#     frames = metadata['nframes']
#     fps = metadata['fps']
#     im_size = metadata['size']
#     wait_ms = int(1000. / fps)
#
#     if vis:
#         cv2.namedWindow('frame')
#     else:
#         fpath, fn = os.path.split(video_name)
#         result_file = fn.replace('.', '_result.')
#         if res_prefix:
#             out_name = os.path.join(res_prefix, result_file)
#         else:
#             out_name = os.path.join(fpath, result_file)
#         writer = imageio.get_writer(out_name, fps=fps)
#
#     colors = {}
#     scale = 0
#     for fidx, frame in enumerate(reader):
#         #
#         # if fidx % 4 != 0:
#         #     continue
#         im = frame[:, :, (2, 1, 0)].copy()
#         results = detect_one_image(im, pred_func)
#         im, colors = draw_final_outputs_cv(im, results)
#
#         if vis:
#             cv2.imshow('frame', im)
#             if (cv2.waitKey( wait_ms ) & 0xFF == ord('q')):
#                 break
#         else:
#             im = np.round(im * 255).astype(np.uint8)
#             writer.append_data(im[:, :, ::-1])
#             if fidx % 50 == 0:
#                 print('Processing {}/{} frame.'.format(fidx, frames))
#
#     try:
#         writer.close()
#     except:
#         pass


# class EvalCallback(Callback):
#     def __init__(self, in_names, out_names):
#         self._in_names, self._out_names = in_names, out_names
#
#     def _setup_graph(self):
#         self.pred = self.trainer.get_predictor(self._in_names, self._out_names)
#         # self.df = get_eval_dataflow()
#
#     def _before_train(self):
#         interval = cfg.TRAIN.EVAL_INTERVAL * cfg.TRAIN.NUM_EPOCH_PARTITIONS
#         max_epoch = cfg.TRAIN.EPOCHS_PER_CYCLE * cfg.TRAIN.NUM_CYCLES
#         eval_times = self.trainer.max_epoch // interval
#         # EVAL_TIMES = 5  # eval 5 times during training
#         # interval = self.trainer.max_epoch // (EVAL_TIMES + 1)
#         self.epochs_to_eval = set([interval * k for k in range(1, eval_times + 1)])
#         self.epochs_to_eval.add(self.trainer.max_epoch)
#         self.epochs_to_eval.add(1)
#         logger.info("[EvalCallback] Will evaluate at epoch " + str(sorted(self.epochs_to_eval)))
#
#     def _eval(self):
#         df = get_eval_dataflow(batch_size=2)
#         all_results = eval_dataset(
#             df, lambda img, w, h: detect_batch(img, w, h, self.pred))
#         # all_results = eval_dataset(df, lambda img: detect_one_image(img, self.pred))
#         output_file = os.path.join(
#             logger.get_logger_dir(), 'outputs{}.json'.format(self.global_step))
#         with open(output_file, 'w') as f:
#             json.dump(all_results, f)
#         try:
#             scores = print_evaluation_scores(output_file)
#         except Exception:
#             logger.exception("Exception in COCO evaluation.")
#             scores = {}
#         for k, v in scores.items():
#             self.trainer.monitors.put_scalar(k, v[0])
#
#     def _trigger_epoch(self):
#         if self.epoch_num in self.epochs_to_eval:
#             self._eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load a model for evaluation or training. Can overwrite BACKBONE.WEIGHTS')
    parser.add_argument('--resume', action='store_true', help='resume training if set.')
    parser.add_argument('--logdir', help='log directory', default='train_log/aspp')
    parser.add_argument('--evaluate', help="Run evaluation on COCO. "
                                           "This argument is the path to the output json evaluation file")
    parser.add_argument('--evalfromjson', action='store_true', help='Run evaluation using pre-evaluated .json file')
    parser.add_argument('--predict', help="Run prediction on a given image. "
                                          "This argument is the path to the input image file")
    parser.add_argument('--pred-video', action='store_true', help='flag to predict from a video')
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
        # cfg.USE_CUSTOM_PSROI = False
        # cfg.FOR_FLOPS = True
        finalize_configs(is_training=False)

        MODEL = TrainModel(1, cfg.PREPROC.INPUT_SHAPE_EVAL)

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

    is_training = not (args.evaluate or args.predict or args.export_graph)
    finalize_configs(is_training=is_training)

    # define model
    input_shape = cfg.PREPROC.INPUT_SHAPE_TRAIN if is_training else cfg.PREPROC.INPUT_SHAPE_EVAL
    batch_size = cfg.PREPROC.BATCH_SIZE
    if args.evaluate:
        batch_size = cfg.PREPROC.EVAL_BATCH_SIZE
    if args.predict:
        batch_size = 1
    MODEL = TrainModel(batch_size, input_shape)

    if args.evaluate or args.predict or args.export_graph or args.flops:
        assert args.load
        # finalize_configs(is_training=False)
        # MODEL = TrainModel()

        if args.predict:
            cfg.TEST.RESULT_SCORE_THRESH = cfg.TEST.RESULT_SCORE_THRESH_VIS

        pred = OfflinePredictor(PredictConfig(
            model=MODEL,
            session_init=get_model_loader(args.load),
            input_names=MODEL.get_inference_tensor_names()[0],
            output_names=MODEL.get_inference_tensor_names()[1]))

        if args.export_graph:
            from tensorflow.python.framework import graph_io
            export_path, export_name = os.path.split(args.export_graph)
            graph_io.write_graph(pred.sess.graph, export_path, export_name+'txt', as_text=True)
            graph_io.write_graph(pred.sess.graph, export_path, export_name, as_text=False)
        elif args.evaluate:
            assert args.evaluate.endswith('.json') or args.evaluate.endswith('.npz'), args.evaluate
            if args.evalfromjson:
                ret = print_evaluation_scores(args.evaluate)
            else:
                ret = offline_evaluate(pred, args.evaluate)
            print('mIoU = {:.3f}'.format(ret['miou']))
            # print('AP/class (head, torso, wp)')
            # for k, v in ret.items():
            #     rv = [round(vi, 3) for vi in v]
            #     if len(rv) > 1:
            #         print('{}: {} ({}, {}, {})'.format(k, rv[0], rv[1], rv[2], rv[3]))
            #     else:
            #         print('{}: {}'.format(k, rv[0]))

        elif args.predict:
            # assert cfg.DATA.CLASS_NAMES, "class names should be provided"
            if args.pred_video:
                predict_video(pred, args.predict)
            else:
                predict(pred, args.predict)
    else:
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
            lr = 0.5 * (1. + np.cos((e % cepoch) / float(cepoch - 1) * np.pi))
            lr = min_lr + (max_lr - min_lr) * lr
            return lr

        callbacks = [
            PeriodicCallback(
                ModelSaver(max_to_keep=10, keep_checkpoint_every_n_hours=1),
                every_k_epochs=cfg.TRAIN.SAVE_EPOCH_STEP),
            HyperParamSetterWithFunc(
                'learning_rate', partial(_compute_lr, max_lr=max_lr, min_lr=min_lr, cepoch=cyclic_epoch)),
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
