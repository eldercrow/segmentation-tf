# -*- coding: utf-8 -*-
# File: eval.py

import tqdm
import os
from collections import namedtuple, defaultdict
import numpy as np
import cv2
import json

from tensorpack.utils.utils import get_tqdm_kwargs

from dataset.dataset_utils import load_many_from_db #, load_class_names
from common import SSDResize
from config import config as cfg
from utils.iou import IoU


def detect_one_image(img, model_func):
    """
    Run detection on one image, using the TF callable.
    This function should handle the preprocessing internally.

    Args:
        img: an image
        model_func: a callable from TF model,
            takes image and returns (boxes, probs, labels, [masks])

    Returns:
        [DetectionResult]
    """
    # orig_shape = img.shape[:2]
    resizer = SSDResize(cfg.PREPROC.INPUT_SHAPE_EVAL)
    resized_img = resizer.augment(img)
    resized_img = np.expand_dims(resized_img, 0) # (1, h, w, 3)

    preds = model_func(resized_img)
    return preds[0][0]


def detect_batch(img_batch, model_func):
    '''
    Run detection for a batch of images.
    This img_batch should be given from a eval_dataflow.

    Returns:
        [DetectionResult]
    '''
    # boxes: (N, na, 4)
    # probs: (N, na)
    # labels: (N, na)
    preds = model_func(np.stack(img_batch, axis=0))
    return preds[0] # (N, H, W)


def eval_dataset(df, detect_func):
    '''
    '''
    if cfg.DATA.NAME == 'cityscapes':
        return eval_cityscapes(df, detect_func)
    # elif cfg.DATA.NAME in ('pvtdb', 'dss', 'voc'):
    #     return eval_dss(df, detect_func)
    else:
        raise ValueError


def eval_cityscapes(df, detect_func):
    """
    Args:
        df: a DataFlow which produces (image, image_id)
        detect_func: a callable, takes [image] and returns [DetectionResult]

    Returns:
        list of dict, to be dumped to COCO json format
    """
    df.reset_state()
    all_results = {}
    with tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()) as pbar:
        for img_batch, img_id_batch in df.get_data():
            preds_batch = detect_func(img_batch)
            for preds, img_id in zip(preds_batch, img_id_batch):
                all_results[img_id] = preds.astype(np.uint8)
            pbar.update(1)
    return all_results


# # https://github.com/pdollar/coco/blob/master/PythonAPI/pycocoEvalDemo.ipynb
def print_evaluation_scores(fn_all_results):
    '''
    '''
    if cfg.DATA.NAME in ('cityscapes'):
        return print_cityscapes_evaluation_scores(fn_all_results)
    else:
        raise ValueError


def print_cityscapes_evaluation_scores(fn_all_results):
    # ret = odict()
    # assert cfg.DATA.DSS.BASEDIR and os.path.isdir(cfg.DATA.DSS.BASEDIR)
    db_name = cfg.DATA.NAME

    # load the default testset defined in config
    db = load_many_from_db(db_name, add_gt=True, is_train=False)
    # db = DSSDetection.load_many(names='test')
    #
    db_all = { d['id']: d for d in db }
    mIoU = IoU(num_classes=19)

    all_results = np.load(fn_all_results)

    for img_id, preds in all_results.items():
        fn_label = db_all[img_id]['fn_label']
        labels = cv2.imread(os.path.expanduser(fn_label), cv2.IMREAD_GRAYSCALE)

        if preds.shape != labels.shape:
            hh, ww = labels.shape
            preds = cv2.resize(preds, (ww, hh), cv2.INTER_NEAREST)
        preds = np.ravel(preds)
        labels = np.ravel(labels)
        vidx = np.where(labels != 255)[0]
        preds = preds[vidx].astype(np.int64)
        labels = labels[vidx].astype(np.int64)
        # add an entry
        mIoU.add(preds, labels)

    _, miou = mIoU.value()
    ret = {'miou': miou}
    return ret
