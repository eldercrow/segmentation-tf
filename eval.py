# -*- coding: utf-8 -*-
# File: eval.py

import tqdm
import sys, os
from collections import namedtuple, defaultdict
import numpy as np
import cv2
import json
import itertools

from scipy.io import loadmat
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from tensorpack.utils.utils import get_tqdm_kwargs, get_tqdm

from dataset.dataset_utils import load_many_from_db #, load_class_names
from common import SSDResize, CropPadTransform
from config import config as cfg
from utils.iou import IoU


def pred_image(img, model_func):
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
    return preds[0][0] # (H, W)


def pred_batch(img_batch, model_func):
    '''
    Run detection for a batch of images.
    This img_batch should be given from a eval_dataflow.

    Returns:
        [DetectionResult]
    '''
    # boxes: (N, na, 4)
    # probs: (N, na)
    # labels: (N, na)
    # preds = model_func(np.stack(img_batch, axis=0))
    preds = model_func(img_batch)
    return preds[0] # (N, H, W)


def pred_dataflow(df, model_func, tqdm_bar=None):
    '''
    '''
    if cfg.DATA.NAME in ('cityscapes', 'cocostuff'):
        return pred_cityscapes(df, model_func, tqdm_bar)
    # elif cfg.DATA.NAME in ('pvtdb', 'dss', 'voc'):
    #     return eval_dss(df, detect_func)
    else:
        raise ValueError


def pred_cityscapes(df, model_func, tqdm_bar=None):
    """
    Args:
        df: a DataFlow which produces (image, image_id)
        model_func: a callable, takes [image] and returns [prediction]

    Returns:
        list of dict, to be dumped to COCO json format
    """
    df.reset_state()
    all_results = {}
    with ExitStack() as stack:
        if tqdm_bar is None:
            tqdm_bar = stack.enter_context(get_tqdm(total=df.size()))
        # with tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()) as pbar:
        for img_batch, img_id_batch in df: #.get_data():
            preds_batch = pred_batch(img_batch, model_func)
            for preds, img_id in zip(preds_batch, img_id_batch):
                all_results[img_id] = preds.astype(np.uint8)
            tqdm_bar.update(1) # pbar.update(1)
    return all_results


def multithread_pred_dataflow(dataflows, model_funcs):
    '''
    '''
    num_worker = len(model_funcs)
    assert len(dataflows) == num_worker
    if num_worker == 1:
        return pred_dataflow(dataflows[0], model_funcs[0])
    kwargs = {'thread_name_prefix': 'EvalWorker'} if sys.version_info.minor >= 6 else {}
    with ThreadPoolExecutor(max_workers=num_worker, **kwargs) as executor, \
            tqdm.tqdm(total=sum([df.size() for df in dataflows])) as pbar:
        futures = []
        for dataflow, pred in zip(dataflows, model_funcs):
            futures.append(executor.submit(pred_dataflow, dataflow, pred, pbar))
        all_results_list = [fut.result() for fut in futures] # will be list of dict
        # all_results_list = list(itertools.chain(*[fut.result() for fut in futures])) # will be list of dict
    all_results = { k: v for res_dict in all_results_list for k, v in res_dict.items() }
    return all_results


# # https://github.com/pdollar/coco/blob/master/PythonAPI/pycocoEvalDemo.ipynb
def print_evaluation_scores(fn_all_results):
    '''
    '''
    if cfg.DATA.NAME in ('cityscapes',):
        return print_cityscapes_evaluation_scores(fn_all_results, 19)
    elif cfg.DATA.NAME in ('cocostuff',):
        return print_cityscapes_evaluation_scores(fn_all_results, 183)
    else:
        raise ValueError


def print_cityscapes_evaluation_scores(fn_all_results, num_classes):
    # ret = odict()
    # assert cfg.DATA.DSS.BASEDIR and os.path.isdir(cfg.DATA.DSS.BASEDIR)
    db_name = cfg.DATA.NAME

    bg_idx = 255 # for cityscapes

    if db_name == 'cocostuff':
        bg_idx = 0 # for cocostuff
        hh, ww = cfg.PREPROC.INPUT_SHAPE_EVAL
        aug = CropPadTransform(0, 0, ww, hh, bg_idx)
        # aug = CropPadTransform(0, 0, ww, hh, 255)

    # load the default testset defined in config
    db = load_many_from_db(db_name, add_gt=True, is_train=False)
    # db = DSSDetection.load_many(names='test')
    #
    db_all = { d['id']: d for d in db }
    mIoU = IoU(num_classes=num_classes)

    all_results = np.load(fn_all_results)

    for img_id, preds in all_results.items():
        fn_label = db_all[img_id]['fn_label']
        if fn_label.endswith('.mat'):
            labels = loadmat(fn_label)['S'].astype(int)
            labels = labels.astype(np.uint8)
            # labels = (labels - 1).astype(np.uint8)
            # labels = cv2.resize(labels, (ww, hh), interpolation=cv2.INTER_NEAREST)
            scale = min(ww / float(labels.shape[1]), hh / float(labels.shape[0]))
            labels = cv2.resize(labels, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            labels = aug.apply_image(labels)
        else:
            labels = cv2.imread(os.path.expanduser(fn_label), cv2.IMREAD_GRAYSCALE)

        if preds.shape != labels.shape:
            hh, ww = labels.shape
            preds = cv2.resize(preds, (ww, hh), cv2.INTER_NEAREST)
        preds = np.ravel(preds)
        labels = np.ravel(labels)
        vidx = np.where(labels != bg_idx)[0]
        preds = preds[vidx].astype(np.int64)
        labels = labels[vidx].astype(np.int64)
        # add an entry
        mIoU.add(preds, labels)

    ious_all, miou = mIoU.value()
    ret = {'miou': miou}
    return ret
