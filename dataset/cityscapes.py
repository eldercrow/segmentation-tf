# -*- coding: utf-8 -*-
# File: cityscapes.py

import cv2
import numpy as np
import os, sys
from termcolor import colored
from tabulate import tabulate

import tensorflow as tf
from tensorpack.utils import logger
from tensorpack.utils.timer import timed_operation
from tensorpack.utils.argtools import log_once

from config import config as cfg

from dataset.dataset_utils import load_from_cache, save_to_cache


# __all__ = ['DSSDetection', 'DSSMeta']

# class _DSSMeta(object):
#     # just to check validity
#     INSTANCE_TO_BASEDIR = {
#         'dss2016china_train': 'DSS2016china',
#         'dss2018china_train': 'DSS2018china',
#         'dss2016us_train': 'DSS2016us',
#         'dss2016china_test': 'DSS2016china'
#     }
#
#     def valid(self):
#         return hasattr(self, 'cat_names')
#
#     def create(self, cat_ids, cat_names):
#         """
#         cat_ids: list of ids
#         cat_names: list of names
#         """
#         assert not self.valid()
#         # assert len(cat_ids) == DSS_NUM_CATEGORY and len(cat_names) == DSS_NUM_CATEGORY
#         self.cat_names = cat_names
#         self.class_names = ['BG'] + self.cat_names
#
#         cfg.DATA.DSS.CLASS_NAMES = self.class_names # including BG
#         cfg.DATA.DSS.NUM_CATEGORY = len(self.cat_names)
#
#
# DSSMeta = _DSSMeta()


class CITYSCAPESSegmentation(object):
    def __init__(self, basedir, name):
        # assert name in DSSMeta.INSTANCE_TO_BASEDIR.keys(), name
        self.name = name

        self._basedir = os.path.expanduser(basedir)
        assert os.path.isdir(self._basedir), self._basedir
        # self._imgdir = os.path.abspath(os.path.join(dbdir, 'leftImg8bit'))
        # assert os.path.isdir(self._imgdir), self._imgdir
        # self._anndir = os.path.abspath(os.path.join(dbdir, 'gtFine'))
        # assert os.path.isdir(self._anndir), self._anndir

        fn_imageset = os.path.join(self._basedir, name.split('_')[1] + '_list.txt')
        with open(fn_imageset, 'r') as fh:
            list_all = [l.split(' ') for l in fh.readlines()]
        self._imageset = [(l[0].strip(), l[1].strip()) for l in list_all]

        logger.info("Image list loaded from {}.".format(fn_imageset))

    def load(self, add_gt=True):
        """
        Args:
            add_gt: whether to add ground truth bounding box annotations to the dicts

        Returns:
            a list of dict, each has keys including:
                'height', 'width', 'id', 'file_name',
                and (if add_gt is True) 'boxes', 'class', 'is_crowd', and optionally
                'segmentation'.
        """
        assert add_gt == True, 'Temporal, add_gt must be true for now'

        with timed_operation('Load images and labels for {}'.format(self.name)):
            # first try to load from cache
            try:
                imgs = load_from_cache(self.name, ctime=os.path.getmtime(__file__))
                logger.info('Loaded from cache {}'.format(self.name + '.pkl'))
            except IOError:
                imgs = [{'fn_img': f[0], 'fn_label': f[1]} for f in self._imageset]
                valid_imgs = []
                for img in imgs:
                    try:
                        self._use_absolute_file_name(img)
                    except IOError:
                        logger.info('skipping {}'.format(img['file_name']))
                        continue
                    valid_imgs.append(img)
                imgs, valid_imgs = valid_imgs, imgs
                save_to_cache(imgs, self.name)
            return imgs

    def _use_absolute_file_name(self, img):
        """
        Change relative filename to abosolute file name.
        """
        fn_img, fn_label = img['fn_img'], img['fn_label']

        img['id'] = '_'.join(os.path.split(fn_img)[-1].split('_')[:3])
        img['fn_img'] = os.path.join(self._basedir, fn_img)
        img['fn_label'] = os.path.join(self._basedir, fn_label)

        if not os.path.isfile(img['fn_img']):
            raise IOError
        if not os.path.isfile(img['fn_label']):
            raise IOError

    # def print_class_histogram(self, imgs):
    #     nr_class = len(DSSMeta.class_names)
    #     hist_bins = np.arange(nr_class + 1)
    #
    #     # Histogram of ground-truth objects
    #     gt_hist = np.zeros((nr_class,), dtype=np.int)
    #     for entry in imgs:
    #         # filter crowd?
    #         gt_inds = np.where(
    #             (entry['class'] > 0) & (entry['difficult'] == 0))[0]
    #         gt_classes = entry['class'][gt_inds]
    #         gt_hist += np.histogram(gt_classes, bins=hist_bins)[0]
    #     data = [[DSSMeta.class_names[i], v] for i, v in enumerate(gt_hist)]
    #     data.append(['total', sum([x[1] for x in data])])
    #     table = tabulate(data, headers=['class', '#box'], tablefmt='pipe')
    #     logger.info("Ground-Truth Boxes:\n" + colored(table, 'cyan'))

    @staticmethod
    def load_many(basedir='', names=[], add_gt=True):
        """
        Load and merges several instance files together.

        Returns the same format as :meth:`DSSDetection.load`.
        """
        # to simplify things
        if not basedir:
            basedir = cfg.DATA.CITYSCAPES.BASEDIR
        if isinstance(names, str) and names in ('train', 'test'):
            names = getattr(cfg.DATA.CITYSCAPES, names.upper())

        if not isinstance(names, (list, tuple)):
            names = [names]
        ret = []
        for n in names:
            db = CITYSCAPESSegmentation(basedir, n)
            ret.extend(db.load(add_gt))
        return ret


if __name__ == '__main__':
    basedir = os.path.expanduser('~/dataset/cityscapes')
    names = ['cityscapes_train']
    annots = CITYSCAPESSegmentation.load_many(basedir, names)

    # c = DSSDetection(cfg.DATA.BASEDIR, 'train2014')
    # gt_boxes = c.load(add_gt=True, add_mask=True)
    # print("#Images:", len(gt_boxes))
    # c.print_class_histogram(gt_boxes)
