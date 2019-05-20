# -*- coding: utf-8 -*-
# File: config.py

import numpy as np
import os
import pprint
from tensorpack.utils import logger
from tensorpack.utils.gpu import get_num_gpu
from ast import literal_eval

__all__ = ['config', 'finalize_configs']


class AttrDict():
    def __getattr__(self, name):
        ret = AttrDict()
        setattr(self, name, ret)
        return ret

    def __str__(self):
        return pprint.pformat(self.to_dict(), indent=1)

    __repr__ = __str__

    def to_dict(self):
        """Convert to a nested dict. """
        return {k: v.to_dict() if isinstance(v, AttrDict) else v
                for k, v in self.__dict__.items()}

    def update_args(self, args):
        """Update from command line args. """
        for cfg in args:
            keys, v = cfg.split('=', maxsplit=1)
            keylist = keys.split('.')

            dic = self
            for i, k in enumerate(keylist[:-1]):
                assert k in dir(dic), "Unknown config key: {}".format(keys)
                dic = getattr(dic, k)
            key = keylist[-1]

            oldv = getattr(dic, key)
            if not isinstance(oldv, str):
                if isinstance(oldv, (list, tuple)):
                    v = literal_eval(v)
                else:
                    v = eval(v)
            setattr(dic, key, v)


config = AttrDict()
_C = config     # short alias to avoid coding

# mode flags ---------------------
_C.TRAINER = 'replicated'  # options: 'horovod', 'replicated'
_C.OPENVINO = False
_C.FOR_FLOPS = False

# dataset -----------------------
_C.DATA.NAME = 'cityscapes' # oneof 'voc', 'coco', 'mapillary', 'dss', 'pvtdb'
_C.DATA.NUM_CATEGORY = 19    # 80 categories
_C.DATA.CACHEDIR = './data/cache/tpack_seg'
# _C.DATA.TRAIN = ['coco_train2017',]   # i.e., trainval35k
# _C.DATA.VAL = 'coco_val2017'   # For now, only support evaluation on single dataset
_C.DATA.BASEDIR = ''
_C.DATA.CLASS_NAMES = []  # NUM_CLASS strings. Needs to be populated later by data loader
_C.DATA.IGNORE_NAMES = ['ignore', 'other',]

_C.DATA.CITYSCAPES.BASEDIR = './data/cityscapes'
_C.DATA.CITYSCAPES.TRAIN = ['cityscapes_train',]
_C.DATA.CITYSCAPES.TEST = ['cityscapes_val',]
_C.DATA.CITYSCAPES.INPUT_SHAPE_TRAIN = (640, 1280)
_C.DATA.CITYSCAPES.INPUT_SHAPE_EVAL = (1024, 2048)
_C.DATA.CITYSCAPES.NUM_CATEGORY = 19

# basemodel ----------------------
_C.BACKBONE.WEIGHTS = '/path/to/ImageNet-ResNet50.npz'

# schedule -----------------------
# The schedule and learning rate here is defined for a total batch size of 8.
# If not running with 8 GPUs, they will be adjusted automatically in code.
_C.TRAIN.NUM_GPUS = None         # by default, will be set from code
_C.TRAIN.WEIGHT_DECAY = 4e-5
_C.TRAIN.MAX_LR = 1e-02
_C.TRAIN.MIN_LR = 5e-05
_C.TRAIN.NUM_EPOCH_PARTITIONS = 1
_C.TRAIN.EPOCHS_PER_CYCLE = 180
_C.TRAIN.NUM_CYCLES = 2
_C.TRAIN.SAVE_EPOCH_STEP = 1
_C.TRAIN.EVAL_INTERVAL = 10

# preprocessing --------------------
# Alternative old (worse & faster) setting: 600, 1024
# sizes = np.arange(13, 28) * 32
# sizes[0], sizes[7] = sizes[7], sizes[0]
# _C.PREPROC.SHORT_EDGE_SIZE = sizes.tolist()
# _C.PREPROC.MAX_SIZE = 1440
# mean and std in RGB order.
# Un-scaled version: [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
_C.PREPROC.PIXEL_MEAN = [123.675, 116.28, 103.53]
_C.PREPROC.PIXEL_STD = [58.395, 57.12, 57.375]

_C.PREPROC.INPUT_SHAPE_TRAIN = (736, 736)
_C.PREPROC.INPUT_SHAPE_EVAL = (1024, 2048)
_C.PREPROC.NUM_WORKERS = 5
_C.PREPROC.BATCH_SIZE = 16
_C.PREPROC.EVAL_BATCH_SIZE = 10


def finalize_configs(is_training):
    """
    Run some sanity checks, and populate some configs from others
    """
    if not _C.DATA.CLASS_NAMES:
        # from dataset.dataset_utils import load_class_names
        # classes, _ = load_class_names(_C.DATA.NAME)
        num_classes = getattr(_C.DATA, _C.DATA.NAME.upper()).NUM_CATEGORY
        _C.DATA.NUM_CATEGORY = num_classes
        # _C.DATA.CLASS_NAMES = classes
    _C.DATA.NUM_CLASS = _C.DATA.NUM_CATEGORY
    _C.PREPROC.INPUT_SHAPE_TRAIN = getattr(_C.DATA, _C.DATA.NAME.upper()).INPUT_SHAPE_TRAIN
    _C.PREPROC.INPUT_SHAPE_EVAL = getattr(_C.DATA, _C.DATA.NAME.upper()).INPUT_SHAPE_EVAL

    if is_training:
        os.environ['TF_AUTOTUNE_THRESHOLD'] = '1'
        assert _C.TRAINER in ['horovod', 'replicated'], _C.TRAINER

        # setup NUM_GPUS
        if _C.TRAINER == 'horovod':
            import horovod.tensorflow as hvd
            ngpu = hvd.size()
        else:
            assert 'OMPI_COMM_WORLD_SIZE' not in os.environ
            ngpu = get_num_gpu()
        assert ngpu % 8 == 0 or 8 % ngpu == 0, ngpu
        if _C.TRAIN.NUM_GPUS is None:
            _C.TRAIN.NUM_GPUS = ngpu
        else:
            if _C.TRAINER == 'horovod':
                assert _C.TRAIN.NUM_GPUS == ngpu
            else:
                assert _C.TRAIN.NUM_GPUS <= ngpu
    else:
        # autotune is too slow for inference
        os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
        # no random scaling of inputs
        # _C.PREPROC.SHORT_EDGE_SIZE = _C.PREPROC.SHORT_EDGE_SIZE[0:1]

    logger.info("Config: ------------------------------------------\n" + str(_C))
