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

    def to_dict(self, to_lower=False):
        """Convert to a nested dict. """
        return {k.lower() if to_lower else k: v.to_dict() if isinstance(v, AttrDict) else v
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
_C.DATA.CACHEDIR = '~/dataset/cache/tpack_seg'
# _C.DATA.TRAIN = ['coco_train2017',]   # i.e., trainval35k
# _C.DATA.VAL = 'coco_val2017'   # For now, only support evaluation on single dataset
_C.DATA.BASEDIR = ''
_C.DATA.CLASS_NAMES = []  # NUM_CLASS strings. Needs to be populated later by data loader
_C.DATA.IGNORE_NAMES = ['ignore', 'other',]
_C.DATA.IGNORE_LABEL = 255

_C.DATA.CITYSCAPES.BASEDIR = './data/cityscapes'
_C.DATA.CITYSCAPES.TRAIN = ['cityscapes_train',]
_C.DATA.CITYSCAPES.TEST = ['cityscapes_val',]
_C.DATA.CITYSCAPES.INPUT_SHAPE_TRAIN = (768, 768)
_C.DATA.CITYSCAPES.INPUT_SHAPE_EVAL = (1024, 2048)
_C.DATA.CITYSCAPES.NUM_CATEGORY = 19
_C.DATA.CITYSCAPES.IGNORE_LABEL = 255

_C.DATA.COCOSTUFF.BASEDIR = './data/cocostuff'
_C.DATA.COCOSTUFF.TRAIN = ['cocostuff_train',]
_C.DATA.COCOSTUFF.TEST = ['cocostuff_test',]
_C.DATA.COCOSTUFF.INPUT_SHAPE_TRAIN = (640, 640)
_C.DATA.COCOSTUFF.INPUT_SHAPE_EVAL = (640, 640)
_C.DATA.COCOSTUFF.NUM_CATEGORY = 183
_C.DATA.COCOSTUFF.IGNORE_LABEL = 0

_C.DATA.CAMVID.BASEDIR = '~/dataset/CamVid'
_C.DATA.CAMVID.TRAIN = ['camvid_train',]
_C.DATA.CAMVID.TEST = ['camvid_val',]
_C.DATA.CAMVID.INPUT_SHAPE_TRAIN = (640, 640)
_C.DATA.CAMVID.INPUT_SHAPE_EVAL = (720, 960)
_C.DATA.CAMVID.NUM_CATEGORY = 12
_C.DATA.CAMVID.IGNORE_LABEL = 12

# basemodel ----------------------
_C.BACKBONE.WEIGHTS = '/path/to/ImageNet-ResNet50.npz'
_C.BACKBONE.FILTER_SCALE = 1.0

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

_C.INFERENCE.SHIFT_PREDICTION = 0

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

_C.PREPROC.LABEL_MEAN = [255,] # 255 for cityscapes, 0 for cocostuff

_C.PREPROC.INPUT_SHAPE_TRAIN = (736, 736)
_C.PREPROC.INPUT_SHAPE_EVAL = (1024, 2048)
_C.PREPROC.NUM_WORKERS = 5
_C.PREPROC.BATCH_SIZE = 16
_C.PREPROC.EVAL_BATCH_SIZE = 10

_C.APPLY_PRUNING = False
_C.PRUNING.NAME = 'icnet_pruning'
_C.PRUNING.BEGIN_PRUNING_STEP = 0
_C.PRUNING.END_PRUNING_STEP = 15000
_C.PRUNING.TARGET_SPARSITY = 0.31
_C.PRUNING.PRUNING_FREQUENCY = 150
_C.PRUNING.SPARSITY_FUNCTION_BEGIN_STEP = 0
_C.PRUNING.SPARSITY_FUNCTION_END_STEP = 15000
_C.PRUNING.SPARSITY_FUNCTION_EXPONENT = 2

_C.QUANTIZE_NETWORK = False

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
    _C.PREPROC.LABEL_MEAN = getattr(_C.DATA, _C.DATA.NAME.upper()).IGNORE_LABEL

    _C.DATA.IGNORE_LABEL = getattr(_C.DATA, _C.DATA.NAME.upper()).IGNORE_LABEL

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
        # assert ngpu % 8 == 0 or 8 % ngpu == 0, ngpu
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
