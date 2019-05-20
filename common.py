# -*- coding: utf-8 -*-
# File: common.py

import numpy as np
import cv2

from tensorpack.dataflow import RNGDataFlow
from tensorpack.dataflow.imgaug import transform
from tensorpack.dataflow.imgaug.base import ImageAugmentor

import pycocotools.mask as cocomask


class DataFromListOfDict(RNGDataFlow):
    def __init__(self, lst, keys, shuffle=False):
        self._lst = lst
        self._keys = keys
        self._shuffle = shuffle
        self._size = len(lst)

    def size(self):
        return self._size

    def get_data(self):
        if self._shuffle:
            self.rng.shuffle(self._lst)
        for dic in self._lst:
            dp = [dic[k] for k in self._keys]
            yield dp


class CropPadTransform(transform.ImageTransform):
    def __init__(self, x0, y0, x1, y1, mean_rgbgr):
        super(CropPadTransform, self).__init__()
        self._init(locals())

    def apply_image(self, img):
        ndim = img.ndim
        if ndim == 2:
            img = np.expand_dims(img, 2) # (h, w, 1)

        hh, ww, nch = img.shape
        if self.x0 >= 0 and self.y0 >= 0 and self.x1 <= ww and self.y1 <= hh:
            r_img = img[self.y0:self.y1, self.x0:self.x1]
        else:
            # image crop region
            ix0 = int(np.maximum(0, self.x0))
            ix1 = int(np.minimum(ww, self.x1))
            iy0 = int(np.maximum(0, self.y0))
            iy1 = int(np.minimum(hh, self.y1))
            # paste region
            rx0 = int(np.maximum(0, -self.x0))
            ry0 = int(np.maximum(0, -self.y0))
            rx1 = rx0 + (ix1 - ix0)
            ry1 = ry0 + (iy1 - iy0)
            # crop or pad
            w = self.x1 - self.x0
            h = self.y1 - self.y0
            r_img = np.tile(np.reshape(self.mean_rgbgr, [1, 1, nch]), [h, w, 1])
            r_img[ry0:ry1, rx0:rx1] = img[iy0:iy1, ix0:ix1]

        if ndim == 2:
            r_img = np.squeeze(r_img, axis=2)
        return r_img

    def apply_coords(self, coords):
        coords[:, 0] -= self.x0
        coords[:, 1] -= self.y0
        return coords


class SSDCropRandomShape(transform.TransformAugmentorBase):
    """ Random crop with a random shape"""

    def __init__(self,
                 base_hw=(720, 720),
                 scale_exp=2.0,
                 aspect_exp=1.1,
                 mean_rgbgr=np.array([127, 127, 127])):
        """
        Randomly crop a box of shape (h, w), sampled from [min, max] (both inclusive).
        If max is None, will use the input image shape.

        Args:
            wmin, hmin, wmax, hmax: range to sample shape.
            max_aspect_ratio (float): the upper bound of ``max(w,h)/min(w,h)``.
        """
        # if max_aspect_ratio is None:
        #     max_aspect_ratio = 9999999
        self._init(locals())

    def _get_augment_params(self, img):
        h, w = img.shape[:2]
        area = h * w
        scale_e, asp_e = self.rng.uniform(-1.0, 1.0, size=[2])
        # random scale and aspect ratio
        scale = np.power(self.scale_exp, scale_e)
        asp = np.sqrt(np.power(self.aspect_exp, asp_e))
        # define crop box size
        ww = int(self.base_hw[1] * scale * asp + 0.5)
        hh = int(self.base_hw[0] * scale / asp + 0.5)
        # and crop box itself
        x0 = 0 if w == ww else self.rng.randint(min(w - ww, 0), max(w - ww, 0))
        y0 = 0 if h == hh else self.rng.randint(min(h - hh, 0), max(h - hh, 0))
        x1 = x0 + ww
        y1 = y0 + hh
        return CropPadTransform(x0, y0, x1, y1, self.mean_rgbgr)


class SSDResize(transform.TransformAugmentorBase):
    ''' Resize to a fixed shape after crop '''
    def __init__(self, size, interp=cv2.INTER_LINEAR):
        if not isinstance(size, (list, tuple)):
            size = [size, size]
        self._init(locals())

    def _get_augment_params(self, img):
        h, w = img.shape[:2]
        return transform.ResizeTransform(h, w, self.size[0], self.size[1], self.interp)


class SSDColorJitter(ImageAugmentor):
    ''' Random color jittering '''
    def __init__(self, \
                 mean_rgbgr=[127.0, 127.0, 127.0], \
                 rand_l=0.1 * 255, \
                 rand_c=0.2, \
                 rand_h=0.1 * 255):
        super(SSDColorJitter, self).__init__()
        min_rgbgr = -mean_rgbgr
        max_rgbgr = min_rgbgr + 255.0
        self._init(locals())

    def _get_augment_params(self, _):
        return self.rng.uniform(-1.0, 1.0, [8])

    def _augment(self, img, rval):
        rflag = (rval[5:] > 0.3333).astype(float)
        rval[0] *= (self.rand_l * rflag[0])
        rval[1] = np.power(1.0 + self.rand_c, rval[3] * rflag[1])
        rval[2:4] *= (self.rand_h * rflag[2])
        rval[4] = -(rval[2] + rval[3])

        for i in range(3):
            add_val = (rval[0] + rval[i+2] - self.mean_rgbgr[i]) * rval[1] + self.mean_rgbgr[i]
            img[:, :, i] = img[:, :, i] * rval[1] + add_val
            # img[:, :, i] = np.maximum(0.0, np.minimum(255.0,
            #     (img[:, :, i] + add_val) * rval[1] + self.mean_rgbgr[i]))
        return np.maximum(0.0, np.minimum(255.0, img))


class CustomResize(transform.TransformAugmentorBase):
    """
    Try resizing the shortest edge to a certain number
    while avoiding the longest edge to exceed max_size.
    """

    def __init__(self, size, max_size, stride=32, interp=cv2.INTER_LINEAR):
        """
        Args:
            size (int): the size to resize the shortest edge to.
            max_size (int): maximum allowed longest edge.
        """
        if not isinstance(size, (list, tuple)):
            size = [size]
        self._init(locals())

    def _get_augment_params(self, img):
        h, w = img.shape[:2]
        if len(self.size) > 1:
            sidx = np.random.randint(0, len(self.size))
        else:
            sidx = 0
        size = self.size[sidx]

        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        if self.stride > 1:
            neww = int(np.round(neww / float(self.stride)) * self.stride)
            newh = int(np.round(newh / float(self.stride)) * self.stride)
            assert neww % self.stride == 0
            assert newh % self.stride == 0
        return transform.ResizeTransform(h, w, newh, neww, self.interp)


def box_to_point8(boxes):
    """
    Args:
        boxes: nx4

    Returns:
        (nx4)x2
    """
    b = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]]
    b = b.reshape((-1, 2))
    return b


def point8_to_box(points):
    """
    Args:
        points: (nx4)x2
    Returns:
        nx4 boxes (x1y1x2y2)
    """
    p = points.reshape((-1, 4, 2))
    minxy = p.min(axis=1)   # nx2
    maxxy = p.max(axis=1)   # nx2
    return np.concatenate((minxy, maxxy), axis=1)


def segmentation_to_mask(polys, height, width):
    """
    Convert polygons to binary masks.

    Args:
        polys: a list of nx2 float array

    Returns:
        a binary matrix of (height, width)
    """
    polys = [p.flatten().tolist() for p in polys]
    rles = cocomask.frPyObjects(polys, height, width)
    rle = cocomask.merge(rles)
    return cocomask.decode(rle)


def clip_boxes(boxes, shape):
    """
    Args:
        boxes: (...)x4, float
        shape: h, w
    """
    orig_shape = boxes.shape
    boxes = boxes.reshape([-1, 4])
    h, w = shape
    boxes[:, [0, 1]] = np.maximum(boxes[:, [0, 1]], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], w)
    boxes[:, 3] = np.minimum(boxes[:, 3], h)
    return boxes.reshape(orig_shape)


def filter_boxes_inside_shape(boxes, shape):
    """
    Args:
        boxes: (nx4), float
        shape: (h, w)

    Returns:
        indices: (k, )
        selection: (kx4)
    """
    assert boxes.ndim == 2, boxes.shape
    assert len(shape) == 2, shape
    h, w = shape
    indices = np.where(
        (boxes[:, 0] >= 0) &
        (boxes[:, 1] >= 0) &
        (boxes[:, 2] <= w) &
        (boxes[:, 3] <= h))[0]
    return indices, boxes[indices, :]

