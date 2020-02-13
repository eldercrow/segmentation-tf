import tensorflow as tf

from utils.shape_utils import combined_static_and_dynamic_shape
from config import config as cfg


def icnet_inference(logits, image):
    '''
    logits: (N, H', W', nc), classification logits
    image: (N, H, W, _), image or label, just for shape inference)
    '''
    shape_image = combined_static_and_dynamic_shape(image)
    logits = tf.image.resize_bilinear(logits, shape_image[1:3], align_corners=True)

    if cfg.INFERENCE.SHIFT_PREDICTION == 0:
        output = tf.argmax(logits, axis=-1, name='segmentation_output', output_type=tf.int32)
    else:
        output = tf.argmax(logits, axis=-1, output_type=tf.int32)
        output = tf.add(output, cfg.INFERENCE.SHIFT_PREDICTION, name='segmentation_output')
    return output
