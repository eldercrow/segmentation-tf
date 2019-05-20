import tensorflow as tf

from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.argscope import argscope

from utils.shape_utils import combined_static_and_dynamic_shape

# from config import config as cfg


def focal_loss(labels, logits, gamma=2.0):
	"""
	"""
	y_pred = tf.nn.softmax(logits, dim=-1) # [batch_size, num_classes]
	labels = tf.one_hot(labels, depth=y_pred.shape[1], dtype=y_pred.dtype)

	loss = -labels * ((1 - y_pred) ** gamma) * tf.log(tf.maximum(1e-08, y_pred))
	loss = tf.reduce_sum(loss, axis=1)
	return loss


@under_name_scope()
def aspp_losses(cls_logits, labels, num_classes):
    '''
    Args:
        labels: (H, W) label image
        cls_logits: (H', W', nc) logits

        For now, H' and W' are H/8 and W/8, respectively.
    '''
    # shape_labels = combined_static_and_dynamic_shape(labels)
    # cls_logits = tf.image.resize_bilinear(cls_logits, shape_labels[1:3], align_corners=True)
    shape_logits = combined_static_and_dynamic_shape(cls_logits)
    labels = tf.image.resize_nearest_neighbor(labels, shape_logits[1:3], align_corners=True)

    logits = tf.reshape(cls_logits, [-1, shape_logits[-1]])
    labels = tf.reshape(labels, [-1])

    idx = tf.logical_and(tf.greater_equal(labels, 0), tf.less(labels, num_classes))
    idx = tf.where(idx)[:, 0]

    valid_logits = tf.gather(logits, idx)
    valid_labels = tf.gather(labels, idx)

    cls_loss = focal_loss(labels=valid_labels, logits=valid_logits)
    # cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=valid_labels, logits=valid_logits)
    cls_loss = tf.reduce_mean(cls_loss, name='cls_loss')

    correct = tf.equal(valid_labels, tf.argmax(valid_logits, axis=-1, output_type=valid_labels.dtype))
    acc = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
    add_moving_summary(cls_loss, acc)
    # return the loss
    return cls_loss
