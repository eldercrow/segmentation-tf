import tensorflow as tf

from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.argscope import argscope

from utils.shape_utils import combined_static_and_dynamic_shape

# from config import config as cfg


@under_name_scope()
def icnet_losses(cls_logits, labels, num_classes):
    '''
    Args:
        cls_logits: dict of {name: (logit tensor, weight)}
                    each tensor has the shape of (H', W', nc)
        labels: (H, W) label image
    '''
    def _compute_loss(logits, labels, num_classes):
        # shape_labels = combined_static_and_dynamic_shape(labels)
        # cls_logits = tf.image.resize_bilinear(cls_logits, shape_labels[1:3], align_corners=True)
        shape_logits = combined_static_and_dynamic_shape(logits)
        labels = tf.image.resize_nearest_neighbor(labels, shape_logits[1:3], align_corners=True)

        logits = tf.reshape(logits, [-1, shape_logits[-1]])
        labels = tf.reshape(labels, [-1])

        idx = tf.logical_and(tf.greater_equal(labels, 0), tf.less(labels, num_classes))
        idx = tf.where(idx)[:, 0]

        valid_logits = tf.gather(logits, idx)
        valid_labels = tf.gather(labels, idx)

        # cls_loss = focal_loss(labels=valid_labels, logits=valid_logits)
        cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=valid_labels, logits=valid_logits)

        correct = tf.equal(valid_labels, tf.argmax(valid_logits, axis=-1, output_type=valid_labels.dtype))
        acc = tf.cast(correct, tf.float32)
        return tf.reduce_mean(cls_loss), tf.reduce_mean(acc)

    total_loss = 0.
    for k, v in cls_logits.items():
        loss, acc = _compute_loss(v[0], labels, num_classes)
        loss = tf.multiply(loss, v[1], name='cls_loss_{}'.format(k))
        acc = tf.identity(acc, name='acc_{}'.format(k))
        add_moving_summary(loss, acc)
        total_loss += loss

    # return the loss
    return total_loss
