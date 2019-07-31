from tensorpack.tfutils.scope_utils import under_name_scope
from icnet.model import ICNet_BN as ICNet


@under_name_scope()
def icnet_features(images, is_training, num_classes=19):
    '''
    Args:
        hlist: list of three features, [1/8, 1/16, 1/32].
    '''
    layers = ICNet(images, is_training, num_classes).setup()
    k_out = ['conv6_cls', 'sub4_out', 'sub24_out']
    out = { k: layers[k] for k in k_out }
    return out
    # h0, h1, h2 = hlist
    # shape_h0 = combined_static_and_dynamic_shape(h0)
    # shape_h1 = combined_static_and_dynamic_shape(h1)
    #
    # with ssdnet_argscope():
    #     # merge h1 and h2, create 1/16 feature
    #     h2 = tf.depth_to_space(h2, 2)
    #     h12 = tf.concat([h1, h2], axis=-1) # 128
    #     h12 = Conv2D('h12', h12, 256, 1, activation=BNReLU)
    #
    #     with tf.variable_scope('top'):
    #         feat = Conv2D('conv1', h12, 256, 1, activation=BNReLU)
    #     with tf.variable_scope('se'):
    #         s = AvgPooling('avgpool', h12, 49, strides=(16, 20), padding='same')
    #         s = Conv2D('conv1', s, 256, 1, activation=None, use_bias=True)
    #         s = tf.sigmoid(s, name='sigmoid')
    #         s = tf.image.resize_bilinear(s, shape_h1[1:3], align_corners=True)
    #     feat = tf.multiply(feat, s)
    #
    #     feat = tf.image.resize_bilinear(feat, shape_h0[1:3], align_corners=True)
    #     feat = DWConv('convd', feat, 5)
    #     feat_l = Conv2D('conv_h0', h0, 128, 1, activation=BNReLU)
    #
    # with argscope([Conv2D], use_bias=True):
    #     feat = Conv2D('logit_up', feat, num_classes, 1)
    #     feat_l = Conv2D('logit_h0', feat_l, num_classes, 1)
    #
    # out = tf.add(feat, alpha*feat_l, name='cls_logit')
    # return out
