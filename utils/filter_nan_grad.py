import tensorflow as tf

from tensorpack.compat import tfv1
from tensorpack.utils import logger
from tensorpack.tfutils.gradproc import GradientProcessor
# from .summary import add_moving_summary
# from .symbolic_functions import print_stat, rms

class FilterNaNGrad(GradientProcessor):
    """
    Skip the update and print a warning (instead of crashing),
    when the gradient of certain variable is None.
    """
    def __init__(self, verbose=True):
        """
        Args:
            verbose (bool): whether to print warning about None gradients.
        """
        super(FilterNaNGrad, self).__init__()

    def _process(self, grads):
        g = []
        for grad, var in grads:
            # import ipdb
            # ipdb.set_trace()
            grad = tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad)
            g.append((grad, var))
        return g
