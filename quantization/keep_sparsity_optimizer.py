import tensorflow as tf
from tensorpack.tfutils.optimizer import ProxyOptimizer


class KeepSparsityOptimizer(ProxyOptimizer):
    '''
    '''
    def __init__(self, opt, name='KeepSparsityOptimizer', patterns=['W:0']):
        super(KeepSparsityOptimizer, self).__init__(opt, name)
        self._patterns = patterns

    def _match_pattern(self, var):
        for p in self._patterns:
            if p in var.name:
                return True
        return False

    def compute_gradients(self, *args, **kwargs):
        '''
        Compute gradients, and then zero out for pruned values.
        '''
        grads_and_vars = self._opt.compute_gradients(*args, **kwargs)

        modified_gnv = []
        vars_to_modify = []
        for g, v in grads_and_vars:
            if self._match_pattern(v):
                mg = tf.where(tf.not_equal(v.value(), tf.cast(0, v.value().dtype)), \
                              g,
                              tf.zeros_like(g))
                modified_gnv.append((mg, v))
                vars_to_modify.append(v.name)
        return modified_gnv
