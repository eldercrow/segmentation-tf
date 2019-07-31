from tensorpack.callbacks import Callback
from tensorpack.tfutils.common import get_global_step_var

from .quantize_graph import experimental_create_training_graph, experimental_create_eval_graph

class QuantizationCallback(Callback):
    '''
    Tensorpack callback for quantization.
    '''
    def __init__(self, weight_bits=8, activation_bits=8, quant_delay=0, freeze_bn_delay=0):
        self._wbit = weight_bits
        self._abit = activation_bits
        self._quant_delay = quant_delay
        self._freeze_bn_delay = freeze_bn_delay

    def _setup_graph(self):
        '''
        '''
        experimental_create_training_graph(
          weight_bits=self._wbit,
          activation_bits=self._abit,
          symmetric=True,
          quant_delay=self._quant_delay,
          freeze_bn_delay=self._freeze_bn_delay,
          is_bn_training=True)
          # scope=self.model_scope_quan)
    #
    # def _trigger_step(self):
    #     self.trainer.sess.run(self.mask_update_op)

