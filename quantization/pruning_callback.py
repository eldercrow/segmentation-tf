from tensorpack.callbacks import Callback
from tensorpack.tfutils.common import get_global_step_var
from tensorflow.contrib.model_pruning.python import pruning


class PruningCallback(Callback):
    '''
    Tensorpack callback for pruning.
    '''
    def __init__(self, param_dict):
        '''
        Convert parameter dict to parameter string.
        '''
        self.param_dict = param_dict

    def _setup_graph(self):
        '''
        '''
        default_dict = {
                'name': 'model_pruining',
                'begin_pruning_step': 0,
                'end_pruning_step': 12900,
                'target_sparsity': 0.7,
                'pruning_frequency': 129,
                'sparsity_function_begin_step': 0,
                'sparsity_function_end_step': 12900,
                'sparsity_function_exponent': 2,
                }
        for k, v in self.param_dict.items():
            if k in default_dict:
                default_dict[k] = v

        param_list = ['{}={}'.format(k, v) for k, v in default_dict.items()]
        # param_list = [
        #         "name=cifar10_pruning",
        #         "begin_pruning_step=1000",
        #         "end_pruning_step=20000",
        #         "target_sparsity=0.9",
        #         "sparsity_function_begin_step=1000",
        #         "sparsity_function_end_step=20000"
        # ]

        PRUNE_HPARAMS = ",".join(param_list)
        pruning_hparams = pruning.get_pruning_hparams().parse(PRUNE_HPARAMS)
        self.p = pruning.Pruning(pruning_hparams, global_step=get_global_step_var())
        self.p.add_pruning_summaries()
        self.mask_update_op = self.p.conditional_mask_update_op()

    def _trigger_step(self):
        self.trainer.sess.run(self.mask_update_op)
