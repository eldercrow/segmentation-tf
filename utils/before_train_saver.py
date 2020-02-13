import os
from datetime import datetime
import tensorflow as tf
from tensorpack import Callback
from tensorpack import logger

class BeforeTrainSaver(Callback):
    '''
    Save the model once, right before the training begins.
    '''

    def __init__(self, checkpoint_dir=None, var_collections=None):
        if var_collections is None:
            var_collections = [tf.GraphKeys.GLOBAL_VARIABLES]

        if not isinstance(var_collections, list):
            var_collections = [var_collections]
        self.var_collections = var_collections
        if checkpoint_dir is None:
            checkpoint_dir = logger.get_logger_dir()
        if checkpoint_dir is not None:
            if not tf.gfile.IsDirectory(checkpoint_dir):
                tf.gfile.MakeDirs(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir

    def _setup_graph(self):
        assert self.checkpoint_dir is not None, \
            "BeforeTrainSaver() doesn't have a valid checkpoint directory."
        vars = []
        for key in self.var_collections:
            vars.extend(tf.get_collection(key))
        vars = list(set(vars))
        self.path = os.path.join(self.checkpoint_dir, 'model')
        self.saver = tf.train.Saver(
            var_list=vars,
            max_to_keep=1,
            keep_checkpoint_every_n_hours=24,
            write_version=tf.train.SaverDef.V2,
            save_relative_paths=True)
        # Scaffold will call saver.build from this collection
        tf.add_to_collection(tf.GraphKeys.SAVERS, self.saver)

    def _before_train(self):
        # graph is finalized, OK to write it now.
        time = datetime.now().strftime('%m%d-%H%M%S')
        self.saver.export_meta_graph(
            os.path.join(self.checkpoint_dir,
                         'graph-{}.meta'.format(time)),
            collection_list=self.graph.get_all_collection_keys())

        try:
            self.saver.save(
                tf.get_default_session(),
                self.path,
                global_step=tf.train.get_global_step(),
                write_meta_graph=False)
            logger.info("Model saved to {}.".format(tf.train.get_checkpoint_state(self.checkpoint_dir).model_checkpoint_path))
        except (OSError, IOError, tf.errors.PermissionDeniedError,
                tf.errors.ResourceExhaustedError):   # disk error sometimes.. just ignore it
            logger.exception("Exception in BeforeTrainSaver!")
        assert False
