"""Custom callbacks for tensorflow training"""

import collections

import numpy as np
import pandas as pd
from keras import keras_export
from keras.callbacks import Callback


@keras_export("keras.callbacks.CSVLogger")
class PandasLogger(Callback):
    """Callback that logs a pandas df to a specified model"""

    def __init__(self, tf_model):
        self.keys = None
        self.dict_list = []
        self.tf_model = tf_model
        super(PandasLogger, self).__init__()

    def on_train_begin(self, logs=None):
        # self.csv_file = open(self.tempfile, "w")
        # self.df = pd.DataFrame
        self.tf_model.df_loss = None
        return

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif isinstance(k, collections.abc.Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (", ".join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict((k, logs[k]) if k in logs else (k, "NA") for k in self.keys)

        row_dict = collections.OrderedDict({"epoch": epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.dict_list.append(row_dict)

    def on_train_end(self, logs=None):
        self.tf_model.df_loss_log = pd.DataFrame(self.dict_list)
        self.tf_model = None
