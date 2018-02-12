# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

"""
For handling networks and keep tracks of important parameters, TFLearn is
using Tensorflow collections.
"""


class CollectionKeys(object):
    # Collection for network inputs. Used by `Trainer` class for retrieving all
    # data input placeholders.
    INPUTS = 'tflearn_inputs'
    # Collection for network targets. Used by `Trainer` class for retrieving all
    # targets (labels) placeholders.
    TARGETS = 'tflearn_targets'

    # Collection for network train ops. Used by `Trainer` class for retrieving all
    # optimization processes.
    TRAIN_OPS = 'tflearn_train_ops'

    # Collection to retrieve layers variables. Variables are stored according to
    # the following pattern: /CollectionKeys.LAYER_VARIABLES/layer_name (so there
    # will have as many collections as layers with variables).
    LAYER_VARIABLES = 'tflearn_layer_variables'

    # Collection to store all returned tensors for every layer
    LAYER_TENSOR = 'tflearn_layer_tensor'

    # Collection to store all variables that will be restored
    EXCL_RESTORE_VARS = 'tflearn_restore_variables'

    # Collection to store the default graph configuration
    GRAPH_CONFIG = 'tflearn_graph_config'

    # Collection to store all input variable data preprocessing
    DATA_PREP = 'tflearn_data_preprocessing'

    # Collection to store all input variable data preprocessing
    DATA_AUG = 'tflearn_data_augmentation'

    # Collection to store all custom learning rate variable
    LR_VARIABLES = 'tflearn_lr_variables'

    LOGFILE = 'tflearn_logfile'

