# -*- coding: utf-8 -*-
# ====================================================================================
# Copyright (C) 2018, All rights reserved.
#
# 工程：tflearn
# 文件: rnn.py
# 创建者：陈云川/chenyunchuan
# 创建时间：2018/2/10
#
# 描述: The purpose of this file is to ...
# ====================================================================================


# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
from ..collections import CollectionKeys
from ..utils import get_logger
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn, bidirectional_dynamic_rnn, rnn_cell_impl as core_rnn_cell
from tensorflow.contrib.rnn import BasicLSTMCell, BasicRNNCell, CoupledInputForgetGateLSTMCell, GRUCell
from .. import config
from .. import utils
from .. import activations
from .. import initializations

logger = get_logger('tflearn.layers.rnn')

# --------------------------
#  RNN Layers
# --------------------------


def _rnn_template(rnn_, rnn_kwargs, name, dropout, return_seq, return_state, reuse):
    """ RNN Layer Template for one direction RNNs. """
    incoming = rnn_kwargs['inputs']
    sequence_length = rnn_kwargs['sequence_length']
    incoming = tf.convert_to_tensor(incoming, dtype=tf.float32)
    input_shape = utils.get_incoming_shape(incoming)
    ndim = len(input_shape)
    scope = rnn_kwargs['scope']

    assert ndim >= 3, "Input dim should be at least 3."
    if 'time_major ' in rnn_kwargs:
        assert rnn_kwargs['time_major'] is False, "The leading dimension must be batches, not time"
    else:
        rnn_kwargs['time_major'] = False

    with tf.variable_scope(scope, default_name=name, values=[incoming, sequence_length], reuse=reuse) as scope:
        name = scope.name
        # Apply dropout
        if dropout:
            if type(dropout) in [tuple, list]:
                in_keep_prob = dropout[0]
                out_keep_prob = dropout[1]
            elif isinstance(dropout, float):
                in_keep_prob, out_keep_prob = dropout, dropout
            else:
                raise Exception("Invalid dropout type (must be a 2-D tuple of float)")

            rnn_kwargs['cell'] = DropoutWrapper(rnn_kwargs['cell'], in_keep_prob, out_keep_prob)

        rnn_kwargs['inputs'] = incoming
        outputs, state = rnn_(**rnn_kwargs)

        if isinstance(outputs, (tuple, list)):
            outputs = tf.concat(outputs, 2)
            state = tf.concat(state, 1)

        if sequence_length is None:
            last_time_activation = get_timestep_features(outputs, None)
        else:
            last_time_activation = get_timestep_features(outputs, sequence_length - 1)
        if return_seq:
            o = outputs
            # propagate seq_length
            o.seq_length = incoming.seq_length
        else:
            o = last_time_activation

        # Retrieve RNN Variables
        c = CollectionKeys.LAYER_VARIABLES + '/' + scope.name
        for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name):
            tf.add_to_collection(c, v)

        # Track output tensor.
        tf.add_to_collection(CollectionKeys.LAYER_TENSOR + '/' + name, o)
        if return_state:
            tf.add_to_collection(CollectionKeys.LAYER_TENSOR + '/' + name, state)

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, o)

        o.scope = scope
        W, b = tf.get_collection(CollectionKeys.LAYER_VARIABLES + '/' + name)
        o.W = W
        o.b = b
        if not return_state:
            o.seq_length = rnn_kwargs['sequence_length']

        return (o, state) if return_state else o


def rnn(incoming, sequence_length, cell, name, dropout=None, return_seq=False,
        return_state=False, reuse=False, scope=None, **rnn_kwargs):
    _rnn_kwargs = {
        "cell": None,
        "inputs": None,
        "sequence_length": None,
        "initial_state": None,
        "dtype": tf.float32,
        "parallel_iterations": None,
        "swap_memory": False,
        "time_major": False,
        "scope": None
    }
    # _rnn_kwargs.update(rnn_kwargs)
    update_fixed_key_dict(_rnn_kwargs, rnn_kwargs)
    _rnn_kwargs['inputs'] = incoming
    _rnn_kwargs['sequence_length'] = sequence_length
    _rnn_kwargs['cell'] = cell
    _rnn_kwargs['scope'] = scope
    # signature:  _rnn_template(rnn_,  rnn_kwargs, name, dropout, return_seq, return_state, reuse)
    return _rnn_template(dynamic_rnn, _rnn_kwargs, name, dropout, return_seq, return_state, reuse)


def bidirectional_rnn(incoming, sequence_length, cell_fw, cell_bw, name, dropout=None, return_seq=False,
                      return_state=False, reuse=False, scope=None, **rnn_kwargs):
    _rnn_kwargs = {
        "cell_fw": None,
        "cell_bw": None,
        "inputs": None,
        "sequence_length": None,
        "initial_state_fw": None,
        "initial_state_bw": None,
        "dtype": tf.float32,
        "parallel_iterations": None,
        "swap_memory": False,
        "time_major": False,
        "scope": None
    }
    # _rnn_kwargs.update(rnn_kwargs)
    update_fixed_key_dict(_rnn_kwargs, rnn_kwargs)
    _rnn_kwargs['inputs'] = incoming
    _rnn_kwargs['sequence_length'] = sequence_length
    _rnn_kwargs['cell_fw'] = cell_fw
    _rnn_kwargs['cell_bw'] = cell_bw
    _rnn_kwargs['scope'] = scope
    # function signature:       _rnn_template(rnn_,  rnn_kwargs, name, dropout, return_seq, return_state, reuse)
    return _rnn_template(bidirectional_dynamic_rnn, _rnn_kwargs, name, dropout, return_seq, return_state, reuse)


def simple_rnn(incoming, sequence_length, num_units, activation='sigmoid', dropout=None, return_seq=False,
               return_state=False, reuse=False, scope=None, name="SimpleRNN", **rnn_kwargs):
    """ Simple RNN.

    Simple Recurrent Layer.

    Input:
        3-D Tensor [samples, timesteps, input dim].
        1-D Tensor [samples] or None

    Output:
        if `return_seq`: 3-D Tensor [samples, timesteps, output dim].
        else: 2-D Tensor [samples, output dim].

    Arguments:
        incoming: `Tensor`. Incoming 3-D Tensor.
        sequence_length: `Tensor`, 1-D with shape [samples], true sequence lengths.
        num_units: `int`, number of units for this layer.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'sigmoid'.
        dropout: `tuple` of `float`: (input_keep_prob, output_keep_prob). The
            input and output keep probability.
        return_seq: `bool`. If True, returns the full sequence instead of
            last sequence output only.
        return_state: `bool`. If True, returns a tuple with output and
            states: (output, states).
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: `str`. A name for this layer (optional).

    """

    _activation = activations.get(activation)

    cell = BasicRNNCell(num_units, activation=_activation, reuse=reuse)
    return rnn(incoming, sequence_length, cell, name, dropout, return_seq, return_state, reuse, scope, **rnn_kwargs)


def lstm(incoming, sequence_length, num_units, activation='tanh', forget_bias=1.0, dropout=None,
         return_seq=False, return_state=False, reuse=False, scope=None, name="LSTM", **rnn_kwargs):
    """ LSTM.

    Long Short Term Memory Recurrent Layer.

    Input:
        3-D Tensor [samples, timesteps, input dim].

    Output:
        if `return_seq`: 3-D Tensor [samples, timesteps, output dim].
        else: 2-D Tensor [samples, output dim].

    Arguments:
        incoming: `Tensor`. Incoming 3-D Tensor.
        sequence_length: `Tensor`, 1-D with shape [samples], true sequence lengths.
        num_units: `int`, number of units for this layer.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'tanh'.
        dropout: `tuple` of `float`: (input_keep_prob, output_keep_prob). The
            input and output keep probability.
        forget_bias: `float`. Bias of the forget gate. Default: 1.0.
        return_seq: `bool`. If True, returns the full sequence instead of
            last sequence output only.
        return_state: `bool`. If True, returns a tuple with output and
            states: (output, states).
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: `str`. A name for this layer (optional).

    References:
        Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber,
        Neural Computation 9(8): 1735-1780, 1997.

    Links:
        [http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf]
        (http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)

    """
    _activation = activations.get(activation)
    cell = BasicLSTMCell(num_units, forget_bias, True, _activation, reuse)
    return rnn(incoming, sequence_length, cell, name, dropout, return_seq, return_state, reuse, scope, **rnn_kwargs)


def gru(incoming, sequence_length, num_units, activation='tanh', weights_init=None,
        dropout=None, return_seq=False, return_state=False, reuse=False, scope=None, name="GRU", **rnn_kwargs):
    """ GRU.

    Gated Recurrent Unit Layer.

    Input:
        3-D Tensor Layer [samples, timesteps, input dim].

    Output:
        if `return_seq`: 3-D Tensor [samples, timesteps, output dim].
        else: 2-D Tensor [samples, output dim].

    Arguments:
        incoming: `Tensor`. Incoming 3-D Tensor.
        sequence_length: `Tensor`. lengths of incoming sequences, 2-D Tensor.
        num_units: `int`, number of units for this layer.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'tanh'.
        dropout: `tuple` of `float`: (input_keep_prob, output_keep_prob). The
            input and output keep probability.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (See tflearn.initializations).
        return_seq: `bool`. If True, returns the full sequence instead of
            last sequence output only.
        return_state: `bool`. If True, returns a tuple with output and
            states: (output, states).
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: `str`. A name for this layer (optional).

    References:
        Learning Phrase Representations using RNN Encoder–Decoder for
        Statistical Machine Translation, K. Cho et al., 2014.

    Links:
        [http://arxiv.org/abs/1406.1078](http://arxiv.org/abs/1406.1078)

    """
    _activation = activations.get(activation)
    weights_init = initializations.get(weights_init)()
    cell = GRUCell(num_units, _activation, reuse, weights_init)
    return rnn(incoming, sequence_length, cell, name, dropout, return_seq, return_state, reuse, scope, **rnn_kwargs)


def coupled_input_forget_gate_lstm(incoming, sequence_length,
                                   num_units, use_peepholes=False, weights_init=None,
                                   num_proj=None, proj_clip=None,
                                   num_unit_shards=1, num_proj_shards=1,
                                   forget_bias=1.0, activation='tanh',
                                   dropout=None, return_seq=False, return_state=False,
                                   reuse=False, scope=None, name="GRU", **rnn_kwargs):

    _activation = activations.get(activation)
    weights_init = initializations.get(weights_init)()
    cell = CoupledInputForgetGateLSTMCell(num_units, use_peepholes, weights_init,
                                          num_proj, proj_clip,
                                          num_unit_shards, num_proj_shards,
                                          forget_bias, True, _activation, reuse)
    return rnn(incoming, sequence_length, cell, name, dropout, return_seq, return_state, reuse, scope, **rnn_kwargs)


class DropoutWrapper(core_rnn_cell.RNNCell):
    """Operator adding dropout to inputs and outputs of the given cell."""

    def __init__(self, cell, input_keep_prob=1.0, output_keep_prob=1.0,
                 seed=None):
        """Create a cell with added input and/or output dropout.

        Dropout is never used on the state.

        Arguments:
          cell: an RNNCell, a projection to output_size is added to it.
          input_keep_prob: unit Tensor or float between 0 and 1, input keep
            probability; if it is float and 1, no input dropout will be added.
          output_keep_prob: unit Tensor or float between 0 and 1, output keep
            probability; if it is float and 1, no output dropout will be added.
          seed: (optional) integer, the randomness seed.

        Raises:
          TypeError: if cell is not an RNNCell.
          ValueError: if keep_prob is not between 0 and 1.
        """
        # super().__init__()

        if not isinstance(cell, core_rnn_cell.RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")
        if isinstance(input_keep_prob, float) and not (0.0 <= input_keep_prob <= 1.0):
            raise ValueError("Parameter input_keep_prob must be between 0 and 1: %d" % input_keep_prob)
        if isinstance(output_keep_prob, float) and not (0.0 <= output_keep_prob <= 1.0):
            raise ValueError("Parameter output_keep_prob must be between 0 and 1: %d" % output_keep_prob)
        self._cell = cell
        self._input_keep_prob = input_keep_prob
        self._output_keep_prob = output_keep_prob
        self._seed = seed

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        """Run the cell with the declared dropouts."""

        is_training = config.get_training_mode()

        if not isinstance(self._input_keep_prob, float) or self._input_keep_prob < 1:
            inputs = tf.cond(is_training,
                             lambda: tf.nn.dropout(inputs, self._input_keep_prob, seed=self._seed),
                             lambda: inputs)
        output, new_state = self._cell(inputs, state)
        if not isinstance(self._output_keep_prob, float) or self._output_keep_prob < 1:
            output = tf.cond(is_training,
                             lambda: tf.nn.dropout(output, self._output_keep_prob, seed=self._seed),
                             lambda: output)
        return output, new_state


# --------------------
#   TensorFlow Utils
# --------------------
def retrieve_seq_length_op(lengths, data):
    """
    Retrieve sequence lengths from inputs.

    Input:
        1-D Tensor [samples] or None. The true lengths of each sequences.
        n-D Tensor [samples, d1, d2, ...], sequences of features.

    Output:
        1-D Tensor [samples].

    Arguments:
        data: Incoming n-D Tensor.
        lengths: The true lengths of the incoming tensor represented sequences.
                 if `None` is specified, then it is assumes that the lengths are the maximum length,
                 i.e., tf.shape(incoming)[2]
    """
    if isinstance(lengths, tf.Tensor) or lengths is None:
        return lengths
    elif hasattr(data, 'seq_length'):
        return data.seq_length
    else:
        # assumed to be full sequences for 3-D tensors.
        return None


def get_timestep_features(input_, time_index, validate_indices=None):
    """
    Slice the last timestep features. Default (time_index=None), get the last timestep.
    """
    if time_index is None:
        return input_[:, -1, ...]
    assert isinstance(time_index, (tf.Tensor, np.ndarray, list, tuple))
    tf_shape = tf.shape(input_)
    batch_size = tf_shape[0]
    max_timesteps = tf_shape[1]
    tail_size = tf.reduce_prod(tf_shape[2:])
    o = tf.reshape(input_, [batch_size*max_timesteps, tail_size])
    idxes = tf.range(batch_size) * max_timesteps + time_index
    o = tf.gather(o, idxes, axis=0)
    new_shape = tf.concat([tf_shape[:1], tf_shape[2:]], axis=0)
    return tf.reshape(o, new_shape)


def update_fixed_key_dict(to_dict, from_dict):
    for key in from_dict:
        if key not in to_dict:
            logger.warning('got key not in destination dict: {}'.format(key))
        else:
            to_dict[key] = from_dict[key]
    return to_dict
