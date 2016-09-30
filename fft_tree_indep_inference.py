# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Inference routine for k-cardinality-constrained graphical models.

This implements the "FFT Tree" algorithm from
Tarlow et al., "Fast Exact Inference for Recursive Cardinality Models"
http://www.cs.toronto.edu/~dtarlow/tszaf-fast_cardinality.pdf
"""

from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from shared_util import create_mask
from shared_util import normalize
from shared_util import repeat
from shared_util import sample_categorical


def fft_tree_indep_vars(yp,
                        zp,
                        num_samples,
                        batch_size,
                        max_len,
                        k_constraints=None,
                        lam=0.001):
  """Runs FFT Tree inference on a set of (otherwise) independent variables.

  Args:
    yp: The unary potentials for the binary variables y_i.
        This should have shape [batch_size, num_y].
    zp: The cardinality potential (log-scores for allowable k)
        This should have shape [batch_size, num_y+1].
    num_samples: # of samples to return from inference. Scalar.
    batch_size: batch size.
    max_len: max length of input sequence.
    k_constraints: If the z-potentials have a hard cutoff for some
        k-sparsity, we can be much more numerically stable
        by sparsifying the intermediate messages before normalizing.
    lam: lambda for message-damping, improves numerical stability
        at the cost of some accuracy in the inference/sampling

  Returns:
    marginals: Marginal distribution for each y_i.
               This should have shape [batch_size, num_y].
    samples: A set of samples for each y_i.
             This should have shape [batch_size, num_samples, num_y].
    log_z: Log normalizing constant for distribution.
           This should have shape [batch_size].
  """

  tree_depth = np.ceil(np.log(max_len) / np.log(2.0)).astype(np.int32) + 1
  width = np.power(2, tree_depth).astype(np.int32)

  # have to pad it to always take on zeros on the pad values
  yp_unary_padded = tf.concat(
      1, [yp, tf.ones([batch_size, width / 2 - max_len]) * -100.0])
  yp_stitched = tf.reshape(
      tf.concat(1, [tf.zeros([batch_size * width / 2, 1]),
                    tf.reshape(yp_unary_padded, [-1, 1])]), [batch_size, -1])
  yb_padded = tf.exp(yp_stitched)

  log_z = 0

  yb_padded = tf.concat(0, tf.split(1, width / 2, yb_padded))
  yb_padded, yb_z = normalize(yb_padded)

  yb_padded = tf.concat(1, tf.split(0, width / 2, yb_padded))

  yb_z = tf.concat(1, tf.split(0, width / 2, yb_z))

  with tf.name_scope("initial_log_z"):
    log_z += tf.expand_dims(tf.reduce_sum(tf.log(yb_z), 1), 1)

  upward_msgs = []
  upward_msgs.append(yb_padded)

  prev_msg = yb_padded

  with tf.name_scope("upward_msgs"):
    for d in xrange(1, tree_depth):
      # for lv 0, we want blocks of 4
      block_size = np.power(2, d + 1)
      num_splits = width / block_size
      sub_probs = tf.split(1, num_splits, prev_msg)

      left, right = tf.split(1, 2, tf.concat(0, sub_probs))
      left = tf.pad(left, [[0, 0], [0, block_size / 2]])
      right = tf.pad(right, [[0, 0], [0, block_size / 2]])
      convs = positive_conv(left, right)

      new_msg, new_msg_z = normalize(convs)
      new_msg_z = tf.concat(1, tf.split(0, num_splits, new_msg_z))
      log_z += tf.expand_dims(tf.reduce_sum(tf.log(new_msg_z), 1), 1)

      if k_constraints is not None:
        k_mask = create_mask(
            tf.reshape(
                repeat(num_splits, tf.expand_dims(
                    tf.minimum(2**d + 1, k_constraints + 1), 1)), [-1]),
            block_size)
        new_msg *= k_mask
        k_uni = k_mask / tf.maximum(
            tf.expand_dims(tf.reduce_sum(k_mask, 1), 1), 1.0)
        new_msg = new_msg * (1.0 - lam) + k_uni * lam
      else:
        uni_mask = create_mask(
            tf.reshape(
                repeat(num_splits, tf.tile(
                    tf.expand_dims(tf.reshape(2**d + 1, [-1]), 1),
                    [batch_size, 1])), [-1]), block_size)
        uni = uni_mask / tf.maximum(
            tf.expand_dims(tf.reduce_sum(uni_mask, 1), 1), 1.0)
        new_msg = new_msg * (1.0 - lam) + uni * lam

      new_msg = tf.concat(1, tf.split(0, num_splits, new_msg))
      upward_msgs.append(new_msg)
      prev_msg = new_msg

  with tf.name_scope("k_beliefs"):
    k_pot_msg = tf.pad(tf.exp(zp), [[0, 0], [0, width - max_len - 1]])
    k_pot_msg, k_pot_z = normalize(k_pot_msg)
    log_z += tf.log(k_pot_z)

    all_marginals = []

    k_belief, k_z = normalize(k_pot_msg * prev_msg)
    log_k_z = tf.log(k_z)

  log_z += log_k_z

  all_marginals.append(k_belief)

  samples = []
  sample_indices = []

  with tf.name_scope("k_samples"):
    if num_samples > 0:
      rep_k_b = repeat(num_samples, k_belief)
      k_samples, k_indices = sample_categorical(tf.log(rep_k_b))
      samples.append(k_samples)
      sample_indices.append(k_indices)
      prev_indices = k_indices

  downward_msgs = []
  downward_msgs.append(k_pot_msg)
  prev_msg = k_pot_msg

  with tf.name_scope("bwd_msgs"):
    for d in reversed(xrange(1, tree_depth)):
      up_msgs = upward_msgs[d - 1]
      block_size = np.power(2, d + 1)
      num_splits = width / block_size
      prev_msgs = tf.concat(0, tf.split(1, num_splits, prev_msg))
      left_right_up_msgs = tf.concat(0, tf.split(1, num_splits, up_msgs))
      left_up_msgs, right_up_msgs = tf.split(1, 2, left_right_up_msgs)

      if num_samples > 0:
        reshaped_prev_indices = tf.concat(0, tf.split(1, num_splits,
                                                      prev_indices))
        prev_domain_lens = reshaped_prev_indices + 1
        rep_up_msgs = repeat(num_samples, up_msgs)
        left_up_msgs_repl, right_up_msgs_repl = tf.split(1, 2, tf.concat(
            0, tf.split(1, num_splits, rep_up_msgs)))

        num_rows = num_splits * batch_size * num_samples

        len_mask = create_mask(
            tf.reshape(prev_domain_lens, [-1]), block_size / 2)
        right_up_msgs_repl *= len_mask
        right_up_msgs_repl = tf.pad(right_up_msgs_repl,
                                    [[0, 0], [0, block_size / 2]])
        left_up_msgs_repl *= len_mask
        left_up_msgs_repl = tf.pad(left_up_msgs_repl,
                                   [[0, 0], [0, block_size / 2]])
        left_up_msgs_repl, _ = normalize(left_up_msgs_repl)

        right_up_msgs_repl = tf.reverse_sequence(right_up_msgs_repl, tf.reshape(
            tf.to_int64(prev_domain_lens), [-1]), 1)
        right_up_msgs_repl, _ = normalize(right_up_msgs_repl)

        sliced_factor_marg, _ = normalize(left_up_msgs_repl *
                                          right_up_msgs_repl)
        sliced_factor_marg = tf.slice(sliced_factor_marg, [0, 0],
                                      [num_rows, block_size / 2])

        left_samples, left_indices = sample_categorical(
            tf.log(sliced_factor_marg))
        left_samples = tf.slice(left_samples, [0, 0],
                                [num_rows, block_size / 2])

        right_indices = prev_domain_lens - left_indices - 1
        right_samples = tf.one_hot(
            tf.reshape(right_indices, [-1]), block_size / 2)

        cur_samples = tf.concat(1, tf.split(0, num_splits, tf.concat(
            1, [left_samples, right_samples])))
        cur_indices = tf.concat(1, tf.split(0, num_splits, tf.concat(
            1, [left_indices, right_indices])))
        samples.append(cur_samples)
        sample_indices.append(cur_indices)
        prev_indices = cur_indices

      left_down_msgs = positive_correl(prev_msgs,
                                       tf.pad(right_up_msgs,
                                              [[0, 0], [0, block_size / 2]]))
      left_down_msgs = tf.slice(left_down_msgs, [0, 0],
                                [num_splits * batch_size, block_size / 2])
      left_down_msgs, _ = normalize(left_down_msgs)
      left_margs, _ = normalize(left_up_msgs * left_down_msgs)

      right_down_msgs = positive_correl(prev_msgs,
                                        tf.pad(left_up_msgs,
                                               [[0, 0], [0, block_size / 2]]))
      right_down_msgs = tf.slice(right_down_msgs, [0, 0],
                                 [num_splits * batch_size, block_size / 2])
      right_down_msgs, _ = normalize(right_down_msgs)
      right_margs, _ = normalize(right_up_msgs * right_down_msgs)

      new_msg = tf.concat(1, tf.split(0, num_splits, tf.concat(
          1, [left_down_msgs, right_down_msgs])))
      margs = tf.concat(1, tf.split(0, num_splits,
                                    tf.concat(1, [left_margs, right_margs])))

      all_marginals.append(margs)
      downward_msgs.append(new_msg)
      prev_msg = new_msg

  final_marginals = all_marginals[-1]

  final_unary_marginals = tf.reshape(
      tf.split(1, 2, tf.reshape(final_marginals, [-1, 2]))[1], [batch_size, -1])

  final_samples = samples[-1]
  final_unary_samples = tf.reshape(
      tf.split(1, 2, tf.reshape(final_samples, [-1, 2]))[1],
      [num_samples, batch_size, -1])

  final_unary_samples = tf.transpose(final_unary_samples, [1, 0, 2])

  final_unary_marginals = tf.slice(final_unary_marginals, [0, 0],
                                   [batch_size, max_len])
  final_unary_samples = tf.slice(final_unary_samples, [0, 0, 0],
                                 [batch_size, num_samples, max_len])

  return final_unary_marginals, final_unary_samples, log_z


def positive_conv(a, b):
  """Pairwise convolution on the positive domain of batches of 1-d vectors.

  Args:
    a: discrete function on the positive domain (e.g. real-valued vector
       with a[0] = f(0), etc). Shape of [batch_size, domain_size].
    b: same as a.
  Returns:
    Discrete function on positive domain representing convolution of a and b.

  """
  batch_size = a.get_shape().dims[0].value
  width = a.get_shape().dims[1].value
  a = tf.pad(a, [[0, 0], [width, 0]])
  a = tf.transpose(a)
  b = tf.pad(b, [[0, 0], [width, 0]])
  b = tf.reverse(b, [False, True])
  b = tf.transpose(b)
  reshaped_a = tf.reshape(a, [1, 1, width * 2, batch_size])
  reshaped_b = tf.reshape(b, [1, width * 2, batch_size, 1])
  res = tf.nn.depthwise_conv2d(
      reshaped_a, reshaped_b, strides=[1, 1, 1, 1], padding="SAME")
  res = tf.reshape(tf.transpose(res), [batch_size, width * 2])
  res = tf.slice(res, [0, width], [batch_size, width])
  return res


def positive_correl(a, b):
  """Pairwise cross-correlation of 1-d vectors on the positive domain.

  Args:
    a: discrete function batch on the positive domain (e.g. real-valued vector
       with a[0] = f(0), etc). Shape of [batch_size, domain_size].
    b: same as a.
  Returns:
    Discrete function on positive domain representing correlation of a and b.

  """
  batch_size = a.get_shape().dims[0].value
  width = a.get_shape().dims[1].value
  a = tf.pad(a, [[0, 0], [width - 1, 1]])
  a = tf.transpose(a)
  b = tf.pad(b, [[0, 0], [width - 1, 1]])
  b = tf.transpose(b)
  reshaped_a = tf.reshape(a, [1, 1, width * 2, batch_size])
  reshaped_b = tf.reshape(b, [1, width * 2, batch_size, 1])
  res = tf.nn.depthwise_conv2d(
      reshaped_a, reshaped_b, strides=[1, 1, 1, 1], padding="SAME")
  res = tf.reshape(tf.transpose(res), [batch_size, width * 2])
  res = tf.slice(res, [0, width - 1], [batch_size, width])
  return res
