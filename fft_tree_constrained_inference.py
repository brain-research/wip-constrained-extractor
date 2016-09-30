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

"""Inference routine for k-cardinality- and tree-constrained graphical models.

WARNING: This code is very experimental and if you choose to use it, you should
  be very sure that's what you want.

This implements a tree-knapsack marginal inference and sampling algorithm
based on the "major extension" model of Tarlow et al.,
"Fast Exact Inference for Recursive Cardinality Models"
http://www.cs.toronto.edu/~dtarlow/tszaf-fast_cardinality.pdf

This supports inference over a set of binary token variables, each associated
to a token span. The spans are arranged hierarchically in a tree with the
constraint that each span can only be active if its parent span is also active.
Additionally, all token variables are subject to a k-cardinality constraint.
"""

from __future__ import absolute_import
# this will break a lot of stuff with tf.pad, etc. if uncommented
# from __future__ import division
from __future__ import print_function

import math

import numpy as np
import tensorflow as tf

import data
import discourse_tree as d_t
import fft_tree_indep_inference as ffttii
import model_base
import shared_util as su

# use 16bit ints instead of 32bit for now because of numerical issues
tf_int8 = tf.int8
tf_int16 = tf.int32
tf_int32 = tf.int32
np_int8 = np.int8
np_int16 = np.int32
np_int32 = np.int32
tf_real = tf.float32


def padded_gather_nd(params, indices, r, idx_rank):
  """Version of gather_nd that supports gradients and blank indices.

  Works like gather_nd, but if an index is given as -1, a 0 will be inserted
  in that spot in the output tensor.

  Args:
    params: tensor from which to gather (see gather_nd).
    indices: tensor of indices (see gather_nd).
    r: rank of params
    idx_rank: rank of indices

  Returns:
    result: tensor shaped like indices containing things gathered from params
  """

  # treats -1 indices as always gathering zeros
  # pad 0 onto beginning of final dim of params
  broadcasted_shift = tf.reshape(
      tf.one_hot(
          [r - 1], r, dtype=tf.int32), [1] * (idx_rank - 1) + [r])
  shifted_idx = indices + broadcasted_shift
  # unused indices might contain garbage, just 0 this out
  shifted_idx = tf.maximum(0, shifted_idx)
  padded_params = tf.pad(params, [[0, 0]] * (r - 1) + [[1, 0]])

  # no gather_nd for now because gradient doesn't work
  #   return tf.gather_nd(padded_params,shifted_idx)

  # HACK: work around lack of gradient for gather_nd
  # params has shape of rank r
  # indices has shape of rank idx_rank
  params_shape = [d.value for d in padded_params.get_shape()]
  idx_shape = [d.value for d in shifted_idx.get_shape()]
  flat_params_x_size = 1
  for dim in params_shape:
    flat_params_x_size *= dim
  flat_idx_x_size = 1
  for dim in idx_shape[:-1]:
    flat_idx_x_size *= dim

  index_strides = tf.concat(
      0, [tf.cumprod(
          params_shape[1:], reverse=True), [1]])
  index_strides = tf.reshape(index_strides, [1] * (idx_rank - 1) + [-1])
  flat_idx = tf.reduce_sum(shifted_idx * index_strides, idx_rank - 1)
  flat_idx = tf.reshape(flat_idx, [flat_idx_x_size])
  flat_params = tf.reshape(padded_params, [flat_params_x_size])

  result = tf.gather(flat_params, flat_idx)
  result = tf.reshape(result, idx_shape[:-1])

  return result


def to_int8(t):
  if isinstance(t, list):
    return [tf.cast(tt, tf_int8) for tt in t]
  return tf.cast(t, tf_int8)


def to_int16(t):
  if isinstance(t, list):
    return [tf.cast(tt, tf_int16) for tt in t]
  return tf.cast(t, tf_int16)


def to_int32(t):
  if isinstance(t, list):
    return [tf.cast(tt, tf_int32) for tt in t]
  return tf.cast(t, tf_int32)


class TreeConstrainedInferencer(object):
  """Graph to perform tree-constrained inference."""

  def do_tree_inference(self, hps, tree_inference_inputs, word_logits):
    """Perform tree-constrained inference on token inputs.

    Args:
      hps: bag of hyperparameters.
      tree_inference_inputs: a TreeInferenceInputs graph containing placeholders
        for the various tensors describing a batch of differently shaped trees,
        and the graph edges and nodes involved in message-passing inference.
      word_logits: batch of scores for each word token.

    Returns:
      tok_marg: batch of node-marginals for each binary token indicator.
      tok_samples: batch of samples for each binary token indicator.
      log_z: batch of log-partition function values for each graphical model.
    """

    word_logits_padded = tf.pad(word_logits, tf.pack(
        [[0, 0],
         [0, hps.num_art_steps - tree_inference_inputs.article_max_len]]))
    word_logits_padded.set_shape([hps.batch_size, hps.num_art_steps])

    logit_mask = su.create_log_mask(
        tf.maximum(1, tree_inference_inputs.article_len), hps.num_art_steps)
    masked_word_logits = word_logits_padded + logit_mask

    # get "integrated logits" to compute fast span sums
    integrated_logits = tf.pad(tf.cumsum(
        masked_word_logits, axis=1), [[0, 0], [1, 0]])

    span_start_idx = tree_inference_inputs.span_start_idx

    span_end_idx = tree_inference_inputs.span_end_idx

    # expand dim since we are gathering only 1 thing
    span_integrated_scores_start = padded_gather_nd(
        integrated_logits, tf.expand_dims(span_start_idx, 2), 2, 4)

    span_integrated_scores_end = padded_gather_nd(
        integrated_logits, tf.expand_dims(span_end_idx, 2), 2, 4)

    span_scores = span_integrated_scores_end - span_integrated_scores_start

    span_scores = tf.reshape(span_scores, [hps.batch_size, hps.max_num_spans])

    span_marg, span_samples, log_z = self.do_span_inference(
        hps, tree_inference_inputs, span_scores)

    # grab the span margs and samples out into the token margs and samples

    # expand dim since we are gathering only 1 thing
    span_idx_for_tok_marg = tf.expand_dims(
        tree_inference_inputs.span_idx_for_tok_marg, 2)
    span_idx_for_tok_samples = replicate_samples_2(span_idx_for_tok_marg,
                                                   hps.num_samples)
    span_idx_for_tok_samples = add_leading_idx_2(span_idx_for_tok_samples)
    span_idx_for_tok_marg = add_leading_idx_1(span_idx_for_tok_marg)

    tok_marg = padded_gather_nd(span_marg, span_idx_for_tok_marg, 2, 4)

    tok_samples = padded_gather_nd(span_samples, span_idx_for_tok_samples, 3, 5)

    return tok_marg, tok_samples, log_z

  def do_span_inference(self, hps, tree_inference_inputs, span_scores):
    """Perform tree-constrained inference on span inputs.

    Args:
      hps: bag of hyperparameters.
      tree_inference_inputs: a TreeInferenceInputs graph containing placeholders
        for the various tensors describing a batch of differently shaped trees,
        and the graph edges and nodes involved in message-passing inference.
      span_scores: batch of scores for each span of tokens in the parse.

    Returns:
      span_marg: batch of node-marginals for each binary span indicator.
      span_samples: batch of samples for each binary span indicator.
      log_z: batch of log-partition function values for each graphical model.
    """

    # start from the bottom
    init_span_beliefs = tf.exp(span_scores)

    max_depth = len(hps.tree_widths_at_level)

    # add extra indices for gather_nd-ing and tile out samples
    span_off_to_node_msg = tree_inference_inputs.span_off_to_node_msg
    span_on_to_node_msg = tree_inference_inputs.span_on_to_node_msg
    span_belief_to_node_idx = add_all_leading_idx_1(
        to_int32(tree_inference_inputs.span_belief_to_node_idx))

    nodes_up_to_sum_tree_idx = add_all_leading_idx_2(
        to_int32(tree_inference_inputs.nodes_up_to_sum_tree_idx))
    nodes_up_to_sum_tree_log_z_idx = add_all_leading_idx_2(
        to_int32(tree_inference_inputs.nodes_up_to_sum_tree_log_z_idx))
    sum_tree_msg_start_depths = to_int32(
        tree_inference_inputs.sum_tree_msg_start_depths)
    sum_tree_msg_end_depths = to_int32(
        tree_inference_inputs.sum_tree_msg_end_depths)
    sum_tree_up_to_parent_idx = add_all_leading_idx_2(
        to_int32(tree_inference_inputs.sum_tree_up_to_parent_idx))
    sum_tree_up_to_parent_log_z_idx = add_all_leading_idx_2(
        to_int32(tree_inference_inputs.sum_tree_up_to_parent_log_z_idx))

    sum_tree_down_to_nodes_idx = add_all_leading_idx_2(
        to_int32(tree_inference_inputs.sum_tree_down_to_nodes_idx))
    node_to_span_off_belief_idx = add_all_leading_idx_2(
        to_int32(tree_inference_inputs.node_to_span_off_belief_idx))

    node_to_span_on_belief_range_idx = to_int32(
        tree_inference_inputs.node_to_span_on_belief_range_idx)

    node_to_span_on_belief_start_idx = []
    node_to_span_on_belief_end_idx = []
    for nsbrgi in node_to_span_on_belief_range_idx:
      starts, ends = tf.split(3, 2, nsbrgi)
      node_to_span_on_belief_start_idx.append(starts)
      node_to_span_on_belief_end_idx.append(ends)

    node_to_span_on_belief_start_idx = add_all_leading_idx_2(
        node_to_span_on_belief_start_idx)
    node_to_span_on_belief_end_idx = add_all_leading_idx_2(
        node_to_span_on_belief_end_idx)

    parent_on_down_to_sum_tree_idx = add_all_leading_idx_2(
        to_int32(tree_inference_inputs.parent_on_down_to_sum_tree_idx))
    parent_off_down_to_sum_tree_idx = add_all_leading_idx_2(
        to_int32(tree_inference_inputs.parent_off_down_to_sum_tree_idx))

    global_sum_tree_msg_start_depths = to_int32(
        tree_inference_inputs.global_sum_tree_msg_start_depths)
    global_sum_tree_msg_end_depths = to_int32(
        tree_inference_inputs.global_sum_tree_msg_end_depths)

    node_up_to_global_idx = add_leading_idx_1(
        to_int32(tree_inference_inputs.node_up_to_global_idx))
    node_up_to_global_log_z_idx = add_leading_idx_1(
        to_int32(tree_inference_inputs.node_up_to_global_log_z_idx))
    global_down_to_node_idx = add_leading_idx_1(
        to_int32(tree_inference_inputs.global_down_to_node_idx))

    span_off_belief_to_span_off_marginal_idx = add_leading_idx_1(
        to_int32(
            tree_inference_inputs.span_off_belief_to_span_off_marginal_idx))

    node_sample_to_span_off_belief_sample_idx = add_all_leading_idx_3(
        replicate_all_samples_3(
            to_int32(tree_inference_inputs.node_to_span_off_belief_idx),
            hps.num_samples))
    parent_on_sample_down_to_sum_tree_idx = add_all_leading_idx_3(
        replicate_all_samples_3(
            to_int32(tree_inference_inputs.parent_on_down_to_sum_tree_idx),
            hps.num_samples))
    parent_off_sample_down_to_sum_tree_idx = add_all_leading_idx_3(
        replicate_all_samples_3(
            to_int32(tree_inference_inputs.parent_off_down_to_sum_tree_idx),
            hps.num_samples))
    sum_tree_sample_down_to_nodes_idx = add_all_leading_idx_3(
        replicate_all_samples_3(
            to_int32(tree_inference_inputs.sum_tree_down_to_nodes_idx),
            hps.num_samples))

    global_down_to_node_sample_idx = add_leading_idx_2(
        replicate_samples_2(
            to_int32(tree_inference_inputs.global_down_to_node_idx),
            hps.num_samples))

    span_sample_off_belief_to_span_sample_off_marginal_idx = add_leading_idx_2(
        replicate_samples_2(
            to_int32(
                tree_inference_inputs.span_off_belief_to_span_off_marginal_idx),
            hps.num_samples))

    # Handle base case of forward inference pass

    init_node_beliefs = (
        span_off_to_node_msg[0] + span_on_to_node_msg[0] * padded_gather_nd(
            init_span_beliefs, span_belief_to_node_idx[0], 2, 4))

    cur_node_out_up_msg = init_node_beliefs

    cur_node_out_log_zs = tf.to_float(tf.zeros_like(cur_node_out_up_msg))

    sum_tree_layers = []
    parent_constraint_layers = []

    for d in xrange(1, max_depth):

      # Sum up siblings at current level

      cur_fft_tree_width = hps.fft_tree_widths_at_level[max_depth - d - 1]
      cur_sum_tree_msg_start_depths = sum_tree_msg_start_depths[d - 1]
      cur_sum_tree_msg_end_depths = sum_tree_msg_end_depths[d - 1]

      cur_sum_tree_layer = SiblingSumTreeLayer(
          hps.batch_size,
          hps.max_num_sentences,
          hps.num_samples,
          cur_fft_tree_width,
          cur_sum_tree_msg_start_depths,
          cur_sum_tree_msg_end_depths,
          layer_depth=str(d),
          message_damp_lambda=hps.fft_tree_msg_damp_lambdas[max_depth - d - 1])

      cur_sum_tree_inc_up_msg = padded_gather_nd(
          cur_node_out_up_msg, nodes_up_to_sum_tree_idx[d - 1], 3, 4)
      cur_sum_tree_inc_log_zs = padded_gather_nd(
          cur_node_out_log_zs, nodes_up_to_sum_tree_log_z_idx[d - 1], 3, 4)

      # pylint: disable=line-too-long
      sum_tree_out_up_msg, sum_tree_out_log_zs = cur_sum_tree_layer.compute_up_msg(
          cur_sum_tree_inc_up_msg, cur_sum_tree_inc_log_zs)
      # pylint: enable=line-too-long

      sum_tree_layers.append(cur_sum_tree_layer)

      # Apply parent constraints at current level

      cur_parent_constraint_layer = ParentConstraintLayer(
          init_span_beliefs, layer_depth=str(d))

      parent_constraint_layers.append(cur_parent_constraint_layer)

      # pylint: disable=line-too-long
      cur_node_out_up_msg, cur_node_out_log_zs = cur_parent_constraint_layer.compute_up_msg(
          sum_tree_out_up_msg, sum_tree_up_to_parent_idx[d - 1],
          sum_tree_up_to_parent_log_z_idx[d - 1], span_belief_to_node_idx[d],
          span_off_to_node_msg[d], span_on_to_node_msg[d], sum_tree_out_log_zs)
      # pylint: enable=line-too-long

    # Sentence-level forward pass done
    # now we have the upward messages for each sentence
    up_msg_by_sentence = cur_node_out_up_msg
    up_log_zs_by_sentence = cur_node_out_log_zs

    # Sum up the running log z across sentences and give an extra dim for
    # the global "sentence"
    # Since sum-trees expect a "sentence" dimension
    global_up_sum_tree_inc_log_zs = padded_gather_nd(
        up_log_zs_by_sentence, node_up_to_global_log_z_idx, 3, 3)

    global_up_sum_tree_inc_msg = padded_gather_nd(up_msg_by_sentence,
                                                  node_up_to_global_idx, 3, 3)

    if not hps.single_sentence_concat:
      global_sum_tree_layer = SiblingSumTreeLayer(
          hps.batch_size,
          1,
          hps.num_samples,
          hps.global_fft_tree_width,
          global_sum_tree_msg_start_depths,
          global_sum_tree_msg_end_depths,
          layer_depth="global")

      # pylint: disable=line-too-long
      global_sum_tree_out_up_msg, global_sum_tree_out_up_log_zs = global_sum_tree_layer.compute_up_msg(
          global_up_sum_tree_inc_msg, global_up_sum_tree_inc_log_zs)
      # pylint: enable=line-too-long

      global_sum_tree_out_up_msg = tf.reshape(global_sum_tree_out_up_msg,
                                              [hps.batch_size, -1])

      running_log_zs = tf.reshape(global_sum_tree_out_up_log_zs,
                                  [hps.batch_size, -1])
    else:
      running_log_zs = tf.reshape(global_up_sum_tree_inc_log_zs,
                                  [hps.batch_size, -1])
      global_sum_tree_out_up_msg = global_up_sum_tree_inc_msg

    running_log_z = tf.reshape(running_log_zs, [hps.batch_size, -1])
    running_log_z = tf.reshape(
        tf.slice(running_log_z, [0, 0], [hps.batch_size, 1]),
        [hps.batch_size, 1])

    with tf.name_scope("k_beliefs"):

      # TODO(lvilnis@) insert different type of cardinality potential here!
      # this might not encourage long enough summaries.
      if hps.single_sentence_concat:
        k_pot_msg = su.create_mask(tree_inference_inputs.abstract_len + 1,
                                   hps.global_fft_tree_width)
        k_pot_msg -= su.create_mask(tree_inference_inputs.abstract_len * 0 + 1,
                                    hps.global_fft_tree_width)
      else:
        k_pot_msg = su.create_mask(tree_inference_inputs.abstract_len,
                                   hps.global_fft_tree_width)

      k_belief, _, log_k_z = su.normalize_and_log(k_pot_msg *
                                                  global_sum_tree_out_up_msg)

      running_log_z += log_k_z

      running_log_z = tf.reshape(running_log_z, [hps.batch_size])

    # with cardinality beliefs, start downward message passing
    with tf.name_scope("k_samples"):

      rep_k_b = su.repeat(hps.num_samples, k_belief)
      rep_k_b = tf.reshape(rep_k_b, [hps.num_samples, hps.batch_size, -1])
      rep_k_b = tf.transpose(rep_k_b, [1, 0, 2])
      rep_k_b = tf.reshape(rep_k_b, [hps.batch_size * hps.num_samples, -1])
      k_samples, _ = su.sample_categorical(tf.log(rep_k_b))
      k_samples = tf.reshape(k_samples, [hps.batch_size, hps.num_samples, -1])

    # compute the global samples and global downward messages

    self.k_samples = k_samples

    if not hps.single_sentence_concat:
      k_pot_msg = tf.reshape(k_pot_msg, [hps.batch_size, 1, -1])
      k_samples = tf.reshape(k_samples,
                             [hps.batch_size, 1, hps.num_samples, -1])

      global_sum_tree_down_msgs, _ = global_sum_tree_layer.compute_down_msg(
          k_pot_msg)
      global_sum_tree_down_samples = global_sum_tree_layer.compute_down_samples(
          k_samples)

      global_sum_tree_down_msgs = tf.reshape(global_sum_tree_down_msgs,
                                             [hps.batch_size, -1])
      global_sum_tree_down_samples = tf.reshape(
          global_sum_tree_down_samples, [hps.batch_size, hps.num_samples, -1])

    else:
      global_sum_tree_down_msgs = k_pot_msg
      global_sum_tree_down_samples = k_samples

    # now we gather down to per-sentence messages and samples

    cur_node_inc_down_msg = padded_gather_nd(global_sum_tree_down_msgs,
                                             global_down_to_node_idx, 2, 4)

    cur_node_inc_down_samples = padded_gather_nd(global_sum_tree_down_samples,
                                                 global_down_to_node_sample_idx,
                                                 3, 5)

    # This is stored as [batch,sample,sentence,node] so we need to transpose

    cur_node_inc_down_samples = tf.transpose(cur_node_inc_down_samples,
                                             [0, 2, 1, 3])

    # stores [batch,sentence,width] span_off_marginals for each depth
    all_span_off_marginals = []
    # stores [batch,sentence,sample,width] span_off_samples for each depth
    all_span_off_samples = []

    for d in reversed(xrange(1, max_depth)):

      cur_parent_constraint_layer = parent_constraint_layers[d - 1]
      cur_sum_tree_layer = sum_tree_layers[d - 1]

      # pylint: disable=line-too-long
      cur_span_off_marginals, sum_tree_out_down_msg = cur_parent_constraint_layer.compute_down_msg(
          cur_node_inc_down_msg, node_to_span_off_belief_idx[max_depth - d - 1],
          node_to_span_on_belief_start_idx[max_depth - d - 1],
          node_to_span_on_belief_end_idx[max_depth - d - 1],
          parent_on_down_to_sum_tree_idx[max_depth - d - 1],
          parent_off_down_to_sum_tree_idx[max_depth - d - 1])

      cur_span_off_samples, sum_tree_out_down_samples = cur_parent_constraint_layer.compute_down_samples(
          cur_node_inc_down_samples,
          node_sample_to_span_off_belief_sample_idx[max_depth - d - 1],
          parent_on_sample_down_to_sum_tree_idx[max_depth - d - 1],
          parent_off_sample_down_to_sum_tree_idx[max_depth - d - 1])
      # pylint: enable=line-too-long

      all_span_off_marginals.append(cur_span_off_marginals)
      all_span_off_samples.append(cur_span_off_samples)

      cur_sum_tree_inc_down_msg, _ = cur_sum_tree_layer.compute_down_msg(
          sum_tree_out_down_msg)

      cur_sum_tree_inc_down_samples = cur_sum_tree_layer.compute_down_samples(
          sum_tree_out_down_samples)

      cur_node_inc_down_msg = padded_gather_nd(
          cur_sum_tree_inc_down_msg, sum_tree_down_to_nodes_idx[max_depth - d],
          3, 4)

      cur_node_inc_down_samples = padded_gather_nd(
          cur_sum_tree_inc_down_samples,
          sum_tree_sample_down_to_nodes_idx[max_depth - d], 4, 5)

    # Handle the base case for the final part of sampling

    bottom_span_off_samples = padded_gather_nd(
        cur_node_inc_down_samples,
        node_sample_to_span_off_belief_sample_idx[-1], 4, 5)

    all_span_off_samples.append(bottom_span_off_samples)

    # Handle the base case for the final part of message passing

    bottom_node_beliefs = cur_node_inc_down_msg * init_node_beliefs

    bottom_span_off_beliefs = padded_gather_nd(bottom_node_beliefs,
                                               node_to_span_off_belief_idx[-1],
                                               3, 4)

    bottom_span_off_marginals = bottom_span_off_beliefs

    # Get "integrated" beliefs for easy sums over spans
    integrated_bottom_beliefs = tf.cumsum(bottom_node_beliefs, 2)

    span_on_start_cumulative_belief = padded_gather_nd(
        integrated_bottom_beliefs, node_to_span_on_belief_start_idx[-1], 3, 4)
    span_on_end_cumulative_belief = padded_gather_nd(
        integrated_bottom_beliefs, node_to_span_on_belief_end_idx[-1], 3, 4)

    bottom_span_on_beliefs = (
        span_on_end_cumulative_belief - span_on_start_cumulative_belief)

    bottom_span_belief_normalizer = (
        bottom_span_on_beliefs + bottom_span_off_beliefs)

    bottom_span_off_marginals = su.safe_divide(bottom_span_off_beliefs,
                                               bottom_span_belief_normalizer)

    all_span_off_marginals.append(bottom_span_off_marginals)

    # Gather back out to the (batch,span_id) format

    tree_marg_tensor = tf.concat(
        2, [tf.expand_dims(ss, 2) for ss in all_span_off_marginals])

    # switch off -> on

    span_marginals = 1.0 - padded_gather_nd(
        tree_marg_tensor, span_off_belief_to_span_off_marginal_idx, 4, 3)

    tree_sample_tensor = tf.concat(
        3, [tf.expand_dims(ss, 3) for ss in all_span_off_samples])

    # We have [batch,sample,sent,depth,width] indices
    # in the gather indices
    # but tree_sample_tensor is [batch,sent,sample,depth,width]
    # so we have to transpose
    tree_sample_tensor = tf.transpose(tree_sample_tensor, [0, 2, 1, 3, 4])

    # switch off -> on

    span_samples = 1.0 - padded_gather_nd(
        tree_sample_tensor,
        span_sample_off_belief_to_span_sample_off_marginal_idx, 5, 4)

    return span_marginals, span_samples, running_log_z


class TreeConstrainedExtractor(model_base.Extractor):
  """Implementation of Extractor interface using tree-constrained inference."""

  def __init__(self, tree_inference_inputs, summarizer_features, hps):

    self.hps = hps

    article_max_len = tree_inference_inputs.article_max_len
    batch_size = hps.batch_size
    num_samples = hps.num_samples

    self.word_logits = word_logits = tf.reshape(
        self.get_extractor_logits(hps, tree_inference_inputs,
                                  summarizer_features),
        [hps.batch_size, hps.num_art_steps])

    tree_inferencer = TreeConstrainedInferencer()

    tok_marg, tok_samples, log_z = tree_inferencer.do_tree_inference(
        hps, tree_inference_inputs, word_logits)

    self.log_z = log_z

    sliced_tok_marg = tf.slice(
        tf.reshape(tok_marg, [hps.batch_size, hps.num_art_steps]), [0, 0],
        [batch_size, article_max_len])
    sliced_tok_marg *= tree_inference_inputs.article_sliced_mask

    sliced_tok_samples = tf.slice(
        tf.reshape(tok_samples,
                   [hps.batch_size, num_samples, hps.num_art_steps]), [0, 0, 0],
        [batch_size, num_samples, article_max_len])
    sliced_tok_samples *= tf.reshape(tree_inference_inputs.article_sliced_mask,
                                     [batch_size, 1, -1])

    log_prob_gold_words = tf.reduce_sum(
        tf.to_float(tree_inference_inputs.sliced_extract_label) *
        tf.reshape(word_logits, [batch_size, -1]), 1)
    log_prob_gold_words -= log_z

    self.gold_log_likelihood = log_prob_gold_words

    log_prob_active_words = tf.reduce_sum(
        tf.to_float(sliced_tok_samples) * tf.reshape(word_logits,
                                                     [batch_size, 1, -1]), 2)
    log_prob_active_words -= tf.reshape(log_z, [batch_size, 1])

    self.sample_log_likelihood = log_prob_active_words

    self.marginals = sliced_tok_marg

    self.map_prediction = sliced_tok_marg

    self.samples = sliced_tok_samples

  def get_extractor_logits(self, hps, tree_inference_inputs,
                           summarizer_features):
    del hps, tree_inference_inputs  # don't need these for simple extractor
    return summarizer_features.word_logits


class ParentConstraintLayer(object):
  """Layer of batch junction tree expressing parent->child constraints.

  Contains routines for message passing and sampling with this constraint.

  Attributes:
    init_span_beliefs: scores for spans on this level of tree.
    layer_depth: string describing layer for logging.
  """

  def __init__(self, init_span_beliefs, layer_depth=""):
    self.init_span_beliefs = init_span_beliefs
    self.layer_depth = layer_depth

  def compute_down_samples(self, inc_node_samples,
                           node_sample_to_span_off_belief_sample_idx,
                           parent_on_sample_down_to_sum_tree_idx,
                           parent_off_sample_down_to_sum_tree_idx):
    """Compute downward samples for this layer of the tree.

    Args:
      inc_node_samples: incoming samples from parents.
      node_sample_to_span_off_belief_sample_idx: map from sampled nodes at this
        layer to corresponding span-off variables.
      parent_on_sample_down_to_sum_tree_idx: map from sample of parent-on
        variable to child variable.
      parent_off_sample_down_to_sum_tree_idx: map from sample of parent-off
        variable to child variable.

    Returns:
      span_off_samples: tensor of samples for span-off variables at this level.
      off_samples: tensor of samples for outgoing child variables.
    """

    span_off_samples = padded_gather_nd(
        inc_node_samples, node_sample_to_span_off_belief_sample_idx, 4, 5)

    out_samples = padded_gather_nd(inc_node_samples,
                                   parent_on_sample_down_to_sum_tree_idx, 4, 5)

    out_samples += padded_gather_nd(inc_node_samples,
                                    parent_off_sample_down_to_sum_tree_idx, 4,
                                    5)

    return span_off_samples, out_samples

  def compute_down_msg(self, inc_node_msg, node_to_span_off_belief_idx,
                       node_to_span_on_start_belief_idx,
                       node_to_span_on_end_belief_idx,
                       parent_on_down_to_sum_tree_idx,
                       parent_off_down_to_sum_tree_idx):
    """Compute downward BP messages for this layer of the tree.

    Args:
      inc_node_msg: incoming messages from parent variables.
      node_to_span_off_belief_idx: map from node marginals at this layer to
        corresponding span-off marginals.
      node_to_span_on_start_belief_idx: map marking start of each span marginal.
      node_to_span_on_end_belief_idx: map marking end of each span marginal.
      parent_on_down_to_sum_tree_idx: map from marginal of parent-on
        variable down to child variable.
      parent_off_down_to_sum_tree_idx: map from marginal of parent-off
        variable down to child variable.

    Returns:
      span_off_marginals:
      out_msg:
    """

    node_marginals = self.up_node_msg * inc_node_msg

    span_off_beliefs = padded_gather_nd(node_marginals,
                                        node_to_span_off_belief_idx, 3, 4)

    cumulative_node_beliefs = tf.cumsum(node_marginals, 2)

    span_on_start_cumulative_belief = padded_gather_nd(
        cumulative_node_beliefs, node_to_span_on_start_belief_idx, 3, 4)
    span_on_end_cumulative_belief = padded_gather_nd(
        cumulative_node_beliefs, node_to_span_on_end_belief_idx, 3, 4)

    span_on_beliefs = (
        span_on_end_cumulative_belief - span_on_start_cumulative_belief)

    span_belief_normalizer = span_on_beliefs + span_off_beliefs

    span_off_marginals = su.safe_divide(span_off_beliefs,
                                        span_belief_normalizer)

    out_msg = padded_gather_nd(inc_node_msg, parent_on_down_to_sum_tree_idx, 3,
                               4)

    out_msg += padded_gather_nd(inc_node_msg, parent_off_down_to_sum_tree_idx,
                                3, 4)

    return span_off_marginals, out_msg

  def compute_up_msg(self, inc_msg, sum_tree_up_to_parent_idx,
                     sum_tree_up_to_parent_log_z_idx, span_belief_to_node_idx,
                     span_off_to_node_msg, span_on_to_node_msg, running_log_zs):
    """Compute upward BP messages for this layer of the tree.

    Args:
      inc_msg: incoming messages from child sum tree graph.
      sum_tree_up_to_parent_idx: map from incoming sum tree messages to
        constraint messages.
      sum_tree_up_to_parent_log_z_idx: map from elementwise child logz to
        elementwise parent logz.
      span_belief_to_node_idx: map from span on potential to all children.
      span_off_to_node_msg: map from span-off potential to node message.
      span_on_to_node_msg: map from span-on potential to child-off message.
      running_log_zs: running elementwise log-normalizers for messages.

    Returns:
      message: outgoing message bathc to parents.
      log_zs: outgoing elementwise log-normalizers for messages.
    """

    # First clamp parent span variable to "ON"
    # so map inc messages to their right-shifted counts
    # and multiply by on-belief
    # then add in the off-belief which is 1.0 for both

    log_zs = padded_gather_nd(running_log_zs, sum_tree_up_to_parent_log_z_idx,
                              3, 4)

    message = padded_gather_nd(inc_msg, sum_tree_up_to_parent_idx, 3, 4)

    # renormalize node on and off messages with the running logZ

    zs = tf.exp(log_zs)
    span_on_to_node_msg /= zs
    span_off_to_node_msg /= zs

    # get the span_on for when we have no children

    message = tf.maximum(message, span_on_to_node_msg)

    span_belief_multipliers = padded_gather_nd(self.init_span_beliefs,
                                               span_belief_to_node_idx, 2, 4)

    message *= span_belief_multipliers

    message += span_off_to_node_msg

    self.up_node_msg = message

    return message, log_zs


class SiblingSumTreeLayer(object):
  """Layer of batch junction tree that sums up sibling nodes.

  Contains routines for message passing and sampling these counts.

  Attributes:
    layer_depth: string describing layer for logging.
    k_constraints: If the z-potentials have a hard cutoff for some k-sparsity,
      we can be much more numerically stable by sparsifying the intermediate
      messages before normalizing.
    message_start_levels: masks for summing up multiple sets of siblings at
      once.
    message_end_levels: see message_start_levels.
    message_damp_lambda: lambda for message-damping, improves numerical
      stability at the cost of some accuracy in the inference/sampling.
    min_fft_tree_depth: depth of binary sum tree for inference for smallest
      subtree in batch for this layer.
    fft_tree_depth: depth of binary sum tree for inference.
    combined_msg_width: width of binary sum tree for inference.
  """

  def __init__(self,
               batch_size,
               max_num_sentences,
               num_samples,
               fft_tree_width,
               message_start_levels,
               message_end_levels,
               message_damp_lambda=0.00,
               min_fft_tree_depth=0,
               layer_depth="",
               k_constraints=None):

    self.layer_depth = layer_depth

    self.batch_size = batch_size
    self.max_num_sentences = max_num_sentences
    self.num_samples = num_samples

    self.message_end_levels = tf.reshape(message_end_levels,
                                         [batch_size * max_num_sentences, -1])
    self.message_start_levels = tf.reshape(message_start_levels,
                                           [batch_size * max_num_sentences, -1])

    self.message_damp_lambda = message_damp_lambda
    self.fft_tree_depth = int(math.log(fft_tree_width, 2))
    self.min_fft_tree_depth = min_fft_tree_depth
    self.combined_msg_width = fft_tree_width
    self.k_constraints = k_constraints

  def compute_down_samples(self, inc_samples):
    """Compute downward samples for this layer of the tree.

    Args:
      inc_samples: incoming samples from parents.

    Returns:
      final_samples: outgoing samples to children.
    """
    num_samples = self.num_samples
    batch_size = self.batch_size * self.max_num_sentences

    # sampling routine expects samples dimension on the outside
    inc_samples = tf.reshape(inc_samples, [batch_size, num_samples, -1])
    inc_samples = tf.transpose(inc_samples, [1, 0, 2])
    inc_samples = tf.reshape(inc_samples, [batch_size * num_samples, -1])

    prev_samples = inc_samples

    for d in reversed(xrange(1, self.fft_tree_depth)):
      up_msgs = self.up_msgs[d - 1]
      block_size = np.power(2, d + 1)
      num_splits = self.combined_msg_width / block_size

      less_than_msg_start = tf.less(d, self.message_start_levels)
      greater_than_msg_end = tf.greater_equal(d, self.message_end_levels)
      no_compute_msg = tf.logical_or(less_than_msg_start, greater_than_msg_end)
      no_compute_msg = su.repeat(num_samples, no_compute_msg)

      reshaped_prev_samples = tf.concat(0, tf.split(1, num_splits,
                                                    prev_samples))
      reshaped_prev_idx = tf.expand_dims(
          tf.to_int32(tf.argmax(reshaped_prev_samples, 1)), 1)

      prev_domain_lens = reshaped_prev_idx + 1
      rep_up_msgs = su.repeat(num_samples, up_msgs)
      left_up_msgs_repl, right_up_msgs_repl = tf.split(1, 2, tf.concat(
          0, tf.split(1, num_splits, rep_up_msgs)))

      num_rows = num_splits * batch_size * num_samples

      len_mask = su.create_mask(
          tf.reshape(prev_domain_lens, [-1]), block_size / 2)
      right_up_msgs_repl *= len_mask
      right_up_msgs_repl = tf.pad(right_up_msgs_repl,
                                  [[0, 0], [0, block_size / 2]])
      left_up_msgs_repl *= len_mask
      left_up_msgs_repl = tf.pad(left_up_msgs_repl,
                                 [[0, 0], [0, block_size / 2]])
      left_up_msgs_repl, _ = su.normalize(left_up_msgs_repl)

      right_up_msgs_repl = tf.reverse_sequence(right_up_msgs_repl, tf.reshape(
          tf.to_int64(prev_domain_lens), [-1]), 1)
      right_up_msgs_repl, _ = su.normalize(right_up_msgs_repl)

      sliced_factor_marg, _ = su.normalize(left_up_msgs_repl *
                                           right_up_msgs_repl)
      sliced_factor_marg = tf.slice(sliced_factor_marg, [0, 0],
                                    [num_rows, block_size / 2])

      left_samples, left_idx = su.sample_categorical(tf.log(sliced_factor_marg))
      left_samples = tf.slice(left_samples, [0, 0], [num_rows, block_size / 2])
      right_idx = prev_domain_lens - left_idx - 1
      right_samples = tf.one_hot(tf.reshape(right_idx, [-1]), block_size / 2)

      cur_samples = tf.concat(1, tf.split(0, num_splits, tf.concat(
          1, [left_samples, right_samples])))

      cur_samples = tf.select(no_compute_msg, prev_samples, cur_samples)

      prev_samples = cur_samples

    final_samples = tf.reshape(
        prev_samples,
        [num_samples, self.batch_size, self.max_num_sentences, -1])
    final_samples = tf.transpose(final_samples, [1, 2, 0, 3])

    return final_samples

  def compute_down_msg(self, inc_msg):
    """Compute downward BP messages for this layer of the tree.

    Args:
      inc_msg: incoming messages from parents.

    Returns:
      final_msg: outgoing messages to children.
      final_marg: marginals for variables at this layer.
    """

    batch_size = self.batch_size * self.max_num_sentences

    inc_msg = tf.reshape(inc_msg, [batch_size, -1])

    message_start_levels = self.message_start_levels
    message_end_levels = self.message_end_levels

    prev_msg = inc_msg

    for d in reversed(xrange(1, self.fft_tree_depth)):

      less_than_msg_start = tf.less(d, message_start_levels)
      greater_than_msg_end = tf.greater_equal(d, message_end_levels)
      no_compute_msg = tf.logical_or(less_than_msg_start, greater_than_msg_end)

      up_msgs = self.up_msgs[d - 1]
      block_size = np.power(2, d + 1)
      num_splits = self.combined_msg_width / block_size
      prev_msgs = tf.concat(0, tf.split(1, num_splits, prev_msg))
      left_right_up_msgs = tf.concat(0, tf.split(1, num_splits, up_msgs))
      left_up_msgs, right_up_msgs = tf.split(1, 2, left_right_up_msgs)

      left_down_msgs = ffttii.positive_correl(prev_msgs, tf.pad(
          right_up_msgs, [[0, 0], [0, block_size / 2]]))
      left_down_msgs = tf.slice(left_down_msgs, [0, 0],
                                [num_splits * batch_size, block_size / 2])
      left_down_msgs, _ = su.normalize(left_down_msgs)
      #     left_down_msgs=lam*up_uni+(1-lam)*left_down_msgs
      left_margs, _ = su.normalize(left_up_msgs * left_down_msgs)

      right_down_msgs = ffttii.positive_correl(prev_msgs, tf.pad(
          left_up_msgs, [[0, 0], [0, block_size / 2]]))
      right_down_msgs = tf.slice(right_down_msgs, [0, 0],
                                 [num_splits * batch_size, block_size / 2])
      right_down_msgs, _ = su.normalize(right_down_msgs)
      #     right_down_msgs=lam*up_uni+(1-lam)*right_down_msgs
      right_margs, _ = su.normalize(right_up_msgs * right_down_msgs)

      new_msg = tf.concat(1, tf.split(0, num_splits, tf.concat(
          1, [left_down_msgs, right_down_msgs])))
      new_margs = tf.concat(1, tf.split(0, num_splits,
                                        tf.concat(1,
                                                  [left_margs, right_margs])))

      new_msg = tf.select(no_compute_msg, prev_msg, new_msg)
      margs = tf.select(no_compute_msg, new_msg, new_margs)

      prev_msg = new_msg

    final_msg = tf.reshape(prev_msg,
                           [self.batch_size, self.max_num_sentences, -1])
    final_marg = tf.reshape(margs,
                            [self.batch_size, self.max_num_sentences, -1])

    return final_msg, final_marg

  def compute_up_msg(self, inc_msg, running_log_zs):
    """Compute upward BP messages for this layer of the tree.

    Args:
      inc_msg: incoming messages from children.
      running_log_zs: running elementwise log-normalizers for messages.

    Returns:
      out_msg: outgoing messages to parents.
      log_zs: new elementwise log-normalizers for messages.
    """

    # inc_msg is shape [batch_size,combined_msg_width]
    # message_start_levels is shape [batch_size,combined_msg_width]
    # message_end_levels is shape [batch_size,combined_msg_width]
    # running_log_z is shape [batch_size]

    batch_size = self.batch_size * self.max_num_sentences

    inc_msg = tf.reshape(inc_msg, [batch_size, -1])

    message_start_levels = self.message_start_levels
    message_end_levels = self.message_end_levels

    self.inc_msg = inc_msg

    prev_log_zs = tf.reshape(running_log_zs, [batch_size, -1])

    lam = self.message_damp_lambda

    self.uniforms = []

    self.up_msgs = []

    prev_msg = inc_msg

    for d in xrange(self.min_fft_tree_depth, self.fft_tree_depth):

      less_than_msg_start = tf.less(d, message_start_levels)
      greater_than_msg_end = tf.greater_equal(d, message_end_levels)
      no_compute_msg = tf.logical_or(less_than_msg_start, greater_than_msg_end)

      block_size = np.power(2, d + 1)
      num_splits = self.combined_msg_width / block_size

      sub_probs = tf.split(1, num_splits, prev_msg)

      left, right = tf.split(1, 2, tf.concat(0, sub_probs))
      left = tf.pad(left, [[0, 0], [0, block_size / 2]])
      right = tf.pad(right, [[0, 0], [0, block_size / 2]])
      convs = ffttii.positive_conv(left, right)

      sub_log_zs = tf.split(1, num_splits, prev_log_zs)
      sub_log_zs = tf.concat(0, sub_log_zs)
      sub_log_zs_left, sub_log_zs_right = tf.split(1, 2, sub_log_zs)
      combined_sub_log_zs = tf.tile(sub_log_zs_left + sub_log_zs_right, [1, 2])

      new_msg, _, new_msg_log_zs = su.normalize_and_log_all_reduce(convs)
      new_msg_log_zs += combined_sub_log_zs

      if self.k_constraints is not None:
        k_mask = su.create_mask(
            tf.reshape(
                su.repeat(num_splits, tf.expand_dims(
                    tf.minimum(2**d + 1, self.k_constraints + 1), 1)), [-1]),
            block_size)
        new_msg *= k_mask
        k_uni = k_mask / tf.maximum(
            tf.expand_dims(tf.reduce_sum(k_mask, 1), 1), 1.0)
        new_msg = new_msg * (1.0 - lam) + k_uni * lam
        self.uniforms.append(k_uni)
      else:
        # apply message damping to keep messages well-conditioned
        uni_mask = su.create_mask(
            tf.reshape(
                su.repeat(num_splits, tf.tile(
                    tf.expand_dims(tf.reshape(2**d + 1, [-1]), 1),
                    [batch_size, 1])), [-1]), block_size)
        uni = uni_mask / tf.maximum(
            tf.expand_dims(tf.reduce_sum(uni_mask, 1), 1), 1.0)
        new_msg = new_msg * (1.0 - lam) + uni * lam
        self.uniforms.append(uni)

      new_msg = tf.concat(1, tf.split(0, num_splits, new_msg))
      new_msg = tf.select(no_compute_msg, prev_msg, new_msg)

      self.up_msgs.append(new_msg)
      prev_msg = new_msg

      new_msg_log_zs = tf.concat(1, tf.split(0, num_splits, new_msg_log_zs))

      new_running_log_zs = tf.select(no_compute_msg, prev_log_zs,
                                     new_msg_log_zs)

      prev_log_zs = new_running_log_zs

    out_msg = tf.reshape(prev_msg,
                         [self.batch_size, self.max_num_sentences, -1])
    log_zs = tf.reshape(prev_log_zs,
                        [self.batch_size, self.max_num_sentences, -1])

    return out_msg, log_zs


def add_all_leading_idx_1(ts):
  return [add_leading_idx_1(t) for t in ts]


def add_all_leading_idx_2(ts):
  return [add_leading_idx_2(t) for t in ts]


def add_all_leading_idx_3(ts):
  return [add_leading_idx_3(t) for t in ts]


def add_leading_idx_1(t):
  """See add_leading_idx_3."""

  dims = [d.value for d in t.get_shape().dims]
  b_size = dims[0]
  b_idx = tf.reshape(tf.range(0, b_size), [b_size] + [1] * (len(dims) - 1))
  tiled_idx = tf.tile(b_idx, [1] + dims[1:-1] + [1])
  t_with_b_idx = tf.concat(len(dims) - 1, [tiled_idx, t])
  return t_with_b_idx


def add_leading_idx_2(t):
  """See add_leading_idx_3."""
  dims = [d.value for d in t.get_shape().dims]
  b_size = dims[0]
  s_size = dims[1]
  b_idx = tf.reshape(tf.range(0, b_size), [b_size] + [1] * (len(dims) - 1))
  s_idx = tf.reshape(tf.range(0, s_size), [1, s_size] + [1] * (len(dims) - 2))
  tiled_b_idx = tf.tile(b_idx, [1] + dims[1:-1] + [1])
  tiled_s_idx = tf.tile(s_idx, [dims[0], 1] + dims[2:-1] + [1])
  t_with_b_and_s_idx = tf.concat(len(dims) - 1, [tiled_b_idx, tiled_s_idx, t])
  return t_with_b_and_s_idx


def add_leading_idx_3(t):
  """Utility to automatically add indices used by gather_nd.

  Args:
    t: tensor of shape [b,s,...,n]

  Returns:
    t: tensor of shape [b,s,...,n+2] where the (b,s)-indices are
      prepended onto the values in the final mode.
  """

  dims = [d.value for d in t.get_shape().dims]
  b_size, s_size, ss_size = dims[:3]
  b_idx = tf.reshape(tf.range(0, b_size), [b_size] + [1] * (len(dims) - 1))
  s_idx = tf.reshape(tf.range(0, s_size), [1, s_size] + [1] * (len(dims) - 2))
  ss_idx = tf.reshape(
      tf.range(0, ss_size), [1, 1, ss_size] + [1] * (len(dims) - 3))
  tiled_b_idx = tf.tile(b_idx, [1] + dims[1:-1] + [1])
  tiled_s_idx = tf.tile(s_idx, [dims[0], 1] + dims[2:-1] + [1])
  tiled_ss_idx = tf.tile(ss_idx, [dims[0], dims[1], 1] + dims[3:-1] + [1])
  t_with_b_s_ss_idx = tf.concat(
      len(dims) - 1, [tiled_b_idx, tiled_s_idx, tiled_ss_idx, t])
  return t_with_b_s_ss_idx


def replicate_all_samples_2(ts, num_samples):
  return [replicate_samples_2(t, num_samples) for t in ts]


def replicate_all_samples_3(ts, num_samples):
  return [replicate_samples_3(t, num_samples) for t in ts]


def replicate_samples_2(t, num_samples):
  # take in a rank-r tensor of shape [b,...]
  # return a rank-r+1 tensor of shape [b,num_samples,...]
  return tf.tile(
      tf.expand_dims(t, 1),
      [1, num_samples] + [1] * (len(t.get_shape().dims) - 1))


def replicate_samples_3(t, num_samples):
  # take in a rank-r tensor of shape [b,s,...]
  # return a rank-r+1 tensor of shape [b,s,num_samples,...]
  return tf.tile(
      tf.expand_dims(t, 2),
      [1, 1, num_samples] + [1] * (len(t.get_shape().dims) - 2))


class TreeInferenceInputs(model_base.ModelInputs):

  def __init__(self, hps):

    super(TreeInferenceInputs, self).__init__(hps)

    max_depth = len(hps.tree_widths_at_level)

    # for each sentence, we need gather indices from the sequence of potentials
    # to get the span potentials
    # contains (batch_num,seq_idx) pairs, 1 per
    self.span_start_idx = tf.placeholder(tf.int32,
                                         [hps.batch_size, hps.max_num_spans, 2])
    self.span_end_idx = tf.placeholder(tf.int32,
                                       [hps.batch_size, hps.max_num_spans, 2])

    # gathers the span samples and span marginals back out to the tokens
    # contains (batch_num,span_idx) pairs
    # -1s in here to pick 0s
    self.span_idx_for_tok_marg = tf.placeholder(
        tf.int32, [hps.batch_size, hps.num_art_steps, 1])

    up_fftt_w = list(reversed(hps.fft_tree_widths_at_level))
    up_t_w = list(reversed(hps.tree_widths_at_level))

    # first make the placeholders for the upward messages

    # gather from (batch_id,sentence,msg_idx)
    # skip the 1st level since there's no sum tree below
    self.sum_tree_up_to_parent_idx = [
        tf.placeholder(tf_int16,
                       [hps.batch_size, hps.max_num_sentences, wat, 1])
        for wat in up_t_w[1:]
    ]

    self.sum_tree_up_to_parent_log_z_idx = [
        tf.placeholder(tf_int16,
                       [hps.batch_size, hps.max_num_sentences, wat, 1])
        for wat in up_t_w[1:]
    ]

    # Now the span potentials come in

    # gather from (batch_id,span_id)
    # for all levels, even the 1st
    self.span_belief_to_node_idx = [
        tf.placeholder(tf_int16,
                       [hps.batch_size, hps.max_num_sentences, wat, 1])
        for wat in up_t_w
    ]

    # just contains 1.0 where the "off" setting for spans is
    # for all levels, even the 1st
    self.span_off_to_node_msg = [
        tf.placeholder(tf.float32, [hps.batch_size, hps.max_num_sentences, wat])
        for wat in up_t_w
    ]
    self.span_on_to_node_msg = [
        tf.placeholder(tf.float32, [hps.batch_size, hps.max_num_sentences, wat])
        for wat in up_t_w
    ]

    self.nodes_up_to_sum_tree_idx = [
        tf.placeholder(tf_int16,
                       [hps.batch_size, hps.max_num_sentences, wat, 1])
        for wat in up_fftt_w
    ]

    self.nodes_up_to_sum_tree_log_z_idx = [
        tf.placeholder(tf_int16,
                       [hps.batch_size, hps.max_num_sentences, wat, 1])
        for wat in up_fftt_w
    ]

    self.sum_tree_msg_start_depths = [
        tf.placeholder(tf_int8, [hps.batch_size, hps.max_num_sentences, wat])
        for wat in up_fftt_w
    ]

    self.sum_tree_msg_end_depths = [
        tf.placeholder(tf_int8, [hps.batch_size, hps.max_num_sentences, wat])
        for wat in up_fftt_w
    ]

    # now set up the global summing tree

    # gather from (batch,sentence,node)
    self.node_up_to_global_idx = tf.placeholder(
        tf_int16, [hps.batch_size, hps.global_fft_tree_width, 2])

    self.node_up_to_global_log_z_idx = tf.placeholder(
        tf_int16, [hps.batch_size, hps.global_fft_tree_width, 2])

    self.global_sum_tree_msg_start_depths = tf.placeholder(
        tf_int8, [hps.batch_size, hps.global_fft_tree_width])

    self.global_sum_tree_msg_end_depths = tf.placeholder(
        tf_int8, [hps.batch_size, hps.global_fft_tree_width])

    # gather from (batch,node) to (batch,sentence,node)
    self.global_down_to_node_idx = tf.placeholder(
        tf_int16,
        [hps.batch_size, hps.max_num_sentences, hps.tree_widths_at_level[0], 1])

    # now pass messages and samples down

    down_fftt_w = hps.fft_tree_widths_at_level
    down_t_w = hps.tree_widths_at_level

    self.parent_on_down_to_sum_tree_idx = [
        tf.placeholder(tf_int16,
                       [hps.batch_size, hps.max_num_sentences, wat, 1])
        for wat in down_fftt_w
    ]

    self.parent_off_down_to_sum_tree_idx = [
        tf.placeholder(tf_int16,
                       [hps.batch_size, hps.max_num_sentences, wat, 1])
        for wat in down_fftt_w
    ]

    self.sum_tree_down_to_nodes_idx = [
        tf.placeholder(tf_int16,
                       [hps.batch_size, hps.max_num_sentences, wat, 1])
        for wat in down_t_w
    ]

    # gather from (batch_id,sent_id,node_belief_idx)
    # then we vertically concat all this into [batch,sent,depth_idx,width_idx]

    self.node_to_span_off_belief_idx = [
        tf.placeholder(tf_int16, [hps.batch_size, hps.max_num_sentences,
                                  hps.max_tree_nodes_any_level, 1])
        for _ in xrange(max_depth)
    ]

    self.node_to_span_on_belief_range_idx = [
        tf.placeholder(tf_int16, [hps.batch_size, hps.max_num_sentences,
                                  hps.max_tree_nodes_any_level, 2])
        for _ in xrange(max_depth)
    ]

    # then we gather it back to span_off_marginals
    # from (batch_id,sent_id,depth_idx,width_idx)
    self.span_off_belief_to_span_off_marginal_idx = tf.placeholder(
        tf_int16, [hps.batch_size, hps.max_num_spans, 3])


class TreeInferenceBatch(data.SummaryBatch):
  """Model feeds for batch for tree-constrained inference.

  Contains logic for walking over parse tree graphs and compiling them into
  tensors describing edges between components of message passing graph.

  Attributes:
    feeds: Feed dictionary mapping model placeholders to numpy tensors.
  """

  def __init__(self, hps, tree_inference_inputs, examples, vocab):

    super(TreeInferenceBatch,
          self).__init__(hps, tree_inference_inputs, examples, vocab)

    feeds = self.feeds

    tii = tree_inference_inputs

    feeds[tii.span_start_idx] = span_start_idx = np.zeros(
        [hps.batch_size, hps.max_num_spans, 2], dtype=np.int32)
    feeds[tii.span_end_idx] = span_end_idx = np.zeros(
        [hps.batch_size, hps.max_num_spans, 2], dtype=np.int32)

    feeds[tii.span_idx_for_tok_marg] = span_idx_for_tok_marg = np.zeros(
        [hps.batch_size, hps.num_art_steps, 1], dtype=np.int32)

    # Set up the mappings between tokens and spans
    b = 0
    for ex in examples:
      tok_count = 0
      span_count = 0
      edu_id = -1
      cur_edu_id = -1
      for edu_id_sent in ex.article_edu_ids:
        for edu_id in edu_id_sent:
          if edu_id != cur_edu_id:
            span_start_idx[b, edu_id, 0] = b
            span_start_idx[b, edu_id, 1] = tok_count
            if cur_edu_id != -1:
              span_end_idx[b, cur_edu_id, 0] = b
              span_end_idx[b, cur_edu_id, 1] = tok_count
            span_count += 1
            cur_edu_id = edu_id
          span_idx_for_tok_marg[b, tok_count, 0] = edu_id
          tok_count += 1
      span_end_idx[b, edu_id, 0] = b
      span_end_idx[b, edu_id, 1] = tok_count
      b += 1

    up_fftt_w = list(reversed(hps.fft_tree_widths_at_level))
    up_t_w = list(reversed(hps.tree_widths_at_level))

    sum_tree_up_to_parent_idx = []
    for wat, p in zip(up_t_w[1:], tii.sum_tree_up_to_parent_idx):
      feeds[p] = n = np.full(
          [hps.batch_size, hps.max_num_sentences, wat, 1], -1, dtype=np_int16)
      sum_tree_up_to_parent_idx.append(n)

    sum_tree_up_to_parent_log_z_idx = []
    for wat, p in zip(up_t_w[1:], tii.sum_tree_up_to_parent_log_z_idx):
      feeds[p] = n = np.full(
          [hps.batch_size, hps.max_num_sentences, wat, 1], -1, dtype=np_int16)
      sum_tree_up_to_parent_log_z_idx.append(n)

    span_belief_to_node_idx = []
    for wat, p in zip(up_t_w, tii.span_belief_to_node_idx):
      feeds[p] = n = np.full(
          [hps.batch_size, hps.max_num_sentences, wat, 1], -1, dtype=np_int16)
      span_belief_to_node_idx.append(n)

    span_off_to_node_msg = []
    for wat, p in zip(up_t_w, tii.span_off_to_node_msg):
      feeds[p] = n = np.full(
          [hps.batch_size, hps.max_num_sentences, wat], 0.0, dtype=np.float)
      span_off_to_node_msg.append(n)

    span_on_to_node_msg = []
    for wat, p in zip(up_t_w, tii.span_on_to_node_msg):
      feeds[p] = n = np.full(
          [hps.batch_size, hps.max_num_sentences, wat], 0.0, dtype=np.float)
      span_on_to_node_msg.append(n)

    nodes_up_to_sum_tree_idx = []
    for wat, p in zip(up_fftt_w, tii.nodes_up_to_sum_tree_idx):
      feeds[p] = n = np.full(
          [hps.batch_size, hps.max_num_sentences, wat, 1], -1, dtype=np_int16)
      nodes_up_to_sum_tree_idx.append(n)

    nodes_up_to_sum_tree_log_z_idx = []
    for wat, p in zip(up_fftt_w, tii.nodes_up_to_sum_tree_log_z_idx):
      feeds[p] = n = np.full(
          [hps.batch_size, hps.max_num_sentences, wat, 1], -1, dtype=np_int16)
      nodes_up_to_sum_tree_log_z_idx.append(n)

    sum_tree_msg_start_depths = []
    for wat, p in zip(up_fftt_w, tii.sum_tree_msg_start_depths):
      feeds[p] = n = np.full(
          [hps.batch_size, hps.max_num_sentences, wat], 0, dtype=np_int8)
      sum_tree_msg_start_depths.append(n)

    sum_tree_node_ids = []
    for wat in up_fftt_w:
      n = np.full(
          [hps.batch_size, hps.max_num_sentences, wat], 0, dtype=np.int32)
      sum_tree_node_ids.append(n)

    sum_tree_msg_end_depths = []
    for wat, p in zip(up_fftt_w, tii.sum_tree_msg_end_depths):
      feeds[p] = n = np.full(
          [hps.batch_size, hps.max_num_sentences, wat], 0, dtype=np_int8)
      sum_tree_msg_end_depths.append(n)

    node_up_to_global_idx = np.full(
        [hps.batch_size, hps.global_fft_tree_width, 2], -1, dtype=np_int16)
    feeds[tii.node_up_to_global_idx] = node_up_to_global_idx

    node_up_to_global_log_z_idx = np.full(
        [hps.batch_size, hps.global_fft_tree_width, 2], -1, dtype=np_int16)
    feeds[tii.node_up_to_global_log_z_idx] = node_up_to_global_log_z_idx

    global_sum_tree_msg_start_depths = np.full(
        [hps.batch_size, hps.global_fft_tree_width], 0, dtype=np_int8)
    feeds[
        tii.global_sum_tree_msg_start_depths] = global_sum_tree_msg_start_depths

    global_sum_tree_msg_end_depths = np.full(
        [hps.batch_size, hps.global_fft_tree_width], 0, dtype=np_int8)
    feeds[tii.global_sum_tree_msg_end_depths] = global_sum_tree_msg_end_depths

    global_down_to_node_idx = np.full(
        [hps.batch_size, hps.max_num_sentences, hps.tree_widths_at_level[0], 1],
        -1,
        dtype=np_int16)
    feeds[tii.global_down_to_node_idx] = global_down_to_node_idx

    down_fftt_w = hps.fft_tree_widths_at_level
    down_t_w = hps.tree_widths_at_level

    parent_on_down_to_sum_tree_idx = []
    for wat, p in zip(down_fftt_w, tii.parent_on_down_to_sum_tree_idx):
      feeds[p] = n = np.full(
          [hps.batch_size, hps.max_num_sentences, wat, 1], -1, dtype=np_int16)
      parent_on_down_to_sum_tree_idx.append(n)

    parent_off_down_to_sum_tree_idx = []
    for wat, p in zip(down_fftt_w, tii.parent_off_down_to_sum_tree_idx):
      feeds[p] = n = np.full(
          [hps.batch_size, hps.max_num_sentences, wat, 1], -1, dtype=np_int16)
      parent_off_down_to_sum_tree_idx.append(n)

    sum_tree_down_to_nodes_idx = []
    for wat, p in zip(down_t_w, tii.sum_tree_down_to_nodes_idx):
      feeds[p] = n = np.full(
          [hps.batch_size, hps.max_num_sentences, wat, 1], -1, dtype=np_int16)
      sum_tree_down_to_nodes_idx.append(n)

    max_depth = len(hps.tree_widths_at_level)

    # pylint: disable=line-too-long
    node_to_span_off_belief_idx = []
    for p in tii.node_to_span_off_belief_idx:
      feeds[p] = n = np.full(
          [hps.batch_size, hps.max_num_sentences, hps.max_tree_nodes_any_level,
           1],
          -1,
          dtype=np_int16)
      node_to_span_off_belief_idx.append(n)

    node_to_span_on_belief_range_idx = []
    for p in tii.node_to_span_on_belief_range_idx:
      feeds[p] = n = np.full(
          [hps.batch_size, hps.max_num_sentences, hps.max_tree_nodes_any_level,
           2],
          -1,
          dtype=np_int16)
      node_to_span_on_belief_range_idx.append(n)

    span_off_belief_to_span_off_marginal_idx = np.full(
        [hps.batch_size, hps.max_num_spans, 3], -1, dtype=np_int16)
    feeds[
        tii.
        span_off_belief_to_span_off_marginal_idx] = span_off_belief_to_span_off_marginal_idx
    # pylint: enable=line-too-long

    # "examples" is size of batch
    b = 0
    for ex in examples:

      # for each node, we want start, width
      # for both the node layers
      node_spans = np.zeros([hps.max_num_spans, 2], dtype=np.int32)
      # and the sum tree layers
      sum_tree_spans = np.zeros([hps.max_num_spans, 2], dtype=np.int32)

      # make trees for every sentence
      num_sents = len(ex.article_edu_ids)
      dtrees = []
      for s in xrange(num_sents):

        dtree = d_t.build_discourse_tree(ex.article_edu_ids[s],
                                         ex.article_parent_ids[s])
        dtrees.append(dtree)

        all_nodes = list(d_t.discourse_tree_depth_first_walk(dtree))

        # layout node layers (simple greedy layout)
        node_layer_offsets = np.zeros([max_depth], dtype=np.int32)
        for n in all_nodes:
          cur_offset = node_layer_offsets[n.level]
          cur_width = n.total_num_leaves + 1
          node_layer_offsets[n.level] += cur_width
          node_spans[n.node_id, 0] = cur_offset
          node_spans[n.node_id, 1] = cur_width

        # now to layout sum tree layers, we must get all nodes at each level
        # and sort by biggest sum tree and round up to power of 2
        sum_tree_layer_offsets = np.zeros([max_depth], dtype=np.int32)
        sorted_node_triples = [(node.level, -node.child_sum_tree.width, node)
                               for node in all_nodes
                               if node.child_sum_tree is not None]
        sorted_node_triples.sort()

        for cur_lvl, neg_cur_width, cur_node in sorted_node_triples:
          cur_offset = sum_tree_layer_offsets[cur_lvl]
          cur_width = -neg_cur_width
          sum_tree_layer_offsets[cur_lvl] += cur_width

          # lay out the children here
          child_offset = cur_offset
          for leaf in d_t.sum_tree_depth_first_leaf_walk(
              cur_node.child_sum_tree):
            sum_tree_spans[leaf.span_id, 0] = child_offset
            sum_tree_spans[leaf.span_id, 1] = leaf.width
            child_offset += leaf.width

        span_off_belief_offsets = np.zeros([max_depth], dtype=np.int32)

        # Now we have the spans for the nodes and sum tree layers
        # we can go through and wire up all the pointers
        for n in all_nodes:

          node_children_size = n.total_num_leaves - n.node_size + 1
          node_span_start = node_spans[n.node_id, 0]
          node_span_len = node_spans[n.node_id, 1]
          node_span_end = node_span_start + node_span_len

          # node/upward downward messages that don't involve sum trees
          # span off to node -- just put a 1.0 in the zero position for the node
          span_off_to_node_msg[-n.level - 1][b, s, node_span_start] = 1.0
          span_on_to_node_msg[-n.level - 1][b, s, node_span_start +
                                            n.node_size] = 1.0

          # span belief to node -- gather from the span-id,potential array
          span_belief_to_node_idx[-n.level - 1][
              b, s, node_span_start + n.node_size:node_span_end, 0] = n.node_id

          # sum-tree-down-to-nodes -- all nodes have this
          # pylint: disable=line-too-long
          # pylint: disable=g-backslash-continuation
          sum_tree_down_to_nodes_idx[n.level][b, s, node_span_start:node_span_end, 0] =\
              np.arange(sum_tree_spans[n.node_id, 0],
                        sum_tree_spans[n.node_id, 0]+node_span_len)
          # pylint: enable=g-backslash-continuation
          # pylint: enable=line-too-long

          # messages to the span-off and span-on beliefs for final infer step
          # TODO(lvilnis@) add "tree" nomenclature here to differentiate
          cur_span_off_belief_offset = span_off_belief_offsets[n.level]
          node_to_span_off_belief_idx[n.level][b, s, cur_span_off_belief_offset,
                                               0] = node_span_start

          # have to subtract 1 from range since normally spans are exclusive
          # but these spans are inclusive since we use cumsum trick
          node_to_span_on_belief_range_idx[n.level][
              b, s, cur_span_off_belief_offset, 0] = node_span_start
          node_to_span_on_belief_range_idx[n.level][
              b, s, cur_span_off_belief_offset, 1] = node_span_end - 1

          # after we su.normalize with the span-on beliefs,
          # we gather out to the span-off marginals
          span_off_belief_to_span_off_marginal_idx[b, n.node_id, 0] = s
          span_off_belief_to_span_off_marginal_idx[b, n.node_id, 1] = n.level
          span_off_belief_to_span_off_marginal_idx[
              b, n.node_id, 2] = cur_span_off_belief_offset

          span_off_belief_offsets[n.level] += 1

          # sum tree upward/downward messages
          if n.child_sum_tree is not None:
            start_depths = sum_tree_msg_start_depths[-n.level - 1]
            end_depths = sum_tree_msg_end_depths[-n.level - 1]
            stnodeids = sum_tree_node_ids[-n.level - 1]
            end_depth = n.child_sum_tree.depth
            node_to_sum_tree_gather = nodes_up_to_sum_tree_idx[-n.level - 1]
            node_to_sum_tree_log_z_gather = nodes_up_to_sum_tree_log_z_idx[
                -n.level - 1]
            leaves = list(d_t.sum_tree_depth_first_leaf_walk(n.child_sum_tree))
            for leaf in leaves:
              leaf_start_depth = leaf.depth
              leaf_start = sum_tree_spans[leaf.span_id, 0]
              leaf_length = sum_tree_spans[leaf.span_id, 1]
              leaf_end = leaf_start + leaf_length

              # set the start/end depths
              start_depths[b, s, leaf_start:leaf_end] = leaf_start_depth
              stnodeids[b, s, leaf_start:leaf_end] = leaf.span_id
              end_depths[b, s, leaf_start:leaf_end] = end_depth

              # set the node-up-to-sum-tree gather indices
              node_start = node_spans[leaf.span_id, 0]
              node_length = node_spans[leaf.span_id, 1]
              node_end = node_start + node_length
              node_to_sum_tree_gather[b, s, leaf_start:leaf_start + node_length,
                                      0] = np.arange(node_start, node_end)
              node_to_sum_tree_log_z_gather[b, s, leaf_start:leaf_end,
                                            0] = node_start

            # the downward messages go to the position of the leftmost leaf
            first_leaf_start = sum_tree_spans[leaves[0].span_id, 0]

            # set the parent=ON down to sum tree gather indices
            # so we shift parent indices by the size of the parent and copy down
            parent_on_down_to_sum_tree_idx[n.level][
                b, s, first_leaf_start:first_leaf_start + node_children_size,
                0] = np.arange(node_span_start + n.node_size, node_span_end)

            # set the parent=OFF down to sum tree gather indices
            # grab the node start index from the parent
            parent_off_down_to_sum_tree_idx[n.level][b, s, first_leaf_start,
                                                     0] = node_span_start

            # now the sum-tree-up-to-parent messages
            # shift right by the parent size since these are on
            sum_tree_up_to_parent_idx[-n.level - 1][
                b, s, node_span_start + n.node_size:node_span_end,
                0] = np.arange(first_leaf_start,
                               first_leaf_start + node_children_size)
            sum_tree_up_to_parent_log_z_idx[-n.level - 1][
                b, s, node_span_start:node_span_end, 0] = first_leaf_start

      # after the sentence-specific schedules, we lay out the global sum tree
      # message gather indices
      node_id_to_sent = dict(
          [(dtrees[s].node_id, s) for s in xrange(num_sents)])
      node_id_to_dtree = dict(
          [(dtrees[s].node_id, dtrees[s]) for s in xrange(num_sents)])
      global_sum_tree = d_t.build_sum_tree(
          [(dt.total_num_leaves + 1, dt.node_id) for dt in dtrees])
      global_leaves = d_t.sum_tree_depth_first_leaf_walk(global_sum_tree)
      global_sum_tree_offset = 0

      # TODO(lvilnis@) can this be combined with per-sentence sum-tree layout?
      global_sum_tree_msg_end_depths[
          b, :global_sum_tree.width] = global_sum_tree.depth
      global_spans = np.zeros([hps.max_num_sentences, 2], dtype=np.int32)
      for leaf in global_leaves:
        new_offset = global_sum_tree_offset + leaf.width
        global_sum_tree_msg_start_depths[b, global_sum_tree_offset:
                                         new_offset] = leaf.depth
        true_leaf_width = node_id_to_dtree[leaf.span_id].total_num_leaves + 1
        leaf_sent = node_id_to_sent[leaf.span_id]
        global_spans[leaf_sent, 0] = global_sum_tree_offset
        global_spans[leaf_sent, 1] = true_leaf_width
        node_up_to_global_idx[b, global_sum_tree_offset:global_sum_tree_offset +
                              true_leaf_width, 0] = leaf_sent
        node_up_to_global_idx[b, global_sum_tree_offset:global_sum_tree_offset +
                              true_leaf_width, 1] = np.arange(0,
                                                              true_leaf_width)
        node_up_to_global_log_z_idx[b, global_sum_tree_offset:new_offset,
                                    0] = leaf_sent
        node_up_to_global_log_z_idx[b, global_sum_tree_offset:new_offset, 1] = 0
        global_sum_tree_offset = new_offset

      # now that we have the global upwards pass
      # need to gather global messages back to individual sentences
      for s in xrange(num_sents):
        sent_start = global_spans[s, 0]
        sent_len = global_spans[s, 1]
        sent_end = sent_start + sent_len
        global_down_to_node_idx[b, s, :sent_len, 0] = np.arange(
            sent_start, sent_end, dtype=np.int32)

      b += 1
