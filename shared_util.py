"""Shared utilities for extractive summarization models."""

from __future__ import absolute_import

import tensorflow as tf

from tensorflow.contrib.rnn.python.ops import lstm_ops

relu = tf.contrib.layers.relu
linear = tf.contrib.layers.linear

block_lstm = lstm_ops._block_lstm  # pylint: disable=protected-access


def repeat(num_repeats, t):
  return tf.tile(t, [num_repeats, 1])


def repeat_row(num, t):
  return tf.reshape(tf.tile(tf.reshape(t, [-1, 1]), tf.pack([1, num])), [-1])


def create_mask(lengths, max_len):
  """Create a binary mask corresponding to lengths."""
  # cast to float to get efficient GPU kernels for tf.tile
  lengths = tf.cast(lengths, dtype=tf.float32)
  tile_shape = tf.pack([1] * (lengths.get_shape().ndims) + [max_len])
  tiled_lengths = tf.tile(tf.expand_dims(lengths, -1), tile_shape)
  upto_max = tf.range(0, max_len, 1)
  upto_max = tf.cast(upto_max, dtype=tf.float32)
  expanded_upto_max = tf.reshape(upto_max, tile_shape)
  tiled_upto_max = tf.tile(expanded_upto_max,
                           lengths.get_shape().concatenate([1]))
  mask = tf.less(tiled_upto_max, tiled_lengths)
  return tf.to_float(mask)


def gumbel_noise(shape):
  """Return tensor of samples from Gumbel random variable."""
  uniform_noise = tf.random_uniform(shape, 0.0, 1.0)
  neg_log_uniform_noise = -tf.log(uniform_noise)
  neg_log_neg_log_uniform_noise = -tf.log(neg_log_uniform_noise)
  return neg_log_neg_log_uniform_noise


def sample_categorical(log_scores):
  """Use Gumbel-max trick to sample from categorical distribution."""
  in_shape = log_scores.get_shape()
  noised = log_scores + gumbel_noise(in_shape)
  indices = tf.to_int32(tf.argmax(noised, 1))
  ret_sample = tf.one_hot(indices, in_shape[1])
  ret_indices = tf.expand_dims(indices, 1)
  return ret_sample, ret_indices


def safe_divide(n, d):
  """Divide n by d, replace divide by 0 with 0 for handle masked inputs."""
  zeros_d = tf.zeros_like(d)
  ones_d = tf.ones_like(d)
  safe_denom = tf.select(tf.equal(d, zeros_d), ones_d, d)
  return n / safe_denom


def normalize(vec):
  """Length-normalize vec, passing through length-0 inputs unchanged."""
  # just return zeros if incoming vec equals zeros
  z = tf.expand_dims(tf.reduce_sum(vec, 1), 1)
  return safe_divide(vec, z), z


def safe_log(t):
  """Logarithm that returns 0 for log(0), for easy masking."""
  zeros_t = tf.zeros_like(t)
  ones_t = tf.ones_like(t)
  safe = tf.log(tf.select(tf.equal(t, zeros_t), ones_t, t))
  return safe


def safe_divide_and_log(n, d):
  """Log and div returning 0 for log(0) and div by 0, for easy masking."""
  zeros_d = tf.zeros_like(d)
  ones_d = tf.ones_like(d)
  is_zero = tf.equal(d, zeros_d)
  safe_denom = tf.select(is_zero, ones_d, d)
  safe = tf.log(safe_denom)
  return n / safe_denom, safe


def all_reduce_sum(t, dim):
  """Like reduce_sum, but broadcasts sum out to every entry in reduced dim."""
  t_shape = t.get_shape()
  rank = t.get_shape().ndims
  return tf.tile(
      tf.expand_dims(tf.reduce_sum(t, dim), dim),
      [1] * dim + [t_shape[dim].value] + [1] * (rank - dim - 1))


def normalize_all_reduce(vec):
  """Like normalize, but broadcasts norm out to every entry."""
  # just return zeros if incoming vec equals zeros
  z = all_reduce_sum(vec, 1)
  return safe_divide(vec, z), z


def normalize_and_log(vec):
  """Like normalize, but also returns log-normalizer."""
  # just return zeros if incoming vec equals zeros
  z = tf.expand_dims(tf.reduce_sum(vec, 1), 1)
  normed, logz = safe_divide_and_log(vec, z)
  return normed, z, logz


def normalize_and_log_all_reduce(vec):
  """Like normalize_all_reduce, but also returns log-normalizer."""
  # just return zeros if incoming vec equals zeros
  z = all_reduce_sum(vec, 1)
  normed, logz = safe_divide_and_log(vec, z)
  return normed, z, logz


def create_log_mask(lengths, max_len):
  """Create a float mask in log-space corresponding to lengths."""
  reg_mask = create_mask(lengths, max_len)
  return (1.0 - reg_mask) * -100.0


def deep_birnn(hps, inputs, sequence_length, num_layers=1):
  """Efficient deep bidirectional rnn.

  Args:
    hps: bag of hyperparameters.
    inputs: [batch, steps, units] tensor of input embeddings for RNN.
    sequence_length: number of steps for each inputs.
    num_layers: depth of RNN.

  Returns:
    Outputs of RNN.
  """
  sequence_length = sequence_length
  sequence_length_mask = tf.expand_dims(
      create_mask(sequence_length, hps.num_art_steps), 2)
  for j in xrange(num_layers):

    with tf.variable_scope("birnn_fwd_%d" % j):
      w = tf.get_variable(
          "w", [hps.word_embedding_size + hps.hidden_size, 4 * hps.hidden_size])
      b = tf.get_variable("b", [4 * hps.hidden_size])
      split_inputs = [tf.reshape(t, [hps.batch_size, -1])
                      for t in tf.split(1, hps.num_art_steps, inputs)]
      (_, _, _, _, _, _, h) = block_lstm(
          tf.to_int64(hps.num_art_steps), split_inputs, w, b, forget_bias=1.0)
      fwd_outs = h
      fwd_outs = tf.concat(1, [tf.expand_dims(fwdo, 1) for fwdo in fwd_outs])
      fwd_outs *= sequence_length_mask

    with tf.variable_scope("birnn_bwd_%d" % j):
      w = tf.get_variable(
          "w", [hps.word_embedding_size + hps.hidden_size, 4 * hps.hidden_size])
      b = tf.get_variable("b", [4 * hps.hidden_size])
      if sequence_length is not None:
        rev_inputs = tf.reverse_sequence(inputs, tf.to_int64(sequence_length),
                                         1)
      else:
        rev_inputs = tf.reverse(inputs, 1)

      split_rev_inputs = [tf.reshape(t, [hps.batch_size, -1])
                          for t in tf.split(1, hps.num_art_steps, rev_inputs)]
      (_, _, _, _, _, _, h) = block_lstm(
          tf.to_int64(hps.num_art_steps),
          split_rev_inputs,
          w,
          b,
          forget_bias=1.0)
      bwd_outs = h
      bwd_outs = tf.concat(1, [tf.expand_dims(bwdo, 1) for bwdo in bwd_outs])
      bwd_outs *= sequence_length_mask

      if sequence_length is not None:
        rev_bwd_outs = tf.reverse_sequence(bwd_outs,
                                           tf.to_int64(sequence_length), 1)
      else:
        rev_bwd_outs = tf.reverse(bwd_outs, 1)

    inputs = tf.concat(2, [fwd_outs, rev_bwd_outs])

    return inputs
