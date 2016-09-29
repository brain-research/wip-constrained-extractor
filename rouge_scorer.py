"""Class for evaluating ROUGE-1 recall scores of paired texts."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class ROUGEScorer(object):
  """Class for evaluating ROUGE-1 recall scores of paired texts."""

  def __init__(self, vocab):

    # Sparse matrix mapping words to their stems

    vocab_size = vocab.vocab_size
    stem_vocab_size = len(vocab.word_stems)
    stem_map_vocab_coord = tf.to_int64(
        tf.expand_dims(tf.range(0, vocab_size), 1))

    stem_map_stem_coord_np = np.zeros(vocab_size)
    for w, idx in vocab.word_indices.iteritems():
      stem_map_stem_coord_np[idx] = vocab.stem_indices[vocab.word_stems[w]]

    stem_map_stem_coord = tf.to_int64(
        tf.expand_dims(
            tf.constant(
                stem_map_stem_coord_np, shape=[vocab_size]), 1))

    stem_map_indices = tf.concat(1, [stem_map_vocab_coord, stem_map_stem_coord])

    stem_values_stopworded_np = np.zeros(vocab_size)
    for w, idx in vocab.word_indices.iteritems():
      if vocab.is_stop[vocab.word_indices[w]] == 1:
        stem_values_stopworded_np[idx] = 0.0
      else:
        stem_values_stopworded_np[idx] = 1.0

    stem_values_stopworded_np[vocab.word_indices["<UNK>"]] = 0.0

    # Separate map to remove stopwords except <UNK>
    stem_values_stopworded_keep_unk_np = stem_values_stopworded_np.copy()
    stem_values_stopworded_keep_unk_np[vocab.word_indices["<UNK>"]] = 1.0

    stem_values_stopworded = tf.constant(
        stem_values_stopworded_np, shape=[vocab_size])
    stem_values_stopworded = tf.to_float(stem_values_stopworded)
    stem_values_stopworded_keep_unk = tf.constant(
        stem_values_stopworded_keep_unk_np, shape=[vocab_size])
    stem_values_stopworded_keep_unk = tf.to_float(
        stem_values_stopworded_keep_unk)

    stem_proj_shape = [vocab_size, stem_vocab_size]

    self.stem_projector_stopworded = tf.SparseTensor(stem_map_indices,
                                                     stem_values_stopworded,
                                                     stem_proj_shape)
    self.stem_projector_stopworded_keep_unk = tf.SparseTensor(
        stem_map_indices, stem_values_stopworded_keep_unk, stem_proj_shape)

  def get_rouge_recall(self, pred_counts, gold_counts):
    """Get ROUGE-1 recall for pair of bags."""
    overlaps, gold_counts = self.get_rouge_recall_suff_stats(pred_counts,
                                                             gold_counts)
    return overlaps / gold_counts

  def get_rouge_recall_suff_stats(self, pred_counts, gold_counts):
    """Get overlapping predicted counts and gold counts for pair of bags."""
    # Map words to their stems.
    pred_stems = tf.transpose(
        tf.sparse_tensor_dense_matmul(
            self.stem_projector_stopworded,
            pred_counts,
            adjoint_a=True,
            adjoint_b=True))
    # <UNK> tokens count as always missing from predicted counts but not
    # from gold counts to avoid overly optimistic evaluation.
    gold_stems = tf.transpose(
        tf.sparse_tensor_dense_matmul(
            self.stem_projector_stopworded_keep_unk,
            gold_counts,
            adjoint_a=True,
            adjoint_b=True))

    # Only count max of 1 point for each overlapping word type
    pred_stems = tf.minimum(1.0, pred_stems)
    gold_stems = tf.minimum(1.0, gold_stems)

    overlaps = tf.reduce_sum(tf.minimum(pred_stems, gold_stems), 1)

    gold_counts = tf.reduce_sum(gold_stems, 1)

    return overlaps, gold_counts
