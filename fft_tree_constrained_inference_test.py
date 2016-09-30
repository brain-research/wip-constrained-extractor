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

"""Tests for fft_tree_constrained_inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import test_util

import data
import fft_tree_constrained_inference as ffttci
import tf_lib
import vocabulary


class FFTTreeConstrainedTest(test_util.TensorFlowTestCase):

  def test_tree_constrained_inference(self):
    self.shared_tree_constrained_inference(get_testing_hps())

  def test_tree_constrained_inference_single_sentence_concat(self):
    self.shared_tree_constrained_inference(
        get_single_sentence_concat_testing_hps())

  def shared_tree_constrained_inference(self, testing_hps):

    np.set_printoptions(precision=4, suppress=True)

    testing_sent_tokens_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    testing_sent_tokens_1 = [str(t) for t in testing_sent_tokens_1]
    testing_sent_edu_ids_1 = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    testing_sent_parent_ids_1 = [1, 1, 2, 2, -1, -1, 2, 2, 3, 3]

    testing_sent_tokens_2 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    testing_sent_tokens_2 = [str(t) for t in testing_sent_tokens_2]
    testing_sent_edu_ids_2 = [5, 5, 5, 6, 7, 7, 7, 8, 8]
    testing_sent_parent_ids_2 = [6, 6, 6, -1, 6, 6, 6, 6, 6]

    dummy_abstract_sentence_1 = ["0", "0", "0", "0", "0", "0", "0", "0", "0",
                                 "0"]

    dummy_abstract_sentence_2 = ["0", "0", "0"]

    extract_labels_1 = [0 for _ in testing_sent_tokens_1]
    extract_labels_2 = [0 for _ in testing_sent_tokens_2]

    vocab_word_indices = range(11)
    dummy_vocab_indices = dict([(str(i), i) for i in vocab_word_indices])
    stem_indices = dummy_vocab_indices
    word_stems = dict([(str(w), str(w)) for w in vocab_word_indices])
    is_stop = [False for _ in vocab_word_indices]
    vocab = vocabulary.Vocabulary(dummy_vocab_indices, stem_indices, word_stems,
                                  is_stop, testing_hps.vocab_size)

    if testing_hps.single_sentence_concat:
      combined_toks = [["0"] + testing_sent_tokens_1 + testing_sent_tokens_2]
      combined_labels = [[1] + extract_labels_1 + extract_labels_2]
      combined_edu_ids = [-1] + testing_sent_edu_ids_1 + testing_sent_edu_ids_2
      combined_parent_ids = [
          -2
      ] + testing_sent_parent_ids_1 + testing_sent_parent_ids_2
      combined_edu_ids = [[i + 1 for i in combined_edu_ids]]
      combined_parent_ids = [[i + 1 for i in combined_parent_ids]]
      nyt_ex_1 = data.SummaryExample(0, combined_toks, combined_edu_ids,
                                     combined_parent_ids, combined_labels,
                                     dummy_abstract_sentence_1)
      nyt_ex_2 = data.SummaryExample(1, combined_toks, combined_edu_ids,
                                     combined_parent_ids, combined_labels,
                                     dummy_abstract_sentence_2)
    else:
      nyt_ex_1 = data.SummaryExample(
          0, [testing_sent_tokens_1, testing_sent_tokens_2],
          [testing_sent_edu_ids_1, testing_sent_edu_ids_2],
          [testing_sent_parent_ids_1, testing_sent_parent_ids_2],
          [extract_labels_1, extract_labels_2], dummy_abstract_sentence_1)

      nyt_ex_2 = data.SummaryExample(
          1, [testing_sent_tokens_1, testing_sent_tokens_2],
          [testing_sent_edu_ids_1, testing_sent_edu_ids_2],
          [testing_sent_parent_ids_1, testing_sent_parent_ids_2],
          [extract_labels_1, extract_labels_2], dummy_abstract_sentence_2)

    with self.test_session() as session:

      tf.set_random_seed(12)

      model_inp = ffttci.TreeInferenceInputs(testing_hps)

      ex_batch = ffttci.TreeInferenceBatch(testing_hps, model_inp,
                                           [nyt_ex_1, nyt_ex_2], vocab)

      inferencer = ffttci.TreeConstrainedInferencer()

      logit_shape = [testing_hps.batch_size, testing_hps.num_art_steps]

      word_logits = tf.constant(
          np.full(logit_shape, 0.0), dtype=tf.float32, shape=logit_shape)

      margs, samples, logz = inferencer.do_tree_inference(
          testing_hps, model_inp, word_logits)

      margs = tf.reshape(margs, [testing_hps.batch_size, -1])
      grad_logz = tf.gradients(logz, word_logits)[0]

      margs_np, samples_np, logz_np, grad_logz_np = session.run(
          [margs, samples, logz, grad_logz],
          ex_batch.feeds)

      emp_marg = np.average(samples_np, axis=1)
      emp_marg = np.reshape(emp_marg, [testing_hps.batch_size, -1])

      # sampled marginals should be pretty close to marginals calculated from BP
      self.assertNDArrayNear(margs_np, emp_marg, 0.05)
      # gradient of logz should be _very_ close to marginals calculated from BP
      self.assertNDArrayNear(margs_np, grad_logz_np, 0.001)
      # for k=3 example, logz should equal log(3)
      self.assertNear(1.08961229, logz_np[1], 0.01)


def get_single_sentence_concat_testing_hps():
  return tf_lib.HParams(
      init_scale=0.08,
      vocab_size=20,
      hidden_size=200,
      word_embedding_size=300,
      batch_size=2,
      min_art_steps=0,  # min/max length for article
      num_art_steps=20,
      min_abs_steps=0,  # min/max length for abstract
      num_abs_steps=1,
      num_samples=2000,  # number of sampled extractions
      single_sentence_concat=True,  # make one tree over all sentences
      max_num_spans=10,  # number of token spans in constraints
      max_num_sentences=1,
      max_tree_nodes_any_level=32,  # max width for BP tree
      # lambda for message damping at each level of BP tree,
      # improves numerical stability
      fft_tree_msg_damp_lambdas=[0.0, 0.0, 0.0, 0.0],
      tree_widths_at_level=[32, 32, 32, 32],  # width for each BP tree level
      fft_tree_widths_at_level=[32, 32, 32],  # width for each sum tree level
      global_fft_tree_width=32)  # width of global sum tree


def get_testing_hps():
  return tf_lib.HParams(
      init_scale=0.08,
      vocab_size=20,
      hidden_size=200,
      word_embedding_size=300,
      batch_size=2,
      min_art_steps=0,  # min/max length for article
      num_art_steps=19,
      min_abs_steps=0,  # min/max length for abstract
      num_abs_steps=1,
      num_samples=2000,  # number of sampled extractions
      single_sentence_concat=False,  # make one tree over all sentences
      max_num_spans=10,  # number of token spans in constraints
      max_num_sentences=2,
      max_tree_nodes_any_level=11,  # max width for BP tree
      # lambda for message damping at each level of BP tree,
      # improves numerical stability
      fft_tree_msg_damp_lambdas=[0.0, 0.0, 0.0],
      tree_widths_at_level=[16, 16, 16],  # width for each BP tree level
      fft_tree_widths_at_level=[16, 16],  # width for each sum tree level
      global_fft_tree_width=32)  # width of global sum tree


if __name__ == "__main__":
  tf.test.main()
