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

"""Tests for learning summarizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.framework import test_util

import compressive_summarizer_models as csm
import data
import vocabulary

import tf_lib


class LearningTest(test_util.TensorFlowTestCase):

  def shared_learning_helper(self, hps, iters=15):
    with self.test_session() as session:

      tf.set_random_seed(12)

      tokstr = "0 1 2 3 4 5 6 7 8 9"
      summstr = "0 1 2 3 4"
      toks = tokstr.split(" ")
      vocab_words = list(set(toks))
      vocab_words.append("<UNK>")
      word_indices = dict(
          {(w, i)
           for w, i in zip(vocab_words, xrange(len(vocab_words)))})
      word_stems = dict({(w, w) for w in vocab_words})
      stem_indices = word_indices
      is_stop = [0] * hps.vocab_size
      vocab = vocabulary.Vocabulary(word_indices, stem_indices, word_stems,
                                    is_stop, hps.vocab_size)

      edu_ids = [0] * len(toks)
      parent_ids = [-1] * len(toks)
      extract_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

      summ_toks = summstr.split(" ")

      ex = data.SummaryExample(0, [toks], [edu_ids], [parent_ids],
                               [extract_labels], [summ_toks])

      model = csm.CompressiveSummarizerModel(hps, vocab)

      batch = data.SummaryBatch(hps, model.model_inputs, [ex], vocab)

      tf.initialize_all_variables().run()

      for _ in xrange(iters):
        loss, avg_rouge, _ = session.run(
            [model.loss, model.rouge_computer.cur_avg_sample_rouge,
             model.train_op], batch.feeds)

      return loss, avg_rouge

  def test_learning_rouge_reinforce(self):
    hps = get_test_hparams()
    hps.extractor = "indep_card"
    hps.extractor_loss = "rouge_reinforce"
    hps.log_z_reg = 0.1
    hps.log_z_cstrt = 500.0
    _, avg_rouge = self.shared_learning_helper(hps, iters=15)
    # and avg ROUGE-1 should be close to 1.0
    self.assertNear(1.0, avg_rouge, 0.05)

  def test_learning_max_likelihood(self):
    hps = get_test_hparams()
    hps.extractor = "indep_card"
    hps.extractor_loss = "oracle_xent"
    loss, avg_rouge = self.shared_learning_helper(hps, iters=15)
    # after 15 iters on the toy example, xent should be close to 0.1 nats
    self.assertNear(0.1, loss, 0.01)
    # and avg ROUGE-1 should be close to 1.0
    self.assertNear(1.0, avg_rouge, 0.05)


def get_test_hparams():
  return tf_lib.HParams(
      init_scale=0.08,
      vocab_size=20,
      hidden_size=200,
      word_embedding_size=300,
      batch_size=1,
      min_art_steps=0,  # min/max length for article
      num_art_steps=20,
      min_abs_steps=0,  # min/max length for abstract
      num_abs_steps=100,
      max_grad_norm=5.0,
      learning_rate=0.001,
      epsilon=1e-6,  # epsilon for Adam optimizer
      max_epoch=10000,
      num_samples=10,  # number of sampled extractions
      max_train_examples=1000,
      max_dev_examples=1000,
      extractor="indep_card",
      extractor_loss="oracle_xent",
      log_z_reg=0.0,  # regularizer for extractor logZ (numerical stability)
      log_z_cstrt=500.0,  # constraint for extractor logZ (numerical stability)
      max_num_sentences=100,  # max # of sentences in article
      max_num_spans=300)  # max # of EDUs in article


if __name__ == "__main__":
  tf.test.main()
