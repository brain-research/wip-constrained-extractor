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

"""Shared inputs and interfaces for summarizer models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import shared_util


class ModelInputs(object):
  """Input placeholders for document/summary batches."""

  def __init__(self, hps):

    self.hps = hps

    self.article = tf.placeholder(tf.int32, [hps.batch_size, hps.num_art_steps])
    self.article_extract_label = tf.placeholder(
        tf.int32, [hps.batch_size, hps.num_art_steps])
    self.article_len = tf.placeholder(tf.int32, [hps.batch_size])

    self.article_max_len = tf.to_int32(hps.num_art_steps)

    self.abstract_bag = tf.placeholder(tf.int32,
                                       [hps.batch_size, hps.vocab_size])
    self.abstract_len = tf.placeholder(tf.int32, [hps.batch_size])
    self.abstract_max_len = tf.reshape(tf.reduce_max(self.abstract_len), [])

    self.sliced_article = tf.slice(self.article, [0, 0], tf.pack(
        [hps.batch_size, self.article_max_len]))
    self.sliced_extract_label = tf.slice(
        self.article_extract_label, [0, 0],
        tf.pack([hps.batch_size, self.article_max_len]))

    self.article_mask = shared_util.create_mask(self.article_len,
                                                hps.num_art_steps)
    self.article_sliced_mask = tf.slice(self.article_mask, [0, 0], tf.pack(
        [hps.batch_size, self.article_max_len]))

    self.cur_batch_size = tf.maximum(
        1.0, tf.reduce_sum(tf.minimum(1.0, tf.to_float(self.article_len))))
    self.sample_batch_size = self.cur_batch_size * tf.to_float(hps.num_samples)


class SummarizerFeatures(object):
  """Features for extractor."""

  def __init__(self, model_inputs, rouge_scorer, hps):

    self.word_embedding = tf.get_variable(
        "word_embedding", [hps.vocab_size, hps.word_embedding_size])
    self.article_inputs = tf.nn.embedding_lookup(self.word_embedding,
                                                 model_inputs.sliced_article)

    self.stopworded_abstract_bag = tf.transpose(
        tf.sparse_tensor_dense_matmul(
            rouge_scorer.stem_projector_stopworded,
            tf.to_float(model_inputs.abstract_bag),
            adjoint_a=True,
            adjoint_b=True))

    with tf.variable_scope("article_enc"):

      article_outs = shared_util.deep_birnn(hps, self.article_inputs,
                                            model_inputs.article_len)

      self.article_feats = shared_util.relu(article_outs, hps.hidden_size)

    with tf.variable_scope("scorer"):
      self.word_logits = tf.reshape(
          shared_util.linear(self.article_feats, 1), [hps.batch_size, -1])


class Extractor(object):
  """Inference and likelihood for extracting summaries from docs."""

  def __init__(self, model_inputs, summarizer_features, hps):
    del model_inputs, summarizer_features, hps  # unused by base class
    self.log_z = None
    self.gold_log_likelihood = None
    self.sample_log_likelihood = None
    self.marginals = None
    self.map_prediction = None
    self.samples = None


class ExtractorLoss(object):
  """Specifies loss function for learning extractors."""

  def __init__(self, model_inputs, summarizer_features, extractor):
    del model_inputs, summarizer_features, extractor  # unused by base class
    self.loss = None
