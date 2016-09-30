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

"""Extractors, loss functions, main model for learning and inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import fft_tree_indep_inference as ffttii
import model_base
import rouge_scorer as rs
import shared_util


class ExtractionRougeComputer(object):
  """Compute micro/macro ROUGE scores for samples and predictions."""

  def __init__(self, hps, model_inputs, extractor, rouge_scorer):

    self.rouge_scorer = rouge_scorer

    # ROUGE scores for model samples

    st_margs = tf.reshape(extractor.marginals, [hps.batch_size, 1, -1])

    st_sample_rouge_overlap, rouge_gold_len = self.get_rouge_stats(
        extractor.samples + st_margs - tf.stop_gradient(st_margs),
        model_inputs.sliced_article, model_inputs.article_max_len,
        model_inputs.abstract_bag, hps.batch_size, hps.num_samples,
        hps.vocab_size)

    sample_rouge_overlap, rouge_gold_len = self.get_rouge_stats(
        extractor.samples, model_inputs.sliced_article,
        model_inputs.article_max_len, model_inputs.abstract_bag, hps.batch_size,
        hps.num_samples, hps.vocab_size)

    mr, umr, zmr = self.make_rouge_stat_counters("sample", sample_rouge_overlap,
                                                 rouge_gold_len)

    self.sample_micro_rouge = mr
    self.update_sample_micro_rouge_stats = umr
    self.zero_sample_micro_rouge_stats = zmr

    self.cur_sample_rouge = sample_rouge_overlap / tf.maximum(1.0,
                                                              rouge_gold_len)
    self.cur_st_sample_rouge = st_sample_rouge_overlap / tf.maximum(
        1.0, rouge_gold_len)
    self.cur_avg_sample_rouge = tf.reduce_sum(
        self.cur_sample_rouge) / model_inputs.sample_batch_size

    self.sample_micro_rouge_summary = tf.scalar_summary("Sample micro ROUGE",
                                                        self.sample_micro_rouge)

    # ROUGE scores for model MAP predictions

    map_rouge_overlap, rouge_gold_len = self.get_rouge_stats(
        extractor.map_prediction, model_inputs.sliced_article,
        model_inputs.article_max_len, model_inputs.abstract_bag, hps.batch_size,
        1, hps.vocab_size)

    mr, umr, zmr = self.make_rouge_stat_counters("map_pred", map_rouge_overlap,
                                                 rouge_gold_len)

    self.map_micro_rouge = mr
    self.update_map_micro_rouge_stats = umr
    self.zero_map_micro_rouge_stats = zmr

    self.map_micro_rouge_summary = tf.scalar_summary("MAP micro ROUGE",
                                                     self.map_micro_rouge)

    self.cur_map_rouge = map_rouge_overlap / tf.maximum(1.0, rouge_gold_len)
    self.cur_avg_map_rouge = tf.reduce_sum(
        self.cur_map_rouge) / model_inputs.cur_batch_size

    # ROUGE scores for oracle extraction

    gold_rouge_overlap, rouge_gold_len = self.get_rouge_stats(
        model_inputs.sliced_extract_label, model_inputs.sliced_article,
        model_inputs.article_max_len, model_inputs.abstract_bag, hps.batch_size,
        1, hps.vocab_size)

    mr, umr, zmr = self.make_rouge_stat_counters("gold", gold_rouge_overlap,
                                                 rouge_gold_len)

    self.gold_micro_rouge = mr
    self.update_gold_micro_rouge_stats = umr
    self.zero_gold_micro_rouge_stats = zmr

    self.gold_micro_rouge_summary = tf.scalar_summary("Gold micro ROUGE",
                                                      self.gold_micro_rouge)

    self.merged_micro_rouge_summaries = tf.merge_summary([
        self.sample_micro_rouge_summary, self.map_micro_rouge_summary,
        self.gold_micro_rouge_summary
    ])

    self.update_micro_rouge_stats = tf.group(
        self.update_sample_micro_rouge_stats, self.update_map_micro_rouge_stats,
        self.update_gold_micro_rouge_stats)

    self.zero_micro_rouge_stats = tf.group(self.zero_sample_micro_rouge_stats,
                                           self.zero_map_micro_rouge_stats,
                                           self.zero_gold_micro_rouge_stats)

  def make_bag_of_words(self, active_words, doc_tokens, batch_size, num_samples,
                        doc_len, vocab_size):
    """Turn sequences of words with extraction labels into a bag of words.

    Args:
      active_words: batch of binary masks indicating picked words.
      doc_tokens: batch of token sequences for each document.
      batch_size: number of sequences in batch.
      num_samples: number of samples of masks for each sequence.
      doc_len: length of sequences.
      vocab_size: size of vocabulary.

    Returns:
      dense_pred_counts: batch of bags of words.
    """

    # can't make a sparse count tensor directly since doesn't support repeated
    # words -- make a new sparse tensor with extra indices to handle repeats
    with tf.variable_scope("bag_of_words"):

      batch_idx = shared_util.repeat_row(num_samples * doc_len,
                                         tf.range(0, batch_size, 1))
      sample_idx = tf.tile(
          shared_util.repeat_row(doc_len, tf.range(0, num_samples, 1)),
          [batch_size])
      tok_idx = tf.tile(tf.range(0, doc_len, 1), [batch_size * num_samples])
      vocab_idx = tf.tile(
          tf.reshape(doc_tokens, [batch_size, 1, doc_len]), [1, num_samples, 1])
      active_tok_indices = tf.concat(1, [
          tf.reshape(batch_idx, [-1, 1]), tf.reshape(sample_idx, [-1, 1]),
          tf.reshape(tok_idx, [-1, 1]), tf.reshape(vocab_idx, [-1, 1])
      ])

      active_toks_sparse = tf.SparseTensor(
          indices=tf.to_int64(active_tok_indices),
          values=tf.reshape(active_words, [-1]),
          shape=tf.to_int64(
              tf.pack([batch_size, num_samples, doc_len, vocab_size])))

      dense_pred_counts = tf.sparse_reduce_sum(
          active_toks_sparse, reduction_axes=2)

    return dense_pred_counts

  def get_rouge_stats(self, active_words, doc_tokens, doc_len, abstract_bag,
                      batch_size, num_samples, vocab_size):
    """Get sufficient statistics for ROUGE-1 recall computation.

    Args:
      active_words: batch of binary masks indicating picked words.
      doc_tokens: batch of token sequences for each document.
      doc_len: length of documents.
      abstract_bag: bags of words for true abstracts.
      batch_size: number of sequences in batch.
      num_samples: number of samples of masks for each sequence.
      vocab_size: size of vocabulary.

    Returns:
      rouge_overlap: overlap of predicted with gold summary for ROUGE.
      rouge_gold_len: counts of true summary lengths for ROUGE.
    """

    dense_extract_bags = self.make_bag_of_words(active_words, doc_tokens,
                                                batch_size, num_samples,
                                                doc_len, vocab_size)

    replicated_abstract_bags = tf.tile(
        tf.expand_dims(abstract_bag, 1), [1, num_samples, 1])

    flat_replicated_abstract_bags = tf.reshape(replicated_abstract_bags,
                                               [-1, vocab_size])
    flat_dense_extract_bags = tf.reshape(dense_extract_bags, [-1, vocab_size])

    flat_overlap, flat_gold_len = self.rouge_scorer.get_rouge_recall_suff_stats(
        tf.to_float(flat_dense_extract_bags),
        tf.to_float(flat_replicated_abstract_bags))

    rouge_overlap = tf.reshape(flat_overlap, [batch_size, num_samples])
    rouge_gold_len = tf.reshape(flat_gold_len, [batch_size, num_samples])

    return rouge_overlap, rouge_gold_len

  def make_rouge_stat_counters(self, varname, overlap, gold_len):
    """Make variables and ops to handle keeping track of global (micro) ROUGE.

    Args:
      varname: name of ROUGE counter variable group.
      overlap: tensor with overlap counts.
      gold_len: tensor with length of gold summaries.

    Returns:
      micro_rouge
      update_micro_rouge_stats
      zero_micro_rouge_stats
    """
    total_rouge_overlap = tf.get_variable(
        "total_%s_rouge_overlap" % varname,
        initializer=tf.zeros_initializer([]))
    total_rouge_gold_len = tf.get_variable(
        "total_%s_rouge_gold_len" % varname,
        initializer=tf.zeros_initializer([]))
    micro_rouge = tf.get_variable(
        "%s_micro_rouge" % varname, initializer=tf.zeros_initializer([]))

    zero_total_rouge_overlap = tf.assign(total_rouge_overlap, 0.0)
    zero_total_rouge_gold_len = tf.assign(total_rouge_gold_len, 0.0)
    zero_micro_rouge = tf.assign(micro_rouge, 0.0)

    zero_micro_rouge_stats = tf.group(zero_total_rouge_overlap,
                                      zero_total_rouge_gold_len,
                                      zero_micro_rouge)

    update_total_rouge_overlap = tf.assign_add(total_rouge_overlap,
                                               tf.reduce_sum(overlap))
    update_total_rouge_gold_len = tf.assign_add(total_rouge_gold_len,
                                                tf.reduce_sum(gold_len))
    update_micro_rouge = tf.assign(micro_rouge, update_total_rouge_overlap /
                                   update_total_rouge_gold_len)

    update_micro_rouge_stats = tf.group(update_total_rouge_overlap,
                                        update_total_rouge_gold_len,
                                        update_micro_rouge)

    return micro_rouge, update_micro_rouge_stats, zero_micro_rouge_stats


class OracleXentExtractorLoss(model_base.ExtractorLoss):
  """Train using cross entropy with supervised extraction labels."""

  def __init__(self, model_inputs, summarizer_features, extractor,
               rouge_computer):
    del rouge_computer  # don't need ROUGE for xent objective
    del summarizer_features  # don't need features for xent objective
    self.loss = -tf.reduce_sum(extractor.gold_log_likelihood) / tf.to_float(
        tf.reduce_sum(model_inputs.article_len))


class ROUGEReinforceExtractorLoss(model_base.ExtractorLoss):
  """Train using REINFORCE to optimize ROUGE-1 recall score with summary."""

  def __init__(self, model_inputs, summarizer_features, extractor,
               rouge_computer):
    del summarizer_features  # don't need features for REINFORCE objective

    hps = model_inputs.hps

    sample_log_likelihood = extractor.sample_log_likelihood

    sample_reward = tf.stop_gradient(rouge_computer.cur_sample_rouge)

    b = hps.batch_size
    k = hps.num_samples
    kf = tf.to_float(k)
    sample_indicators = tf.one_hot(tf.tile(tf.range(0, k, 1), [b]), k)
    leave_one_out_mask = 1.0 - tf.reshape(sample_indicators, [b, k, k])

    sample_avg = 1.0 / (kf - 1.0) * leave_one_out_mask * tf.reshape(
        sample_reward, [b, 1, k])

    sample_avg = tf.reduce_sum(sample_avg, 2)

    baseline = sample_avg

    reinforce_cost = -(sample_reward - baseline) * sample_log_likelihood

    self.loss = tf.reduce_sum(
        -sample_reward + reinforce_cost - tf.stop_gradient(
            reinforce_cost)) / model_inputs.sample_batch_size


class IndependentCardinalityPotentialsExtractor(model_base.Extractor):
  """Extractor that uses cardinality potentials to limit # of extractions."""

  def __init__(self, model_inputs, summarizer_features, hps):

    self.hps = hps

    article_max_len = model_inputs.article_max_len
    article_len = model_inputs.article_len
    abstract_len = model_inputs.abstract_len
    batch_size = hps.batch_size
    num_samples = hps.num_samples

    self.word_logits = word_logits = self.get_extractor_logits(
        hps, model_inputs, summarizer_features)

    word_logits_padded = tf.pad(word_logits, tf.pack(
        [[0, 0], [0, hps.num_art_steps - article_max_len]]))
    word_logits_padded.set_shape([batch_size, hps.num_art_steps])

    logit_mask = shared_util.create_log_mask(
        tf.maximum(1, article_len), hps.num_art_steps)
    masked_word_logits = word_logits_padded + logit_mask

    # constraint that number of selected things must be exactly the length
    # of the gold abstract, with log-score close to -inf on other counts
    cardinality_pots = tf.one_hot(
        abstract_len, hps.num_art_steps + 1, on_value=1.0, off_value=-100.0)

    with tf.name_scope("fft_tree"):
      tok_marg, tok_samples, log_z = ffttii.fft_tree_indep_vars(
          masked_word_logits, cardinality_pots, hps.num_samples,
          batch_size, hps.num_art_steps, tf.maximum(1, abstract_len))

    self.log_z = log_z

    sliced_tok_marg = tf.slice(tok_marg, [0, 0], [batch_size, article_max_len])
    sliced_tok_marg *= model_inputs.article_sliced_mask

    sliced_tok_samples = tf.slice(tok_samples, [0, 0, 0],
                                  [batch_size, num_samples, article_max_len])
    sliced_tok_samples *= tf.reshape(model_inputs.article_sliced_mask,
                                     [batch_size, 1, -1])

    log_prob_gold_words = tf.reduce_sum(
        tf.to_float(model_inputs.sliced_extract_label) *
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

  def get_log_likelihood(self, samples):
    log_prob_active_words = tf.reduce_sum(
        tf.to_float(samples) * tf.reshape(self.word_logits,
                                          [self.hps.batch_size, 1, -1]), 2)
    log_prob_active_words -= tf.reshape(self.log_z, [self.hps.batch_size, 1])
    return log_prob_active_words

  def get_extractor_logits(self, hps, model_inputs, summarizer_features):
    del hps, model_inputs  # unused for the standard extractor logits
    return summarizer_features.word_logits


class CompressiveSummarizerModel(object):
  """Main summarizer model class with learning and inference graphs."""

  def __init__(self, hps, vocab):

    extractors = {"indep_card": IndependentCardinalityPotentialsExtractor}

    losses = {"oracle_xent": OracleXentExtractorLoss,
              "rouge_reinforce": ROUGEReinforceExtractorLoss}

    self.vocab = vocab

    with tf.name_scope("model_inputs"):
      self.model_inputs = model_inputs = model_base.ModelInputs(hps)

    with tf.name_scope("rouge_scorer"):
      self.rouge_scorer = rouge_scorer = rs.ROUGEScorer(vocab)

    with tf.name_scope("summarizer_features"):
      self.summarizer_features = model_base.SummarizerFeatures(model_inputs,
                                                               rouge_scorer,
                                                               hps)
      summarizer_features = self.summarizer_features

    with tf.name_scope("extractor"):
      self.extractor = extractor = extractors[hps.extractor](
          model_inputs, summarizer_features, hps)

    with tf.name_scope("rouge_computer"):
      self.rouge_computer = rouge_computer = ExtractionRougeComputer(
          hps, model_inputs, extractor, rouge_scorer)

    with tf.name_scope("loss"):
      self.extractor_loss = extractor_loss = losses[hps.extractor_loss](
          model_inputs, summarizer_features, extractor, rouge_computer)

    # Gradients and train op

    tvars = tf.trainable_variables()

    if hps.log_z_reg > 0.0:
      log_z_reg = hps.log_z_reg * tf.reduce_sum(
          tf.maximum(tf.abs(self.extractor.log_z) - hps.log_z_cstrt,
                     0.0)) / hps.batch_size
    else:
      log_z_reg = 0.0

    final_loss = extractor_loss.loss + log_z_reg

    self.loss = final_loss

    raw_grads = tf.gradients(final_loss, tvars)

    grads = raw_grads

    grads, _ = tf.clip_by_global_norm(raw_grads, hps.max_grad_norm)

    optimizer = tf.train.AdamOptimizer(hps.learning_rate, hps.epsilon)

    self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    self.global_step = tf.get_variable(
        "global_step", initializer=tf.zeros_initializer([]))

    update_global_step = tf.assign_add(self.global_step, 1.0)

    self.train_op = tf.group(self.train_op,
                             rouge_computer.update_micro_rouge_stats,
                             update_global_step)

    # Loss summary

    self.merged_summaries = tf.merge_summary(
        [tf.scalar_summary("Loss", extractor_loss.loss)])
