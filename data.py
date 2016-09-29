"""Classes and helpers for summary data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class SummaryExample(object):
  """Features for one article/summary paired example.

  Attributes:
    ex_id: integer ID for example.
    article_sentences: list of string token lists for article sentences.
    article_edu_ids: list of integer IDs for the containing EDU (elementary
      discourse unit) for each token.
    article_parent_ids: list of integer IDs for the parent EDU for each token
      according to the discourse parse.
    extract_labels: list of binary integers for each token indicating whether
      the token is in the oracle extraction.
    abstract_sentences: list of string token lists for abstract sentences.
  """

  def __init__(self, ex_id, article_sentences, article_edu_ids,
               article_parent_ids, extract_labels, abstract_sentences):
    self.ex_id = ex_id
    self.article_sentences = article_sentences
    self.extract_labels = extract_labels
    self.abstract_sentences = abstract_sentences
    self.article_edu_ids = article_edu_ids
    self.article_parent_ids = article_parent_ids


class SummaryBatch(object):
  """Model feeds for batch for extractive inference.

  Attributes:
    feeds: Feed dictionary mapping model placeholders to numpy tensors.
  """

  def __init__(self, hps, model_inputs, examples, vocab):
    """Constructor for SummaryBatch.

    Arguments:
      hps: bag of hyperparameters for model.
      model_inputs: model placeholders for input tensors.
      examples: list of SummaryExample.
      vocab: Vocabulary object.
    """
    self.feeds = feeds = {}
    feeds[model_inputs.article] = article = np.zeros(
        [hps.batch_size, hps.num_art_steps], dtype=np.int64)
    feeds[model_inputs.article_len] = article_len = np.zeros(
        [hps.batch_size], dtype=np.int64)

    feeds[model_inputs.article_extract_label] = extract_label = np.zeros(
        [hps.batch_size, hps.num_art_steps], dtype=np.int64)

    feeds[model_inputs.abstract_bag] = abstract_bag = np.zeros(
        [hps.batch_size, hps.vocab_size], dtype=np.int64)
    feeds[model_inputs.abstract_len] = abstract_len = np.zeros(
        [hps.batch_size], dtype=np.int64)

    i = 0
    for nyex in examples:
      article_sentence_tokens = [
          tok.lower() for sent in nyex.article_sentences for tok in sent
      ]
      article_sentence_tokens = article_sentence_tokens[:hps.num_art_steps]
      extract_labels = [lab for sent in nyex.extract_labels for lab in sent]
      extract_labels = extract_labels[:hps.num_art_steps]
      abstract_sentence_tokens = [
          tok.lower() for sent in nyex.abstract_sentences for tok in sent
      ]
      j = 0
      for tok, lab in zip(article_sentence_tokens, extract_labels):
        article[i, j] = vocab.word_indices[tok]
        extract_label[i, j] = lab
        j += 1
      article_len[i] = j
      for tok in abstract_sentence_tokens:
        abstract_bag[i, vocab.word_indices[tok]] += 1
        abstract_len[i] += 1
      i += 1
