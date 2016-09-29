"""Vocabulary handling stopwords and stems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


class Vocabulary(object):
  """Vocabulary for handling stopwords and stems.

  Attributes:
    word_indices: map from word type to vocab index.
    stem_indices: map from stem to vocab index.
    word_stem: map from word to stem.
    is_stop: boolean array indicating whether word type is stopword.
    words: array of words in vocabulary.
    vocab_size: size of vocabulary.
  """

  def __init__(self, word_indices, stem_indices, word_stems, is_stop,
               vocab_size):
    self.word_indices = word_indices
    self.stem_indices = stem_indices
    self.word_stems = word_stems
    self.is_stop = is_stop
    self.words = [""] * vocab_size
    for w, i in self.word_indices.iteritems():
      self.words[i] = w
    self.vocab_size = vocab_size


def parse_vocabulary(lines, max_vocab_size=None):
  """Parse vocabulary from lines of tab-separated text.

  Args:
    lines: list of strings, tab-delimited with format
      (word_type, count, stem, is_stopword).
    max_vocab_size: cutoff for vocab.

  Returns:
    vocab: Vocabulary object
  """
  stem_ctr = collections.Counter()
  word_indices = collections.defaultdict(int)
  word_stems = collections.defaultdict(lambda: "<UNK>")
  is_stop = []
  i = 0
  for line in lines:
    if max_vocab_size is not None and i >= max_vocab_size:
      break
    segs = line.strip().split("\t")
    word, _, stem, stop = segs
    stem_ctr[stem] += 1
    word_indices[word] = i
    word_stems[word] = stem
    is_stop.append(int(stop))
    i += 1

  if max_vocab_size is not None:
    vocab_size = max_vocab_size
  else:
    vocab_size = i

  stem_cts_sorted = sorted(stem_ctr.items(), key=lambda x: x[1], reverse=True)
  stem_indices = {}
  for i in xrange(len(stem_cts_sorted)):
    stem_indices[stem_cts_sorted[i][0]] = i
  unk_stem_index = stem_indices["<UNK>"]
  stem_indices = collections.defaultdict(lambda: unk_stem_index, stem_indices)

  return Vocabulary(word_indices, stem_indices, word_stems, is_stop,
                    vocab_size)


def read_vocabulary(file_name, max_vocab_size=None):
  """Read vocabulary from file."""
  data = open(file_name)
  dat = data.read()
  lines = dat.splitlines()
  return parse_vocabulary(lines, max_vocab_size)
