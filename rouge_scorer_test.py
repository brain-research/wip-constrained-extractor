"""Tests for rouge_scorer module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import test_util

import rouge_scorer
import vocabulary


class ROUGEScorerTest(test_util.TensorFlowTestCase):

  def test_rouge(self):
    with self.test_session() as session:

      vocab_string = """<UNK>\t1\t<UNK>\t1
foo\t1\tfoo\t1
bar\t1\tba\t0
baz\t1\tba\t0"""

      vocab = vocabulary.parse_vocabulary(vocab_string.split("\n"))

      pred_doc = "bar"
      gold_doc = "foo bar bar"

      # foo is a stopword and baz and bar have same stem, so this
      # should have ROUGE=1.0

      def make_bag(doc, vocab):
        bag = np.zeros([len(vocab.words)], dtype=np.float32)
        for tok in doc.split(" "):
          bag[vocab.word_indices[tok]] += 1.0
        return bag

      pred_bag = tf.expand_dims(make_bag(pred_doc, vocab), 0)
      gold_bag = tf.expand_dims(make_bag(gold_doc, vocab), 0)

      scorer = rouge_scorer.ROUGEScorer(vocab)

      rouge = scorer.get_rouge_recall(pred_bag, gold_bag)

      rouge_np = session.run([rouge])

      self.assertNear(rouge_np[0], 1.0, 0.0001)


if __name__ == "__main__":
  tf.test.main()
