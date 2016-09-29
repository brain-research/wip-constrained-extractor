"""Tests for FFT Tree inference for k-cardinality potentials."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import test_util

import fft_tree_indep_inference as ffttii


class FFTTreeIndepTest(test_util.TensorFlowTestCase):

  def test_inference(self):
    with self.test_session() as session:

      tf.set_random_seed(12)

      ns = 10000
      rep = 1
      w = 8
      k = 6

      all_ys = tf.log(tf.reshape(tf.to_float(tf.range(0, w, 1)), [1, w]))
      all_ys = tf.tile(all_ys, [rep, 1])
      z_pots = tf.log(
          tf.reshape(
              tf.to_float(tf.equal(tf.range(0, w + 1, 1), k)), [1, w + 1]))
      z_pots = tf.tile(z_pots, [rep, 1])

      marg_t, samples_t, logz_t = ffttii.fft_tree_indep_vars(all_ys, z_pots, ns,
                                                             rep, w)
      grad_log_z = tf.gradients(logz_t, all_ys)

      marg, samples, _, glogz = session.run(
          [marg_t, samples_t, logz_t, grad_log_z])

      emp_marg = np.average(samples, axis=1)

      # sampled marginals should be pretty close to marginals calculated from BP
      self.assertNDArrayNear(marg, emp_marg, 0.01)
      # gradient of logz should be _very_ close to marginals calculated from BP
      self.assertNDArrayNear(marg, glogz, 0.001)


if __name__ == '__main__':
  tf.test.main()
