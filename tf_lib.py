"""Utility functions for working with TensorFlow."""

import ast


class HParams(object):

  """Creates an object for passing around hyperparameter values.

  Use the parse method to overwrite the default hyperparameters with values
  passed in as a string representation of a Python dictionary mapping
  hyperparameters to values.
  Ex.
  hparams = tf_lib.HParams(batch_size=128, hidden_size=256)
  hparams.parse('{"hidden_size":512}')
  assert hparams.batch_size == 128
  assert hparams.hidden_size == 512
  """

  def __init__(self, **init_hparams):
    object.__setattr__(self, 'keyvals', init_hparams)

  def __getattr__(self, key):
    return self.keyvals.get(key)

  def __setattr__(self, key, value):
    """Returns None if key does not exist."""
    self.keyvals[key] = value

  def parse(self, string):
    new_hparams = ast.literal_eval(string)
    return HParams(**dict(self.keyvals, **new_hparams))

  def values(self):
    return self.keyvals