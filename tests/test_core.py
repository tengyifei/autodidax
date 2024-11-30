import sys

import numpy as np
import pytest

import autodidax as adx


def test_eval():
  def scalar(x, y):
    a = 2.0 * adx.sin(x)
    b = 3.0 * adx.cos(y)
    c = a + (-b)
    return c

  def multi_output(x, y):
    return x > y, x < y

  assert np.allclose(scalar(2.0, 1.5), 1.6063832486482548)
  assert multi_output(2.0, 1.5) == (True, False)


def test_eval_complex():
  x = np.array([1., 2., 3.])
  y = np.array([4., 5., 6.])
  z = adx.reduce_sum(adx.mul(x, y))
  assert isinstance(z, np.number)
  assert np.allclose(z, 32.)
  x = np.array([[1., 2.], [3., 4.]])
  y = adx.transpose(x, perm=(1, 0))
  assert isinstance(y, np.ndarray)
  assert np.allclose(y, [[1., 3.], [2., 4.]])
  x = np.array([1., 2., 3.])
  y = adx.broadcast(x, (3, 3), (0,))
  assert isinstance(y, np.ndarray)
  assert np.allclose(y, [[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])


if __name__ == "__main__":
  retcode = pytest.main(["-qq"])
  if retcode != 0:
    sys.exit(retcode)
