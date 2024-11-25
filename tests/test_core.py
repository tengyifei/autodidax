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


if __name__ == "__main__":
  retcode = pytest.main(["-qq"])
  if retcode != 0:
    sys.exit(retcode)
