import sys

import numpy as np
import pytest

import autodidax as adx


def test_autodiff_basic():
  x = 3.0
  y, sin_deriv_at_3 = adx.jvp(adx.sin, (x,), (1.0,))
  print(y)
  print(sin_deriv_at_3)
  print(adx.cos(3.0))

  assert np.allclose(y, np.sin(x))
  assert np.allclose(sin_deriv_at_3, np.cos(x))


def test_autodiff_complex():
  def f(x):
    y = adx.sin(x) * 2. + 1
    z = -y + x
    return z

  x, x_dot = 3., 1.
  y, y_dot = adx.jvp(f, (x,), (x_dot,))
  print(y)
  print(y_dot)

  assert np.allclose(y, 1.7177599838802657)
  assert np.allclose(y_dot, 2.979984993200891)


def test_autodiff_higher_order():
  def deriv(f):
    return lambda x: adx.jvp(f, (x,), (1.,))[1]

  a = deriv(adx.sin)(3.)
  b = deriv(deriv(adx.sin))(3.)
  c = deriv(deriv(deriv(adx.sin)))(3.)
  d = deriv(deriv(deriv(deriv(adx.sin))))(3.)

  assert np.allclose(a, -0.9899924966004454)
  assert np.allclose(b, -0.1411200080598672)
  assert np.allclose(c, 0.9899924966004454)
  assert np.allclose(d, 0.1411200080598672)


def test_autodiff_control_flow():
  def deriv(f):
    return lambda x: adx.jvp(f, (x,), (1.,))[1]

  def f(x):
    if x > 0.:  # Python control flow
      return 2. * x
    else:
      return x

  assert np.allclose(deriv(f)(3.), 2.0)
  assert np.allclose(deriv(f)(-3.), 1.0)


if __name__ == "__main__":
  retcode = pytest.main(["-qq"])
  if retcode != 0:
    sys.exit(retcode)
