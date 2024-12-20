from contextlib import contextmanager
from typing import Any, List, NamedTuple, Optional, Tuple, Type

import numpy as np

# --- Base Interpreter Classes ---


class Interpreter(NamedTuple):
  level: int
  global_data: Optional[Any] = None
  '''
  The interpreter base class.
  Creates boxed up values (Tracers) and unpacks them to apply interpretation rules.
  Determines the interpretation order with attribute `level`.
  '''
  def __gt__(self, other):
    return self.level > other.level

  def __lt__(self, other):
    return self.level < other.level

  def pure(self, val):
    '''Convert non-Tracer constants to Tracer objects'''
    raise NotImplementedError

  def lift(self, val):
    '''Raise the level of Tracer objects to this Trace object'''
    raise NotImplementedError

  def interp(self, prim, tracers, params):
    raise NotImplementedError


class Tracer:
  _interpreter: Interpreter
  __array_priority__ = 1000
  '''
  Boxes up an operand value of an interpretation rule (Trace).
  '''
  @property
  def aval(self):
    '''Returns the abstract value.'''
    raise NotImplementedError

  @property
  def interpreter(self):
    return self._interpreter

  def full_lower(self):
    return self

  def __add__(self, other):
    return self.aval._add(self, other)

  def __radd__(self, other):
    return self.aval._radd(self, other)

  def __mul__(self, other):
    return self.aval._mul(self, other)

  def __rmul__(self, other):
    return self.aval._rmul(self, other)

  def __neg__(self):
    return self.aval._neg(self)

  def __gt__(self, other):
    return self.aval._gt(self, other)

  def __lt__(self, other):
    return self.aval._lt(self, other)

  def __bool__(self):
    return self.aval._bool(self)

  def __nonzero__(self):
    return self.aval._nonzero(self)

  def __getattr__(self, name):
    return getattr(self.aval, name)


# --- Primitive Operators ---


class Primitive(NamedTuple):
  name: str


class PrimitiveToken:
  '''
  List of primitive tokens.
  Each primitive token corresponds to a primitive implementation rule.
  '''
  add = Primitive('add')
  mul = Primitive('mul')
  neg = Primitive('neg')
  sin = Primitive('sin')
  cos = Primitive('cos')
  greater = Primitive('greater')
  less = Primitive('less')
  transpose = Primitive('transpose')
  broadcast = Primitive('broadcast')
  reduce_sum = Primitive('reduce_sum')
  xla_call = Primitive('xla_call')


PRIM_TOK = PrimitiveToken()


class PrimitiveOperator:
  '''
  List of primitive operators.
  An operator intercepts its operands and pass them to interpreters.
  '''
  add = staticmethod(lambda x, y: bind(PRIM_TOK.add, x, y))
  mul = staticmethod(lambda x, y: bind(PRIM_TOK.mul, x, y))
  neg = staticmethod(lambda x: bind(PRIM_TOK.neg, x))
  sin = staticmethod(lambda x: bind(PRIM_TOK.sin, x))
  cos = staticmethod(lambda x: bind(PRIM_TOK.cos, x))
  greater = staticmethod(lambda x, y: bind(PRIM_TOK.greater, x, y))
  less = staticmethod(lambda x, y: bind(PRIM_TOK.less, x, y))

  @staticmethod
  def transpose(x, perm):
    return bind(PRIM_TOK.transpose, x, perm=perm)

  @staticmethod
  def broadcast(x, shape, axes):
    return bind(PRIM_TOK.broadcast, x, shape=shape, axes=axes)

  @staticmethod
  def reduce_sum(x, axis=None):
    if axis is None:
      axis = tuple(range(np.ndim(x)))
    if isinstance(axis, int):
      axis = (axis,)
    return bind(PRIM_TOK.reduce_sum, x, axis=axis)


POP = PrimitiveOperator()

# --- Evaluation Interpreter ---

EVAL_RULES = {
    PRIM_TOK.add: lambda x, y: np.add(x, y),
    PRIM_TOK.mul: lambda x, y: np.multiply(x, y),
    PRIM_TOK.neg: lambda x: np.negative(x),
    PRIM_TOK.sin: lambda x: np.sin(x),
    PRIM_TOK.cos: lambda x: np.cos(x),
    PRIM_TOK.greater: lambda x, y: np.greater(x, y),
    PRIM_TOK.less: lambda x, y: np.less(x, y),
    PRIM_TOK.transpose: lambda x, *, perm: np.transpose(x, perm),
    PRIM_TOK.reduce_sum: lambda x, *, axis: np.sum(x, axis),
    PRIM_TOK.broadcast: lambda x, *, shape, axes: np.broadcast_to(
        np.expand_dims(x, sorted(axes)), shape),
}


class EvalInterpreter(Interpreter):
  def pure(self, val):
    return val

  def lift(self, val):
    return val

  def interp(self, prim, tracers, params):
    eval_rule = EVAL_RULES[prim]
    output = eval_rule(*tracers, **params)
    return output


# --- Core Machinery ---

INTERPRETER_STACK: List[Interpreter] = [EvalInterpreter(level=0)]
DYNAMIC_INTERPRETER: Optional[Interpreter] = None
JAX_TYPES = {bool, int, float, np.bool_, np.int32, np.int64, np.float32, np.float64, np.ndarray}


@contextmanager
def new_interpreter(ty: Type['Interpreter'], global_data=None):
  '''
  Creates a new interpreter of the given type and pushes it into the trace stack.
  '''
  level = len(INTERPRETER_STACK)
  interpreter = ty(level, global_data)
  INTERPRETER_STACK.append(interpreter)
  try:
    yield interpreter
  finally:
    INTERPRETER_STACK.pop()


@contextmanager
def new_dynamic_interpreter(interpreter: Interpreter):
  '''
  Allows the given interpreter to stay on top of the interpreter stack,
  effectively stashing away all lower-leveled interpreters.
  '''
  global DYNAMIC_INTERPRETER
  prev_dyn_interp, DYNAMIC_INTERPRETER = DYNAMIC_INTERPRETER, interpreter
  try:
    yield
  finally:
    DYNAMIC_INTERPRETER = prev_dyn_interp


def bind(prim, *args, **kwargs):
  '''
  Decides which interpreter to use.
  Wraps up arguments into tracers and applies interpretation rules.
  '''
  interpreter = find_top_interpreter(args)
  tracers = [full_raise(interpreter, arg) for arg in args]
  out_tracer = interpreter.interp(prim, tracers, kwargs)
  return full_lower(out_tracer)


def find_top_interpreter(args) -> Interpreter:
  interpreters = [arg.interpreter for arg in args if isinstance(arg, Tracer)]
  top_interpreter = max(interpreters, default=INTERPRETER_STACK[0])
  if DYNAMIC_INTERPRETER and DYNAMIC_INTERPRETER > top_interpreter:
    top_interpreter = DYNAMIC_INTERPRETER
  return top_interpreter


def full_raise(interpreter: Interpreter, arg: Any) -> Tracer:
  if isinstance(arg, Tracer):
    interp = interpreter
    arg_interp = arg.interpreter
    if arg_interp is interp:
      return arg
    elif arg_interp < interp:
      return interpreter.lift(arg)
    elif arg_interp > interp:
      raise Exception(f'Can\'t lift level {arg_interp.level} to {interp.level}.')
    else:
      raise Exception(f'Different interpreters at same level: {arg_interp}, {interp}.')
  else:
    assert type(arg) in JAX_TYPES, f'Type {type(arg)} not supported.'
    return interpreter.pure(arg)


def full_lower(arg: Any):
  return arg.full_lower() if isinstance(arg, Tracer) else arg


# --- Abstract Arrays ---


class ShapedArray:
  '''
  Base abstract array type.
  Represents arrays with a specific shape and dtype.
  '''
  array_abstraction_level = 1

  def __init__(self, shape: Tuple[int], dtype: np.dtype):
    self.shape = shape
    self.dtype = dtype

  @property
  def ndim(self):
    return len(self.shape)

  _add = staticmethod(POP.add)
  _radd = staticmethod(lambda x, y: POP.add(y, x))
  _mul = staticmethod(POP.mul)
  _rmul = staticmethod(lambda x, y: POP.mul(y, x))
  _neg = staticmethod(POP.neg)
  _gt = staticmethod(POP.greater)
  _lt = staticmethod(POP.less)

  @staticmethod
  def _bool(tracer) -> bool:
    raise Exception("ShapedArray can't be unambiguously converted to bool")

  @staticmethod
  def _nonzero(tracer) -> bool:
    raise Exception("ShapedArray can't be unambiguously converted to bool")

  def __hash__(self):
    return hash((self.shape, self.dtype))

  def __eq__(self, other):
    return (type(self) is type(other) and self.shape == other.shape and self.dtype == other.dtype)

  def __repr__(self):
    return f'ShapedArray(shape={self.shape}, dtype={self.dtype})'

  def __str__(self):
    return f'{self.dtype.name}[{",".join(map(str, self.shape))}]'


class ConcreteArray(ShapedArray):
  '''
  A ShapedArray sub-class with known shape, dtype, and value.
  '''
  array_abstraction_level = 2

  def __init__(self, val):
    self.val = val
    self.shape = val.shape
    self.dtype = val.dtype

  @staticmethod
  def _bool(tracer):
    return bool(tracer.aval.val)

  @staticmethod
  def _nonzero(tracer):
    return bool(tracer.aval.val)

  def __repr__(self):
    return f'ConcreteArray(shape={self.shape}, dtype={self.dtype})'


def get_aval(x):
  if isinstance(x, Tracer):
    return x.aval
  elif type(x) in JAX_TYPES:
    return ConcreteArray(np.asarray(x))
  else:
    raise TypeError(f'Abstract value {x} has unsupported data type {type(x)}')
