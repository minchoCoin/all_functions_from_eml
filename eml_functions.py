from __future__ import annotations

import cmath
import math
from typing import Union


Scalar = Union[int, float, complex]


def _z(value: Scalar) -> complex:
    return complex(value)


def _clog(value: Scalar) -> complex:
    z = _z(value)
    if z == 0:
        # Extended-complex convention used by the EML construction.
        return complex(float("-inf"), 0.0)
    return cmath.log(z)


def _cexp(value: Scalar) -> complex:
    z = _z(value)
    if math.isinf(z.real) and z.imag == 0.0:
        if z.real < 0:
            return 0j
        return complex(float("inf"), 0.0)
    return cmath.exp(z)


def eml(x: Scalar, y: Scalar) -> complex:
    """EML primitive: eml(x, y) = exp(x) - log(y)."""
    return _cexp(x) - _clog(y)


def one_eml() -> complex:
    return 1.0 + 0.0j


def exp_eml(x: Scalar) -> complex:
    return eml(x, one_eml())


def log_eml(x: Scalar) -> complex:
    return eml(one_eml(), exp_eml(eml(one_eml(), x)))


def zero_eml() -> complex:
    return log_eml(one_eml())


def sub_eml(a: Scalar, b: Scalar) -> complex:
    return eml(log_eml(a), exp_eml(b))


def neg_eml(x: Scalar) -> complex:
    return sub_eml(zero_eml(), x)


def add_eml(a: Scalar, b: Scalar) -> complex:
    return sub_eml(a, neg_eml(b))


def inv_eml(x: Scalar) -> complex:
    return exp_eml(neg_eml(log_eml(x)))


def div_eml(a: Scalar, b: Scalar) -> complex:
    return mul_eml(a, inv_eml(b))


def mul_eml(a: Scalar, b: Scalar) -> complex:
    return exp_eml(add_eml(log_eml(a), log_eml(b)))


def pow_eml(a: Scalar, b: Scalar) -> complex:
    return exp_eml(mul_eml(b, log_eml(a)))


def int_eml(n: int) -> complex:
    if n == 0:
        return zero_eml()
    if n == 1:
        return one_eml()
    if n < 0:
        return neg_eml(int_eml(-n))

    acc: complex | None = None
    term = one_eml()
    k = n
    while k > 0:
        if k & 1:
            acc = term if acc is None else add_eml(acc, term)
        term = add_eml(term, term)
        k >>= 1
    return acc if acc is not None else zero_eml()


def rational_eml(p: int, q: int) -> complex:
    if q == 0:
        raise ZeroDivisionError("q must be non-zero")
    val = div_eml(int_eml(abs(p)), int_eml(abs(q)))
    if (p < 0) ^ (q < 0):
        return neg_eml(val)
    return val


def e_eml() -> complex:
    return exp_eml(one_eml())


def minus_one_eml() -> complex:
    return neg_eml(one_eml())


def two_eml() -> complex:
    return int_eml(2)


def half_eml(x: Scalar) -> complex:
    return div_eml(x, two_eml())


def i_eml() -> complex:
    # Use the principal square root so the conventional +i branch is returned.
    return pow_eml(minus_one_eml(), rational_eml(1, 2))


def pi_eml() -> complex:
    return neg_eml(mul_eml(i_eml(), log_eml(minus_one_eml())))


def identity_eml(x: Scalar) -> complex:
    return log_eml(exp_eml(x))


def sqrt_eml(x: Scalar) -> complex:
    return pow_eml(x, rational_eml(1, 2))


def log_base_eml(base: Scalar, x: Scalar) -> complex:
    return div_eml(log_eml(x), log_eml(base))


def sinh_eml(x: Scalar) -> complex:
    return half_eml(sub_eml(exp_eml(x), exp_eml(neg_eml(x))))


def cosh_eml(x: Scalar) -> complex:
    return half_eml(add_eml(exp_eml(x), exp_eml(neg_eml(x))))


def tanh_eml(x: Scalar) -> complex:
    return div_eml(sinh_eml(x), cosh_eml(x))


def sin_eml(x: Scalar) -> complex:
    i = i_eml()
    ix = mul_eml(i, x)
    num = sub_eml(exp_eml(ix), exp_eml(neg_eml(ix)))
    den = mul_eml(two_eml(), i)
    return div_eml(num, den)


def cos_eml(x: Scalar) -> complex:
    i = i_eml()
    ix = mul_eml(i, x)
    num = add_eml(exp_eml(ix), exp_eml(neg_eml(ix)))
    return half_eml(num)


def tan_eml(x: Scalar) -> complex:
    return div_eml(sin_eml(x), cos_eml(x))


def asin_eml(x: Scalar) -> complex:
    i = i_eml()
    inside = add_eml(mul_eml(neg_eml(i), x), sqrt_eml(sub_eml(one_eml(), pow_eml(x, two_eml()))))
    return mul_eml(i, log_eml(inside))


def acos_eml(x: Scalar) -> complex:
    i = i_eml()
    inside = add_eml(x, mul_eml(sqrt_eml(sub_eml(x, one_eml())), sqrt_eml(add_eml(x, one_eml()))))
    return mul_eml(i, log_eml(inside))


def atan_eml(x: Scalar) -> complex:
    i = i_eml()
    half_i = div_eml(neg_eml(i), two_eml())
    ratio = div_eml(add_eml(neg_eml(i), x), sub_eml(neg_eml(i), x))
    return mul_eml(half_i, log_eml(ratio))


def asinh_eml(x: Scalar) -> complex:
    return log_eml(add_eml(x, sqrt_eml(add_eml(pow_eml(x, two_eml()), one_eml()))))


def acosh_eml(x: Scalar) -> complex:
    inside = add_eml(x, mul_eml(sqrt_eml(add_eml(x, one_eml())), sqrt_eml(sub_eml(x, one_eml()))))
    return log_eml(inside)


def atanh_eml(x: Scalar) -> complex:
    ratio = div_eml(add_eml(one_eml(), x), sub_eml(one_eml(), x))
    return half_eml(log_eml(ratio))


def _demo() -> None:
    x = 1.25
    y = 2.5

    print("EML primitive")
    print("eml(1, 1) =", eml(1, 1))
    print()

    print("Constants")
    print("e  =", e_eml())
    print("i  =", i_eml())
    print("pi =", pi_eml())
    print()

    print("Arithmetic")
    print("x + y =", add_eml(x, y))
    print("x - y =", sub_eml(x, y))
    print("x * y =", mul_eml(x, y))
    print("x / y =", div_eml(x, y))
    print("x ^ y =", pow_eml(x, y))
    print("log_y(x) =", log_base_eml(y, x))
    print()

    print("Trigonometric / hyperbolic")
    print("sin(x)  =", sin_eml(x))
    print("cos(x)  =", cos_eml(x))
    print("tan(x)  =", tan_eml(x))
    print("sinh(x) =", sinh_eml(x))
    print("cosh(x) =", cosh_eml(x))
    print("tanh(x) =", tanh_eml(x))


if __name__ == "__main__":
    _demo()
