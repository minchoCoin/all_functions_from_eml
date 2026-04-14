from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _clog(value):
    z = complex(value)
    if z.real == 0.0 and z.imag == 0.0:
        return complex(-np.inf, 0.0)
    return np.log(z)


@njit(cache=True)
def _cexp(value):
    z = complex(value)
    if np.isinf(z.real) and z.imag == 0.0:
        if z.real < 0.0:
            return 0.0 + 0.0j
        return complex(np.inf, 0.0)
    return np.exp(z)


@njit(cache=True)
def eml(x, y):
    return _cexp(x) - _clog(y)


@njit(cache=True)
def one_eml():
    return 1.0 + 0.0j


@njit(cache=True)
def exp_eml(x):
    return eml(x, one_eml())


@njit(cache=True)
def log_eml(x):
    return eml(one_eml(), exp_eml(eml(one_eml(), x)))


@njit(cache=True)
def zero_eml():
    return log_eml(one_eml())


@njit(cache=True)
def sub_eml(a, b):
    return eml(log_eml(a), exp_eml(b))


@njit(cache=True)
def neg_eml(x):
    return sub_eml(zero_eml(), x)


@njit(cache=True)
def add_eml(a, b):
    return sub_eml(a, neg_eml(b))


@njit(cache=True)
def inv_eml(x):
    return exp_eml(neg_eml(log_eml(x)))


@njit(cache=True)
def mul_eml(a, b):
    return exp_eml(add_eml(log_eml(a), log_eml(b)))


@njit(cache=True)
def div_eml(a, b):
    return mul_eml(a, inv_eml(b))


@njit(cache=True)
def pow_eml(a, b):
    return exp_eml(mul_eml(b, log_eml(a)))


@njit(cache=True)
def int_eml(n):
    if n == 0:
        return zero_eml()
    if n == 1:
        return one_eml()
    if n < 0:
        return neg_eml(int_eml(-n))

    acc = 0.0 + 0.0j
    has_acc = False
    term = one_eml()
    k = n
    while k > 0:
        if k & 1:
            if has_acc:
                acc = add_eml(acc, term)
            else:
                acc = term
                has_acc = True
        term = add_eml(term, term)
        k >>= 1
    return acc


@njit(cache=True)
def rational_eml(p, q):
    if q == 0:
        raise ZeroDivisionError("q must be non-zero")
    val = div_eml(int_eml(abs(p)), int_eml(abs(q)))
    if (p < 0) ^ (q < 0):
        return neg_eml(val)
    return val


@njit(cache=True)
def e_eml():
    return exp_eml(one_eml())


@njit(cache=True)
def minus_one_eml():
    return neg_eml(one_eml())


@njit(cache=True)
def two_eml():
    return int_eml(2)


@njit(cache=True)
def half_eml(x):
    return div_eml(x, two_eml())


@njit(cache=True)
def i_eml():
    return pow_eml(minus_one_eml(), rational_eml(1, 2))


@njit(cache=True)
def pi_eml():
    return neg_eml(mul_eml(i_eml(), log_eml(minus_one_eml())))


@njit(cache=True)
def identity_eml(x):
    return log_eml(exp_eml(x))


@njit(cache=True)
def sqrt_eml(x):
    return pow_eml(x, rational_eml(1, 2))


@njit(cache=True)
def log_base_eml(base, x):
    return div_eml(log_eml(x), log_eml(base))


@njit(cache=True)
def sinh_eml(x):
    return half_eml(sub_eml(exp_eml(x), exp_eml(neg_eml(x))))


@njit(cache=True)
def cosh_eml(x):
    return half_eml(add_eml(exp_eml(x), exp_eml(neg_eml(x))))


@njit(cache=True)
def tanh_eml(x):
    return div_eml(sinh_eml(x), cosh_eml(x))


@njit(cache=True)
def sin_eml(x):
    i = i_eml()
    ix = mul_eml(i, x)
    num = sub_eml(exp_eml(ix), exp_eml(neg_eml(ix)))
    den = mul_eml(two_eml(), i)
    return div_eml(num, den)


@njit(cache=True)
def cos_eml(x):
    i = i_eml()
    ix = mul_eml(i, x)
    num = add_eml(exp_eml(ix), exp_eml(neg_eml(ix)))
    return half_eml(num)


@njit(cache=True)
def tan_eml(x):
    return div_eml(sin_eml(x), cos_eml(x))


@njit(cache=True)
def asin_eml(x):
    i = i_eml()
    inside = add_eml(mul_eml(neg_eml(i), x), sqrt_eml(sub_eml(one_eml(), pow_eml(x, two_eml()))))
    return mul_eml(i, log_eml(inside))


@njit(cache=True)
def acos_eml(x):
    i = i_eml()
    inside = add_eml(x, mul_eml(sqrt_eml(sub_eml(x, one_eml())), sqrt_eml(add_eml(x, one_eml()))))
    return mul_eml(i, log_eml(inside))


@njit(cache=True)
def atan_eml(x):
    i = i_eml()
    half_i = div_eml(neg_eml(i), two_eml())
    ratio = div_eml(add_eml(neg_eml(i), x), sub_eml(neg_eml(i), x))
    return mul_eml(half_i, log_eml(ratio))


@njit(cache=True)
def asinh_eml(x):
    return log_eml(add_eml(x, sqrt_eml(add_eml(pow_eml(x, two_eml()), one_eml()))))


@njit(cache=True)
def acosh_eml(x):
    inside = add_eml(x, mul_eml(sqrt_eml(add_eml(x, one_eml())), sqrt_eml(sub_eml(x, one_eml()))))
    return log_eml(inside)


@njit(cache=True)
def atanh_eml(x):
    ratio = div_eml(add_eml(one_eml(), x), sub_eml(one_eml(), x))
    return half_eml(log_eml(ratio))


if __name__ == "__main__":
    print("pi =", pi_eml())
    print("sin(1.25) =", sin_eml(1.25))
