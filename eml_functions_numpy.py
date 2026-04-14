from __future__ import annotations

import numpy as np


def _z(value):
    return np.asarray(value, dtype=np.complex128)


def eml(x, y):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.exp(_z(x)) - np.log(_z(y))


def one_eml():
    return np.complex128(1.0 + 0.0j)


def exp_eml(x):
    return eml(x, one_eml())


def log_eml(x):
    return eml(one_eml(), exp_eml(eml(one_eml(), x)))


def zero_eml():
    return log_eml(one_eml())


def sub_eml(a, b):
    return eml(log_eml(a), exp_eml(b))


def neg_eml(x):
    return sub_eml(zero_eml(), x)


def add_eml(a, b):
    return sub_eml(a, neg_eml(b))


def inv_eml(x):
    return exp_eml(neg_eml(log_eml(x)))


def mul_eml(a, b):
    return exp_eml(add_eml(log_eml(a), log_eml(b)))


def div_eml(a, b):
    return mul_eml(a, inv_eml(b))


def pow_eml(a, b):
    return exp_eml(mul_eml(b, log_eml(a)))


def int_eml(n: int):
    if n == 0:
        return zero_eml()
    if n == 1:
        return one_eml()
    if n < 0:
        return neg_eml(int_eml(-n))

    acc = None
    term = one_eml()
    k = n
    while k > 0:
        if k & 1:
            acc = term if acc is None else add_eml(acc, term)
        term = add_eml(term, term)
        k >>= 1
    return acc


def rational_eml(p: int, q: int):
    if q == 0:
        raise ZeroDivisionError("q must be non-zero")
    val = div_eml(int_eml(abs(p)), int_eml(abs(q)))
    if (p < 0) ^ (q < 0):
        return neg_eml(val)
    return val


def e_eml():
    return exp_eml(one_eml())


def minus_one_eml():
    return neg_eml(one_eml())


def two_eml():
    return int_eml(2)


def half_eml(x):
    return div_eml(x, two_eml())


def i_eml():
    return pow_eml(minus_one_eml(), rational_eml(1, 2))


def pi_eml():
    return neg_eml(mul_eml(i_eml(), log_eml(minus_one_eml())))


def identity_eml(x):
    return log_eml(exp_eml(x))


def sqrt_eml(x):
    return pow_eml(x, rational_eml(1, 2))


def log_base_eml(base, x):
    return div_eml(log_eml(x), log_eml(base))


def sinh_eml(x):
    return half_eml(sub_eml(exp_eml(x), exp_eml(neg_eml(x))))


def cosh_eml(x):
    return half_eml(add_eml(exp_eml(x), exp_eml(neg_eml(x))))


def tanh_eml(x):
    return div_eml(sinh_eml(x), cosh_eml(x))


def sin_eml(x):
    i = i_eml()
    ix = mul_eml(i, x)
    num = sub_eml(exp_eml(ix), exp_eml(neg_eml(ix)))
    den = mul_eml(two_eml(), i)
    return div_eml(num, den)


def cos_eml(x):
    i = i_eml()
    ix = mul_eml(i, x)
    num = add_eml(exp_eml(ix), exp_eml(neg_eml(ix)))
    return half_eml(num)


def tan_eml(x):
    return div_eml(sin_eml(x), cos_eml(x))


def asin_eml(x):
    i = i_eml()
    inside = add_eml(mul_eml(neg_eml(i), x), sqrt_eml(sub_eml(one_eml(), pow_eml(x, two_eml()))))
    return mul_eml(i, log_eml(inside))


def acos_eml(x):
    i = i_eml()
    inside = add_eml(x, mul_eml(sqrt_eml(sub_eml(x, one_eml())), sqrt_eml(add_eml(x, one_eml()))))
    return mul_eml(i, log_eml(inside))


def atan_eml(x):
    i = i_eml()
    half_i = div_eml(neg_eml(i), two_eml())
    ratio = div_eml(add_eml(neg_eml(i), x), sub_eml(neg_eml(i), x))
    return mul_eml(half_i, log_eml(ratio))


def asinh_eml(x):
    return log_eml(add_eml(x, sqrt_eml(add_eml(pow_eml(x, two_eml()), one_eml()))))


def acosh_eml(x):
    inside = add_eml(x, mul_eml(sqrt_eml(add_eml(x, one_eml())), sqrt_eml(sub_eml(x, one_eml()))))
    return log_eml(inside)


def atanh_eml(x):
    ratio = div_eml(add_eml(one_eml(), x), sub_eml(one_eml(), x))
    return half_eml(log_eml(ratio))


if __name__ == "__main__":
    xs = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    print("sin(xs) =", sin_eml(xs))
    print("cosh(xs) =", cosh_eml(xs))
    print("pi =", pi_eml())
