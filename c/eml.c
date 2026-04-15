#include "eml.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double complex eml_clog(double complex z) {
    if (creal(z) == 0.0 && cimag(z) == 0.0) {
        return -INFINITY + 0.0 * I;
    }
    return clog(z);
}

static double complex eml_cexp(double complex z) {
    if (isinf(creal(z)) && cimag(z) == 0.0) {
        if (creal(z) < 0.0) {
            return 0.0 + 0.0 * I;
        }
        return INFINITY + 0.0 * I;
    }
    return cexp(z);
}

double complex eml(double complex x, double complex y) {
    return eml_cexp(x) - eml_clog(y);
}

double complex eml_one(void) {
    return 1.0 + 0.0 * I;
}

double complex eml_exp(double complex x) {
    return eml(x, eml_one());
}

double complex eml_log(double complex x) {
    return eml(eml_one(), eml_exp(eml(eml_one(), x)));
}

double complex eml_zero(void) {
    return eml_log(eml_one());
}

double complex eml_sub(double complex a, double complex b) {
    return eml(eml_log(a), eml_exp(b));
}

double complex eml_neg(double complex x) {
    return eml_sub(eml_zero(), x);
}

double complex eml_add(double complex a, double complex b) {
    return eml_sub(a, eml_neg(b));
}

double complex eml_inv(double complex x) {
    return eml_exp(eml_neg(eml_log(x)));
}

double complex eml_mul(double complex a, double complex b) {
    return eml_exp(eml_add(eml_log(a), eml_log(b)));
}

double complex eml_div(double complex a, double complex b) {
    return eml_mul(a, eml_inv(b));
}

double complex eml_pow(double complex a, double complex b) {
    return eml_exp(eml_mul(b, eml_log(a)));
}

double complex eml_int(int n) {
    if (n == 0) {
        return eml_zero();
    }
    if (n == 1) {
        return eml_one();
    }
    if (n < 0) {
        return eml_neg(eml_int(-n));
    }

    double complex acc = 0.0 + 0.0 * I;
    int has_acc = 0;
    double complex term = eml_one();
    int k = n;

    while (k > 0) {
        if (k & 1) {
            if (has_acc) {
                acc = eml_add(acc, term);
            } else {
                acc = term;
                has_acc = 1;
            }
        }
        term = eml_add(term, term);
        k >>= 1;
    }

    return acc;
}

double complex eml_rational(int p, int q) {
    if (q == 0) {
        fprintf(stderr, "eml_rational: denominator must be non-zero\n");
        exit(EXIT_FAILURE);
    }

    double complex val = eml_div(eml_int(abs(p)), eml_int(abs(q)));
    if ((p < 0) ^ (q < 0)) {
        return eml_neg(val);
    }
    return val;
}

double complex eml_e(void) {
    return eml_exp(eml_one());
}

double complex eml_minus_one(void) {
    return eml_neg(eml_one());
}

double complex eml_two(void) {
    return eml_int(2);
}

double complex eml_half(double complex x) {
    return eml_div(x, eml_two());
}

double complex eml_i(void) {
    return eml_pow(eml_minus_one(), eml_rational(1, 2));
}

double complex eml_pi(void) {
    return eml_neg(eml_mul(eml_i(), eml_log(eml_minus_one())));
}

double complex eml_identity(double complex x) {
    return eml_log(eml_exp(x));
}

double complex eml_sqrt(double complex x) {
    return eml_pow(x, eml_rational(1, 2));
}

double complex eml_log_base(double complex base, double complex x) {
    return eml_div(eml_log(x), eml_log(base));
}

double complex eml_sinh(double complex x) {
    return eml_half(eml_sub(eml_exp(x), eml_exp(eml_neg(x))));
}

double complex eml_cosh(double complex x) {
    return eml_half(eml_add(eml_exp(x), eml_exp(eml_neg(x))));
}

double complex eml_tanh(double complex x) {
    return eml_div(eml_sinh(x), eml_cosh(x));
}

double complex eml_sin(double complex x) {
    double complex i = eml_i();
    double complex ix = eml_mul(i, x);
    double complex num = eml_sub(eml_exp(ix), eml_exp(eml_neg(ix)));
    double complex den = eml_mul(eml_two(), i);
    return eml_div(num, den);
}

double complex eml_cos(double complex x) {
    double complex i = eml_i();
    double complex ix = eml_mul(i, x);
    double complex num = eml_add(eml_exp(ix), eml_exp(eml_neg(ix)));
    return eml_half(num);
}

double complex eml_tan(double complex x) {
    return eml_div(eml_sin(x), eml_cos(x));
}

double complex eml_asin(double complex x) {
    double complex i = eml_i();
    double complex inside = eml_add(
        eml_mul(eml_neg(i), x),
        eml_sqrt(eml_sub(eml_one(), eml_pow(x, eml_two())))
    );
    return eml_mul(i, eml_log(inside));
}

double complex eml_acos(double complex x) {
    double complex i = eml_i();
    double complex inside = eml_add(
        x,
        eml_mul(
            eml_sqrt(eml_sub(x, eml_one())),
            eml_sqrt(eml_add(x, eml_one()))
        )
    );
    return eml_mul(i, eml_log(inside));
}

double complex eml_atan(double complex x) {
    double complex i = eml_i();
    double complex half_i = eml_div(eml_neg(i), eml_two());
    double complex ratio = eml_div(eml_add(eml_neg(i), x), eml_sub(eml_neg(i), x));
    return eml_mul(half_i, eml_log(ratio));
}

double complex eml_asinh(double complex x) {
    return eml_log(eml_add(x, eml_sqrt(eml_add(eml_pow(x, eml_two()), eml_one()))));
}

double complex eml_acosh(double complex x) {
    double complex inside = eml_add(
        x,
        eml_mul(
            eml_sqrt(eml_add(x, eml_one())),
            eml_sqrt(eml_sub(x, eml_one()))
        )
    );
    return eml_log(inside);
}

double complex eml_atanh(double complex x) {
    double complex ratio = eml_div(eml_add(eml_one(), x), eml_sub(eml_one(), x));
    return eml_half(eml_log(ratio));
}
