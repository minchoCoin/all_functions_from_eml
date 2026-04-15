#ifndef EML_H
#define EML_H

#include <complex.h>

double complex eml(double complex x, double complex y);

double complex eml_one(void);
double complex eml_exp(double complex x);
double complex eml_log(double complex x);
double complex eml_zero(void);

double complex eml_sub(double complex a, double complex b);
double complex eml_neg(double complex x);
double complex eml_add(double complex a, double complex b);
double complex eml_inv(double complex x);
double complex eml_mul(double complex a, double complex b);
double complex eml_div(double complex a, double complex b);
double complex eml_pow(double complex a, double complex b);

double complex eml_int(int n);
double complex eml_rational(int p, int q);

double complex eml_e(void);
double complex eml_minus_one(void);
double complex eml_two(void);
double complex eml_half(double complex x);
double complex eml_i(void);
double complex eml_pi(void);

double complex eml_identity(double complex x);
double complex eml_sqrt(double complex x);
double complex eml_log_base(double complex base, double complex x);

double complex eml_sinh(double complex x);
double complex eml_cosh(double complex x);
double complex eml_tanh(double complex x);

double complex eml_sin(double complex x);
double complex eml_cos(double complex x);
double complex eml_tan(double complex x);

double complex eml_asin(double complex x);
double complex eml_acos(double complex x);
double complex eml_atan(double complex x);

double complex eml_asinh(double complex x);
double complex eml_acosh(double complex x);
double complex eml_atanh(double complex x);

#endif
