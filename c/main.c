#include "eml.h"

#include <complex.h>
#include <stdio.h>

static void print_complex(const char *label, double complex z) {
    printf("%-10s = %.15f%+.15fi\n", label, creal(z), cimag(z));
}

static void run_basic_tests(void) {
    double complex x = 1.25 + 0.0 * I;
    double complex y = 2.50 + 0.0 * I;

    puts("EML primitive");
    print_complex("eml(1,1)", eml(1.0 + 0.0 * I, 1.0 + 0.0 * I));
    puts("");

    puts("Constants");
    print_complex("e", eml_e());
    print_complex("i", eml_i());
    print_complex("pi", eml_pi());
    puts("");

    puts("Arithmetic");
    print_complex("x+y", eml_add(x, y));
    print_complex("x-y", eml_sub(x, y));
    print_complex("x*y", eml_mul(x, y));
    print_complex("x/y", eml_div(x, y));
    print_complex("x^y", eml_pow(x, y));
    print_complex("log_y(x)", eml_log_base(y, x));
    puts("");

    puts("Trigonometric / hyperbolic");
    print_complex("sin(x)", eml_sin(x));
    print_complex("cos(x)", eml_cos(x));
    print_complex("tan(x)", eml_tan(x));
    print_complex("sinh(x)", eml_sinh(x));
    print_complex("cosh(x)", eml_cosh(x));
    print_complex("tanh(x)", eml_tanh(x));
}

int main(void) {
    run_basic_tests();
    return 0;
}
