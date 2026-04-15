# all_functions_from_eml

Implementation of the paper "All elementary functions from a single binary operator" (arXiv:2603.21852).

This repository now contains Python and C implementations:

- [eml_functions.py](eml_functions.py): scalar `cmath` version
- [eml_functions_numpy.py](eml_functions_numpy.py): NumPy broadcasting version
- [eml_function_numba.py](eml_function_numba.py): Numba-friendly scalar version
- [eml.h](c/eml.h), [eml.c](c/eml.c): C implementations using `math.h` and `complex.h`

The only primitive is:

```text
EML[x,y] = exp(x) - log(y)
```

Below, every function is written as an EML tree string.

## Canonical EML building blocks

```text
ONE      = 1
EXP(x)   = EML[x,1]
LOG(x)   = EML[1,EML[EML[1,x],1]]
ZERO     = LOG(1)
SUB(x,y) = EML[LOG(x),EXP(y)]
NEG(x)   = SUB(ZERO,x)
ADD(x,y) = SUB(x,NEG(y))
INV(x)   = EXP(NEG(LOG(x)))
MUL(x,y) = EXP(ADD(LOG(x),LOG(y)))
DIV(x,y) = MUL(x,INV(y))
POW(x,y) = EXP(MUL(y,LOG(x)))
```

## Constants As EML Trees

```text
e    = EXP(1)
-1   = NEG(1)
2    = ADD(1,1)
1/2  = DIV(1,2)
i    = POW(NEG(1),DIV(1,2))
pi   = NEG(MUL(i,LOG(NEG(1))))
sqrt(x) = POW(x,DIV(1,2))
log_b(x) = DIV(LOG(x),LOG(b))
```

## Arithmetic EML Trees

```text
x + y = ADD(x,y)
x - y = SUB(x,y)
x * y = MUL(x,y)
x / y = DIV(x,y)
x ^ y = POW(x,y)
```

## Trigonometric EML Trees

```text
sin(x) = DIV(SUB(EXP(MUL(i,x)),EXP(NEG(MUL(i,x)))),MUL(2,i))
cos(x) = DIV(ADD(EXP(MUL(i,x)),EXP(NEG(MUL(i,x)))),2)
tan(x) = DIV(sin(x),cos(x))

asin(x) = MUL(i,LOG(ADD(MUL(NEG(i),x),sqrt(SUB(1,POW(x,2))))))
acos(x) = MUL(i,LOG(ADD(x,MUL(sqrt(SUB(x,1)),sqrt(ADD(x,1))))))
atan(x) = MUL(DIV(NEG(i),2),LOG(DIV(ADD(NEG(i),x),SUB(NEG(i),x))))
```

## Hyperbolic EML Trees

```text
sinh(x)  = DIV(SUB(EXP(x),EXP(NEG(x))),2)
cosh(x)  = DIV(ADD(EXP(x),EXP(NEG(x))),2)
tanh(x)  = DIV(sinh(x),cosh(x))

asinh(x) = LOG(ADD(x,sqrt(ADD(POW(x,2),1))))
acosh(x) = LOG(ADD(x,MUL(sqrt(ADD(x,1)),sqrt(SUB(x,1)))))
atanh(x) = DIV(LOG(DIV(ADD(1,x),SUB(1,x))),2)
```

## Figure-To-Formula Mapping

Figure 1 shows five exact EML trees. The strings below are written to match those tree examples one-for-one.

### 1. `ln x`

```text
EML[1,EML[EML[1,x],1]]
```

### 2. `x`

```text
EML[1,EML[EML[1,EML[x,1]],1]]
```

This is `LOG(EXP(x))`.

### 3. `-x`

```text
EML[EML[1,EML[EML[1,EML[1,EML[EML[1,1],1]]],1]],EML[x,1]]
```

This is `SUB(ZERO,x)`.

### 4. `x^{-1}`

```text
EML[EML[EML[1,EML[EML[1,EML[1,EML[EML[1,1],1]]],1]],EML[EML[1,EML[EML[1,x],1]],1]],1]
```

This is `EXP(NEG(LOG(x)))`.

### 5. `xy`

```text
EML[
  EML[
    EML[1,EML[EML[1,x],1]],
    1
  ],
  EML[
    EML[
      1,
      EML[
        EML[
          1,
          EML[
            EML[1,EML[EML[1,EML[1,EML[EML[1,1],1]]],1]],
            EML[EML[1,EML[EML[1,y],1]],1]
          ]
        ],
        1
      ]
    ],
    1
  ]
]
```

This is the direct expansion of:

```text
MUL(x,y) = EXP(ADD(LOG(x),LOG(y)))
```

## Module Notes

- `eml_functions.py` uses `cmath` and works well for scalar experimentation.
- `eml_functions_numpy.py` supports scalar inputs and NumPy arrays through broadcasting.
- `eml_function_numba.py` keeps the same formulas in a Numba-friendly scalar form and falls back to plain Python if `numba` is not installed.

## Quick Python Example

```python
from eml_functions import add_eml, sin_eml, pi_eml

print(add_eml(1.25, 2.5))
print(sin_eml(1.25))
print(pi_eml())
```

## Quick C Example
```c
//main.c example
#include "eml.h"

#include <complex.h>
#include <stdio.h>

static void print_complex(const char *label, double complex z) {
    printf("%-10s = %.15f%+.15fi\n", label, creal(z), cimag(z));
}

static void run_basic_tests(void) {
    double complex x = 1.25 + 0.0 * I;
    double complex y = 2.50 + 0.0 * I;

    puts("Arithmetic");
    print_complex("x+y", eml_add(x, y));
    print_complex("x-y", eml_sub(x, y));
    print_complex("x*y", eml_mul(x, y));

    puts("Constants");
    print_complex("e", eml_e());
    print_complex("i", eml_i());
    print_complex("pi", eml_pi());
    puts("");
}
int main(void) {
    run_basic_tests();
    return 0;
}
```

```bash
gcc -std=c11 -O2 -o eml_demo main.c eml.c -lm
```

# References
Andrzej Odrzywołek. All elementary functions from a single binary operator. arXiv preprint, arXiv:2603.21852[cs.SC], 2026

[https://github.com/VA00/SymbolicRegressionPackage](https://github.com/VA00/SymbolicRegressionPackage)