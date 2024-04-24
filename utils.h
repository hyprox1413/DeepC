#ifndef UTILS_H
#define UTILS_H

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif // M_PI

/*
 * Implemetation of the Box-Muller transform to generate normally
 * distributed random numbers.
 */

double rand_normal(double mean, double variance);

/* ReLU activation. */

double relu(double x);

#endif // UTILS_H
