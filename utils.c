#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/* 
 * Implemetation of the Box-Muller transform to generate normally
 * distributed random numbers.
 */

double rand_normal(double mean, double variance) {
  int rand_int = rand();
  if (!rand_int) {
    rand_int = 1.0;
  }
  double uniform_1 = (double) rand_int / RAND_MAX;
  rand_int = rand();
  double uniform_2 = (double) rand_int / RAND_MAX;
  double normal_raw = sqrt(-2 * log(uniform_1)) *
      cos(2 * M_PI * uniform_2);
  return normal_raw * sqrt(variance) + mean;
}

/* ReLU activation. */

double relu(double x) {
  if (x < 0) {
    return 0;
  }
  return x;
}
