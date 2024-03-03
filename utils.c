#include <math.h>
#include <stdlib.h>
#include <time.h>

/* 
 * Implemetation of the Box-Muller transform to generate normally
 * distributed random numbers.
 */

double rand_normal(double mean, double variance) {
  double uniform_1 = (double) rand() / RAND_MAX;
  double uniform_2 = (double) rand() / RAND_MAX;
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
