#include "layer.c"

#include <stdio.h>

int main(void) {
  time_t t;
  srand((unsigned) time(&t));

  layer_t first = {};
  first.neurons = 100;
  layer_t second = {};
  second.neurons = 10;
  first.next = &second;
  second.last = &first;
  initialize_model(&first);
  double input[100] = {};
  double *output = predict(&first, input);
  for (int i = 0; i < 10; i ++) {
    printf("%lf ", output[i]);
  }
  printf("\n");
}
