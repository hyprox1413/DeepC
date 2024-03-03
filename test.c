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
  for (int i = 0; i < 100; i++) {
    input[i] = 1.0;
  }
  for (int i = 0; i < 10; i++) {
    printf("%lf ", second.activations[i]);
  }
  double output[10] = {};
  while (1) {
    train(&first, input, output, 0.001);
    for (int i = 0; i < 10; i++) {
      printf("%lf ", second.activations[i]);
    }
    printf("\n");
  }
}
