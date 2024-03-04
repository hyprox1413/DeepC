#include "layer.c"

#define INPUT_DIMENSION (1000)
#define TRAIN_EPOCHS (1000)

#include <stdio.h>

int main(void) {
  time_t t;
  srand((unsigned) time(&t));

  layer_t first = {};
  first.neurons = INPUT_DIMENSION;
  layer_t second = {};
  second.neurons = 10;
  first.next = &second;
  second.last = &first;
  initialize_model(&first);
  double input[INPUT_DIMENSION] = {};
  for (int i = 0; i < INPUT_DIMENSION; i++) {
    input[i] = 1.0;
  }
  for (int i = 0; i < 10; i++) {
    printf("%lf ", second.activations[i]);
  }
  double output[10] = {};
  for (int i = 0; i < 10; i++) {
    output[i] = i;
  }
  for (int i = 0; i < TRAIN_EPOCHS; i++) {
    train(&first, input, output, 0.0001);
    for (int j = 0; j < 10; j++) {
      printf("%lf ", second.activations[j]);
    }
    printf("\n");
  }
}
