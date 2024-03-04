#include "layer.c"

#define INPUT_DIMENSION (784)
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
  layer_t third = {};
  third.neurons = 10;
  second.next = &third;
  third.last = &second;
  initialize_model(&first);
  double input[INPUT_DIMENSION] = {};
  for (int i = 0; i < INPUT_DIMENSION; i++) {
    input[i] = 1;
  }
  predict(&first, input);
  for (int i = 0; i < 10; i++) {
    printf("%lf ", third.activations[i]);
  }
  double output[10] = {};
  for (int i = 0; i < 10; i++) {
    output[i] = i;
  }
  for (int i = 0; i < TRAIN_EPOCHS; i++) {
    train(&first, input, output, 0.000001);
    for (int j = 0; j < 10; j++) {
      printf("%lf ", third.activations[j]);
    }
    printf("\n");
  }
}
