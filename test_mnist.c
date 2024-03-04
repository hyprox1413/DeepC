#include "layer.c"

#include <stdio.h>

#define TRAIN_EPOCHS (10)

int main() {
  double *train_image = malloc(sizeof(double) * 60000 * 784);
  int *train_label = malloc(sizeof(int) * 60000);
  double *test_image = malloc(sizeof(double) * 10000 * 784);
  int *test_label = malloc(sizeof(int) * 10000);

  FILE *fp = fopen("mnist.bin", "rb");
  fread(train_image, sizeof(double), 60000 * 784, fp);
  fread(train_label, sizeof(int), 60000, fp);
  fread(test_image, sizeof(double), 10000 * 784, fp);
  fread(test_label, sizeof(int), 10000, fp);

  /*
  for (int i = 0; i < 60000; i += 1000) {
    for (int y = 0; y < 28; y++) {
      for (int x = 0; x < 28; x++) {
        if (train_image[784 * i + 28 * y + x] > 0.5) {
          printf("%c", 219);
        }
        else {
          printf(" ");
        }
      }
      printf("\n");
    }
    printf("label: %d", train_label[i]);
  }

  for (int i = 0; i < 10000; i += 1000) {
    for (int y = 0; y < 28; y++) {
      for (int x = 0; x < 28; x++) {
        if (test_image[784 * i + 28 * y + x] > 0.5) {
          printf("%c", 219);
        }
        else {
          printf(" ");
        }
      }
      printf("\n");
    }
    printf("label: %d", test_label[i]);
  }
  */

  time_t t;
  srand((unsigned) time(&t));

  layer_t first = {784};
  layer_t second = {10};
  first.next = &second;
  second.last = &first;
  initialize_model(&first);

  for (int epoch = 0; epoch < TRAIN_EPOCHS; epoch++) {
    printf("Epoch %d\n", epoch);
    for (int i = 0; i < 60000; i++) {
      double output[10] = {};
      output[train_label[i]] = 1.0;
      train(&first, &train_image[784 * i], output, 0.0001);
    }
  }

  int correct = 0;
  for (int i = 0; i < 10000; i++) {
    predict(&first, &test_image[784 * i]);  
    int prediction = -1;
    double highest = 0.0;
    for (int j = 0; j < 10; j++) {
      if (second.activations[j] > highest) {
        prediction = j;
        highest = second.activations[j];
      }
    }
    if (prediction == test_label[i]) {
      correct++;
    }
  }
  printf("Total correct: %d\n", correct);
}
