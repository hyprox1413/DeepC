#include "layer.c"

#include <stdio.h>
#include <stdlib.h>

void load_mnist(double *train_image, int *train_label,
                double *test_image, int *test_label) {
  FILE *fp = fopen("mnist_train.csv", "r");
  if (!fp) {
    fprintf(stderr, "Open error.");
  }
  for (int i = 0; i < 60000; i++) {
    if (i % 1000 == 0) {
      printf("Loading training image %d\n", i);
    }
    fscanf(fp, "%d, ", train_label + i);
    for (int j = 0; j < 784; j++) {
      fscanf(fp, "%lf, ", &train_image[784 * i + j]);
      train_image[784 * i + j] /= 256.0;
    }
  }
  fclose(fp);
  fp = fopen("mnist_test.csv", "r");
  if (!fp) {
    fprintf(stderr, "Open error.");
  }
  for (int i = 0; i < 10000; i++) {
    if (i % 1000 == 0) {
      printf("Loading testing image %d\n", i);
    }
    fscanf(fp, "%d, ", test_label + i);
    for (int j = 0; j < 784; j++) {
      fscanf(fp, "%lf, ", &test_image[784 * i + j]);
      test_image[784 * i + j] /= 256.0;
    }
  }
}

int main() {
  double *train_image = malloc(sizeof(double) * 60000 * 784);
  int *train_label = malloc(sizeof(int) * 60000);
  double *test_image = malloc(sizeof(double) * 10000 * 784);
  int *test_label = malloc(sizeof(int) * 10000);

  load_mnist(train_image, train_label, test_image, test_label);

  FILE *fp = fopen("mnist.bin", "wb");
  fwrite(train_image, sizeof(double), 60000 * 784, fp);
  fwrite(train_label, sizeof(int), 60000, fp);
  fwrite(test_image, sizeof(double), 10000 * 784, fp);
  fwrite(test_label, sizeof(int), 10000, fp);
}
