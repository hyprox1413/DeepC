#include "utils.c"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define OUT_OF_MEMORY (-1)
#define NAN_ERROR (-2)

typedef struct layer {
  int neurons;

  /* To add layers to a model, simply assign the next and last pointers.
     However, the weights need to be re-initialized. */

  struct layer *next;
  struct layer *last;
  double *weights;
  double *biases;
  double *activations;
  double *gradients;
} layer_t;

/*
 * Initializes a model using Xavier initialization, uses pointer
 * to first layer as argument.
 */

void initialize_model(layer_t *first_layer) {

  /* first_layer layer must both exist and be first */

  assert(first_layer);
  assert(!first_layer->last);

  first_layer->activations = malloc(sizeof(double) * first_layer->neurons);
  if (!first_layer->activations) {
      fprintf(stderr, "Out of memory.\n");
      exit(OUT_OF_MEMORY);
  }
  layer_t *cur_layer = first_layer->next;
  while (cur_layer) {

    /* remember to free() these at some point lol */

    cur_layer->biases = malloc(sizeof(double) * cur_layer->neurons);
    if (!cur_layer->biases) {
      fprintf(stderr, "Out of memory.\n");
      exit(OUT_OF_MEMORY);
    }
    for (int i = 0; i < cur_layer->neurons; i++) {
      cur_layer->biases[i] = 0.0;
    }
    cur_layer->activations = malloc(sizeof(double) * cur_layer->neurons);
    if (!cur_layer->activations) {
      fprintf(stderr, "Out of memory.\n");
      exit(OUT_OF_MEMORY);
    }
    cur_layer->weights = malloc(sizeof(double)
        * cur_layer->neurons * cur_layer->last->neurons);
    if (!cur_layer->weights) {
      fprintf(stderr, "Out of memory.\n");
      exit(OUT_OF_MEMORY);
    }

    /* Xavier initialization */

    int last_neurons = cur_layer->last->neurons;
    for (int i = 0; i < cur_layer->neurons; i++) {
      for (int j = 0; j < last_neurons; j++) {
        cur_layer->weights[last_neurons * i + j]
          = rand_normal(0, 1.0 / last_neurons);
      }
    }
    cur_layer->gradients = malloc(sizeof(double) * cur_layer->neurons);
    if (!cur_layer->gradients) {
      fprintf(stderr, "Out of memory.\n");
      exit(OUT_OF_MEMORY);
    }
    cur_layer = cur_layer->next;
  }
}

/* 
 * Updates the activations of the network.  Be careful of the 
 * input buffer size.  It is determined by the input dimension
 * of the model.
 */

double *predict(layer_t *first_layer, double *input) {

  /* first_layer layer must both exist and be first */

  assert(first_layer);
  assert(!first_layer->last);
  assert(input);

  memcpy(first_layer->activations, input,
      sizeof(double) * first_layer->neurons);
  layer_t *cur_layer = first_layer->next;
  while (1) {
    for (int i = 0; i < cur_layer->neurons; i++) {
      cur_layer->activations[i] = 0.0; 
      for (int j = 0; j < cur_layer->last->neurons; j++) {
        cur_layer->activations[i] += cur_layer->last->activations[j]
          * cur_layer->weights[cur_layer->last->neurons * i + j];
      }
      cur_layer->activations[i] += cur_layer->biases[i];
      cur_layer->activations[i] = relu(cur_layer->activations[i]);
      if (isnan(cur_layer->activations[i])) {
        fprintf(stderr, "Floating point error.  "
            "Perhaps the learning rate is too high?\n");
        exit(NAN_ERROR);
      }
    }
    if (cur_layer->next) {
      cur_layer = cur_layer->next;
    }
    else {
      break;
    }
  }
  return cur_layer->activations;
}

/*
 * Calculates the partial derivative of the mean squared error loss
 * with respect to the ith neuron in the passed layer.
 *
 * Deprecated since dynamic programming is faster than recursion.
 */

double calc_partial(layer_t *cur_layer, int i, double *output) {
  if (!cur_layer->next) {
    return -2 * (output[i] - cur_layer->activations[i]);
  } 
  else {
    double total = 0;
    for (int j = 0; j < cur_layer->next->neurons; j++) {
      total += cur_layer->next->weights[cur_layer->neurons * j + i]
        * calc_partial(cur_layer->next, j, output);
    }
    return total;
  }
}

/*
 * Updates the weights and biases of the model through deterministic
 * gradient descent with an adjustable learning rate.
 */

void train(layer_t *first_layer, double *input, double *output,
           double learn_rate) {

  /* first_layer layer must both exist and be first */

  assert(first_layer);
  assert(!first_layer->last);
  assert(input);
  assert(output);

  predict(first_layer, input);
  layer_t *cur_layer = first_layer->next;
  while (cur_layer->next) {
    cur_layer = cur_layer->next;
  }
  while (cur_layer->last) {
    for (int i = 0; i < cur_layer->neurons; i++) {
      cur_layer->gradients[i] = 0.0;
    }
    if (cur_layer->next) {
      for (int i = 0; i < cur_layer->neurons; i++) {
        for (int j = 0; j < cur_layer->next->neurons; j++) {
          cur_layer->gradients[i]
            += cur_layer->next->weights[cur_layer->neurons * j + i]
            * cur_layer->next->gradients[j];
        }
      }
    }
    else {
      for (int i = 0; i < cur_layer->neurons; i++) {
        cur_layer->gradients[i] = -2 * (output[i] - cur_layer->activations[i]);
      }
    }
    for (int i = 0; i < cur_layer->neurons; i++) {
      cur_layer->biases[i] -= learn_rate * cur_layer->gradients[i];
      for (int j = 0; j < cur_layer->last->neurons; j++) {
        cur_layer->weights[cur_layer->last->neurons * i + j]
          -= learn_rate * cur_layer->last->activations[j]
          * cur_layer->gradients[i]; 
      }
    }
    cur_layer = cur_layer->last;
  }
}

