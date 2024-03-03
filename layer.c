#include "utils.c"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct layer {
  int neurons;

  /* To add layers to a model, simply assign the next and last pointers.
     However, the weights need to be re-initialized. */

  struct layer *next;
  struct layer *last;
  double *weights;
  double *biases;
  double *activations;
} layer_t;

/*
 * Initializes a model using Xavier initialization, uses pointer
 * to first layer as argument.
 */

void initialize_model(layer_t *first_layer) {

  /* first_layer layer must both exist and be first */

  assert(first_layer);
  assert(!first_layer->last);

  layer_t *cur_layer = first_layer->next;
  while (cur_layer) {

    /* remember to free() these at some point lol */

    cur_layer->weights = malloc(sizeof(double)
        * cur_layer->neurons * cur_layer->last->neurons);
    if (!cur_layer->weights) {
      fprintf(stderr, "Out of memory.");
    }
    cur_layer->biases = calloc(cur_layer->neurons, sizeof(double));
    if (!cur_layer->weights) {
      fprintf(stderr, "Out of memory.");
    }
    cur_layer->activations = calloc(cur_layer->neurons, sizeof(double));
    if (!cur_layer->activations) {
      fprintf(stderr, "Out of memory.");
    }
    int last_neurons = cur_layer->last->neurons;
    for (int i = 0; i < cur_layer->neurons; i++) {
      for (int j = 0; j < last_neurons; j++) {
        cur_layer->weights[last_neurons * i + j]
          = rand_normal(0, 1.0 / last_neurons);
      }
    }
    cur_layer = cur_layer->next;
  }
}

/* 
 * This one does what you think it does.  Be careful of the buffer
 * size of the input argument.  Also, remember to free() the output. 
 */

double *predict(layer_t *first_layer, double *input) {

  /* first_layer layer must both exist and be first */

  assert(first_layer);
  assert(!first_layer->last);
  assert(input);

  free(first_layer->activations);
  first_layer->activations = malloc(sizeof(double) * first_layer->neurons);
  memcpy(first_layer->activations, input,
      sizeof(double) * first_layer->neurons);
  layer_t *cur_layer = first_layer->next;
  while (1) {
    free(cur_layer->activations);
    cur_layer->activations = calloc(cur_layer->neurons, sizeof(double));
    for (int i = 0; i < cur_layer->neurons; i++) {
      for (int j = 0; j < cur_layer->last->neurons; j++) {
        cur_layer->activations[i] += cur_layer->last->activations[j]
          * cur_layer->weights[cur_layer->last->neurons * i + j];
      }
      cur_layer->activations[i] += cur_layer->biases[i];
      cur_layer->activations[i] = relu(cur_layer->activations[i]);
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

double calc_partial(layer_t *cur_layer, int i, double *output) {
  if (!cur_layer->next) {
    return -2 * (output[i] - cur_layer->activations[i]);
  } 
  else {
    double total = 0;
    fprintf(stderr, "%d", i);
    for (int j = 0; j < cur_layer->next->neurons; j++) {
      total += cur_layer->next->weights[cur_layer->neurons * j + i]
        * calc_partial(cur_layer->next, j, output);
    }
    return total;
  }
}

void train(layer_t *first_layer, double *input, double *output,
           double learn_rate) {

  /* first_layer layer must both exist and be first */

  assert(first_layer);
  assert(!first_layer->last);
  assert(input);
  assert(output);

  predict(first_layer, input);
  layer_t *cur_layer = first_layer->next;
  while (cur_layer) {
    for (int i = 0; i < cur_layer->neurons; i++) {
      double partial = calc_partial(cur_layer, i, output);
      cur_layer->biases[i] -= learn_rate * partial;
      for (int j = 0; j < cur_layer->last->neurons; j++) {
        cur_layer->weights[cur_layer->last->neurons * i + j]
          -= learn_rate * cur_layer->last->activations[j] * partial; 
      }
    }
    cur_layer = cur_layer->next;
  }
}

