#include "utils.c"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct layer {
  int neurons;

  /* To add layers to a model, simply assign the next and last pointers.
     However, the weights need to be re-initialized. */

  struct layer *next;
  struct layer *last;
  double *weights;
  double *biases;
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

  double *activations = input;
  double *next_activations = NULL;
  layer_t *cur_layer = first_layer->next;
  int depth = 0;
  while (cur_layer) {
    next_activations = calloc(cur_layer->neurons, sizeof(double));
    for (int i = 0; i < cur_layer->neurons; i++) {
      for (int j = 0; j < cur_layer->last->neurons; j++) {
        next_activations[i] += activations[j]
          * cur_layer->weights[cur_layer->last->neurons * i + j];
      }
      next_activations[i] += cur_layer->biases[i];
      next_activations[i] = relu(next_activations[i]);
    }
    cur_layer = cur_layer->next;
    if (depth) {
      free(activations);
    }
    activations = next_activations;
    depth++;
  }
  return activations;
}
