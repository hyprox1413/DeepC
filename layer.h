#ifndef LAYER_H
#define LAYER_H

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

void initialize_model(layer_t *first_layer);

/*
 * Updates the activations of the network.  Be careful of the
 * input buffer size.  It is determined by the input dimension
 * of the model.
 */

double *predict(layer_t *first_layer, double *input);

/*
 * Calculates the partial derivative of the mean squared error loss
 * with respect to the ith neuron in the passed layer.
 *
 * Deprecated since dynamic programming is faster than recursion.
 */

double calc_partial(layer_t *cur_layer, int i, double *output);

/*
 * Updates the weights and biases of the model through deterministic
 * gradient descent with an adjustable learning rate.
 */

void train(layer_t *first_layer, double *input, double *output,
           double learn_rate);

#endif // LAYER_H
