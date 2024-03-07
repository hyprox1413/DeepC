# DeepC
Lightweight, low-level and fast implementation of feed-forward neural networks in C.  Uses ReLU activation and naive gradient descent with mean squared error loss.

Now with a test for MNIST digits!  Confirmed: It works!

The next step is to run computation on a GPU with CUDA, but I need to get an NVIDIA card.

Credit to https://pjreddie.com/projects/mnist-in-csv/ for the initial MNIST CSV files.  The testing program itself uses a binary dump of the MNIST dataset for fast execution.
Credit to quackduck for challenging me to do this.
