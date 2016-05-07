# Building neural network from scratch using R
Writing my own neural network (nn) without using any external library.
The neural network will then be used on MNIST data to predict handwritten digits (from kaggle).

# Plan
1. Build a working feedforward nn. (completed)
2. Add L2 regularization to nn. (completed)
3. Add convolutional net.

# Results
1. Mini-batch gradient descent on 42,000 handwritten digits (each 28-by-28 pixel grayscale image). Neural network contains one hidden layer with 100 neurons. Training time under 6 minutes. Test accuracy on Kaggle is 91.3%.
