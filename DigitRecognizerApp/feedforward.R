feedforward <- function(input, parameter.list){
  
  # Title:
    # Perform feedforward propogation, and return a's and z's.
  
  # Arguments:
    # input - a matrix/dataframe, nrow=no of examples, ncol=no of features
    # parameter.list - a list of matrices, which contains weights and biases (see initParameters.R)
  
  # Values:
    # ff.a - list of matrices of activation values [nrow=no of examples, ncol=no of neurons in layer]
    # ff.z - list of matrices of z values [nrow=no of examples, ncol=no of neurons in layer]
    #Note: a[[1]] corresponds to input, while z[[1]] is NULL.
  
  if(!exists("sigmoid")) source("sigmoid.R")
  
  # No of layers, including input
  n_layers <- length(parameter.list)+1
  
  # Convert input to matrix form
  x <- as.matrix(input)
  
  # Initialize list for z's and a's
  z <- list()
  a <- list()
  
  # Add input to a[[1]]
  a[[1]] <- x
  z[[1]] <- NULL
  
  # Propagating through the layers...
  for(i in 2:n_layers) {
    # Add a column of '1's (bias) to input
    x <- cbind(matrix(1, nrow=nrow(x)), x)
    # Extract the matrix of weights/bias
    w <- parameter.list[[i-1]]
    # Compute z and a
    z[[i]] <- x %*% t(w)
    a[[i]] <- sigmoid(z[[i]])
    # Set x as the input for the next layer
    x <- a[[i]]
  }
  
  return(list(z=z, a=a))
}