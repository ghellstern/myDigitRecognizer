backpropagation <- function(ff, parameter.list, label, lambda=0, m=nrow(ff$a[[1]])) {
  
  # Title:
    # Perform backpropagation, and return weight/bias gradients
  
  # Arguments:
    # ff- list returned by feedforward function, containing a's and z's for all examples
    # parameter.list- list of matrices containing weights/biases (see initParameter.R)
    # label - a binary matrix of similar size to "output" to 
    #          indicate the true class of all training example ([n_examples, n_classes])
    # lambda - parameter for L2 regularization
    # m - total no of training examples (not equal to the mini-batch size!)
  
  # Values:
    # Return a list of matrices containing gradients of weights/biases
    
  # Additional info:
  #   No of grad matrices returned = no of parameter matrices 
  #     i.e each weight/bias will have its associated gradient returned
  
 
  if(!exists("sigmoid_prime")) source("sigmoid_prime.R")
  
  # No of layers, including input
  n_layers <- length(parameter.list)+1
  
  # No of training examples/mini-batch size
  b <- nrow(ff$a[[1]])
  
  grad <- list()
  
  # Compute error for output layer
  error <- ff$a[[n_layers]] - label # b x no of classes
  
  # Compute grad for output layer, then error for previous layer, then grad again...
  for(i in (n_layers-1):1) {
    
    # Bias gradient for (i+1)th layer (sum over examples)
    grad.b <- apply(error, MARGIN=2, FUN=sum) # 1 x 10
    
    # Weight gradients for (i+1)th layer (sum over examples)
    grad.w <- t(error) %*% ff$a[[i]] # 10 x 100
    
    # Full grad for (i+1)th layer, average over examples
    grad[[i]] <-cbind(grad.b, grad.w)/b
    
    # Add regularization penalty to weights
    if(lambda!=0) {
      grad[[i]][,-1] <- grad[[i]][,-1] + (lambda/m)*parameter.list[[i]][,-1]
    }
  
    # No need to compute error for input layer
    if(i==1) break
    # Backpropagate error for (i)th layer (remove bias terms)
    error <- (error %*% parameter.list[[i]][, -1]) * sigmoid_prime(ff$z[[i]]) # b x 100
  }
  
  return(grad)
}