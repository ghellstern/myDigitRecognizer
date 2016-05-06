sigmoid_prime <- function(z) {
  # Title:
   # Compute sigmoid prime of elements in vector/matrix
  
  # Argument:
    # z - vector/matrix
  
  # Value:
    # Return a vector/matrix
  
  # Additional info:
    # sigmoid_prime(z) = sigmoid(z) * (1-sigmoid(z))
  
  if(!exists("sigmoid")) source("sigmoid.R")
  
  return(sigmoid(z) * (1-sigmoid(z)))
}