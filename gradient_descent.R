gradient_descent <- function(parameter.list, gradient.list, learning_rate) {
  
  # Title:
  #   Perform gradient update on parameters
  
  # Arguments:
  #   parameter.list- a list of matrices containing parameters
  #   gradient.list-  a list of matrices containing gradients of parameters
  #   learning_rate- a number that specify the rate of gradient descent
  
  # Values:
  #   Return a list of matrices containing updated parameters
  
  # No of matrices of parameters:
  n <- length(parameter.list)
  
  updated.parameter.list <- list()
  
  for(i in 1:n) {
    updated.parameter.list[[i]] <- parameter.list[[i]] - learning_rate*gradient.list[[i]]
  }
 
  return(updated.parameter.list) 
}