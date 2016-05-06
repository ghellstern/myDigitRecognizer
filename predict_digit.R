predict_digit <- function(output) {
 
  # Title:
  #   Predict digit based on highest output value
  
  # Arugment:
  #   output- matrix containing the output value; 
  #           dim=[no of examples, no of classes]
  
  # Value:
  #   Return a vector of digits; length=no of examples
  
  return(apply(output, MARGIN = 1, FUN = which.max)-1)
  
    
}