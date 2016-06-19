sigmoid <- function(z) {
  # Title:
    # Compute sigmoid of elements in vector/matrix
  
  # Argument:
    # z - vector/matrix
  
  # Value:
    # Return a vector/matrix
  
  return(1/(1+exp(-z)))
}