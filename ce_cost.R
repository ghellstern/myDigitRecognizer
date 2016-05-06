ce_cost <- function(output, label) {
  # Title:
    # Compute the average cross-entropy cost for a given training set
  
  # Argument:
    # output - a matrix of output values of the neural network ([n_examples, n_labels])
    # label - a binary matrix of similar size to "output" to 
      # indicate the true class of the training example ([n_examples, n_classes])
  
  # Value:
    # Return a single value
  
  # Description:
    # The cross-entropy cost is given by the formula:
    # sum(y*log(a) + (1-y)*log(1-a)) over all output.
  
  # Example:
    # output = [0.7, 0.1, 0.4]
    #          [0.2, 0.5, 0.9]
    # label = [1, 0, 0]
    #         [0, 0, 1]
  
  m1 <- label * log(output)
  m2 <- (1-label) * log(1-output)
  cost <- -mean(apply(cbind(m1, m2), MARGIN=1, FUN=sum, na.rm=T))
  
  return(cost)
}