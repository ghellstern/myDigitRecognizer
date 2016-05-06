ce_cost <- function(output, label, lambda=0, parameters=NULL) {
  # Title:
    # Compute the average cross-entropy cost for a given training set
  
  # Argument:
    # output - a matrix of output values of the neural network ([n_examples, n_labels])
    # label - a binary matrix of similar size to "output" to 
      # indicate the true class of the training example ([n_examples, n_classes])
    # lambda - parameter for L2 regularization
    # parameters - a list of matrices, which contains weights and biases (needed if lambda!=0)
  
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
  
  if(lambda !=0 & is.null(parameters)) {stop("Parameters needed to compute regularized cost")}
  
  m1 <- label * log(output)
  m2 <- (1-label) * log(1-output)
  cost <- -mean(apply(cbind(m1, m2), MARGIN=1, FUN=sum, na.rm=T))
  
  # Compute penalty cost if lambda is provided
  penalty <- 0
  if(lambda != 0) {
    m <- nrow(output)
    for(i in 1:length(parameters)) {
      # Exclude bias
      penalty = penalty + sum(parameters[[i]][,-1]^2)
    }
    
    # Add penalty to cost
    cost = cost + (lambda/(2*m)) * penalty
  }
  
  return(cost)
}