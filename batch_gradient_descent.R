batch_gradient_descent <- function(input, label, parameters, batch, learning_rate) {
  
  # Title:
  #   Perform batch gradient descent.
  
  # Arguments:
  #   Self-explanatory
  
  # Values:
  #   Return trained parameters in original format
  
  source("backpropagation.R")
  source("gradient_descent.R")
  source("feedforward.R")
  source("ce_cost.R")
  
  
  # Perform feedforward propagation to get initial output
  ff <- feedforward(input = input, parameter.list = parameters) 
  
  # Compute initial cost
  initial_cost <- ce_cost(output = ff$a[[3]], label = label)
  print(paste("Initial cost:", initial_cost))
  
  # For monitoring of cost after each epoch
  cost <- NULL 
  
  ptm <- proc.time()
  for(i in 1:batch) {
    # Backpropagation to get gradients of weight and bias
    grad <- backpropagation(ff=ff, parameter.list = parameters, label=label)
    
    # Perform gradient descent
    parameters <- gradient_descent(parameter.list = parameters, 
                                   gradient.list = grad, learning_rate = learning_rate)
    
    
    # Feedforward to get new outputs
    ff <- feedforward(input = input, parameter.list = parameters) 
    
    # Compute cost after each batch run (optional)
    new_cost <- ce_cost(output = ff$a[[3]], label = label)
    cost <- c(cost, new_cost)
    print(paste("Batch", i, "completed."))
    print(paste("Cost: ", new_cost))
  }
  print(proc.time() - ptm)
  
  #================================
  # Cost Visualization
  #================================
  
  plot(x=0:batch, y=c(initial_cost, cost), type="l", xlab="No of batches", ylab="Cost")
  
  return(parameters)
}