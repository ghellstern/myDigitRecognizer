miniBatch_gradient_descent <- function(input, label, parameters, batch_size, epoch, learning_rate) {
  
 
  # Title:
  #   Perform mini-batch/stochastic gradient descent.
  
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
  for(i in 1:epoch) {
    
    # Shuffle the example order before each epoch
    # Columns of mb_matrix keep track of each mini-batch
    mb_matrix <- matrix(sample(nrow(input)), nrow=batch_size)
    
    # Perform gradient update for each mini-batch
    for(j in 1:ncol(mb_matrix))
    {
      # Feedforward mini-batch to compute output
      ff <- feedforward(input = input[mb_matrix[,j],], parameter.list = parameters)
      
      # Backpropagation to get gradients of weight and bias
      grad <- backpropagation(ff=ff, parameter.list = parameters, 
                              label=label[mb_matrix[,j],])
      
      # Perform gradient descent
      parameters <- gradient_descent(parameter.list = parameters, 
                                     gradient.list = grad, learning_rate = learning_rate.mb)
      
      
      # Compute cost after each mini-batch for monitoring (can be turned off)
      # new_cost <- ce_cost(output = ff$a[[3]], label = label)
      # cost <- c(cost, new_cost)
      # print(paste("Epoch:", i,"Mini-batch", j, "completed."))
      # print(paste("Cost: ", new_cost))
    }
    
    # Compute cost after each epoch for monitoring (can be turned off)
    ff<- feedforward(input = input, parameter.list = parameters)
    new_cost <- ce_cost(output = ff$a[[3]], label = label)
    cost <- c(cost, new_cost)
    print(paste("Epoch", i, "completed."))
    print(paste("Cost: ", new_cost))
  }
  print(proc.time() - ptm)
  
  #================================
  # Cost Visualization
  #================================
  
  plot(x=0:epoch, y=c(initial_cost, cost), type="l", xlab="No of epoch", ylab="Cost")
  
  return(parameters)
  
}