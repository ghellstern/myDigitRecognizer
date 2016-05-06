#===========================================================================================
# Gradient checking 
#===========================================================================================

source("feedforward.R")
source("ce_cost.R")

# Sample set
input <- train.sample[1:100,-1]
label <- label.sample[1:100]
parameters <- parameters.sample


# Perform feedforward propagation on sample dataset to get z's & a's
ff <- feedforward(input = input, parameter.list = parameters) 

# Compute initial cost of sample set using output from our feedforward propagation & label
initial_cost <- ce_cost(output = ff$a[[3]], label = label) 
print(paste("Initial cost:", initial_cost))

# Define epsilon to gradient checking
epsilon <- 1e-4

# Select which parameter matrix to check
k = 2
grad.test <- matrix(NA, nrow=nrow(parameters[[k]]), ncol=ncol(parameters[[k]]))
for(i in 1:nrow(parameters[[k]])) {
  for(j in 1:ncol(parameters[[k]])) {
    
    parameters.p <- parameters
    parameters.p[[k]][i,j] <- parameters.p[[k]][i,j] + epsilon
    ff.p <- feedforward(input = input, parameter.list = parameters.p) 
    cost.p <- ce_cost(output = ff.p$a[[3]], label = label)
    
    parameters.n <- parameters
    parameters.n[[k]][i,j] <- parameters.n[[k]][i,j] - epsilon
    ff.n <- feedforward(input = input, parameter.list = parameters.n) 
    cost.n <- ce_cost(output = ff.n$a[[3]], label = label)
    
    grad.test[i,j] <- (cost.p-cost.n)/(2*epsilon)
  }
}

grad.bp <- backpropagation(ff=ff, parameter.list = parameters, label=label)

mean(grad.test - grad.bp[[k]])
View(grad.bp[[k]])
View(grad.test)
