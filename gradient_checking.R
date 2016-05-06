#===========================================================================================
# Gradient checking
#===========================================================================================

train.sample.1 <- train.sample[1:2,]
label.sample.1 <- label.sample[1:2,]

# Set nn size
size <- c(784,20,10)

# Initialize weights and biases for hidden and output layers
source("initParameters.R")
set.seed(20)
parameters.sample <- initParameters(size=size)

# Perform feedforward propagation on sample dataset to get z's & a's
source("feedforward.R")
ff.sample <- feedforward(input = train.sample.1, parameter.list = parameters.sample) 

# Compute initial cost of sample set using output from our feedforward propagation & label
source("ce_cost.R")
initial_cost <- ce_cost(output = ff.sample$a[[3]], label = label.sample.1) # 13.294
print(paste("Initial cost:", initial_cost))

# Define epsilon to gradient checking
epsilon <- 1e-4
grad.test <- matrix(NA, nrow=nrow(parameters.sample[[1]]), ncol=ncol(parameters.sample[[1]]))
for(i in 1:nrow(parameters.sample[[1]])) {
  for(j in 1:ncol(parameters.sample[[1]])) {
    
    parameters.sample.p <- parameters.sample
    parameters.sample.p[[1]][i,j] <- parameters.sample.p[[1]][i,j] + epsilon
    ff.sample.p <- feedforward(input = train.sample.1, parameter.list = parameters.sample.p) 
    cost.p <- ce_cost(output = ff.sample.p$a[[3]], label = label.sample.1)
    
    parameters.sample.n <- parameters.sample
    parameters.sample.n[[1]][i,j] <- parameters.sample.n[[1]][i,j] - epsilon
    ff.sample.n <- feedforward(input = train.sample.1, parameter.list = parameters.sample.n) 
    cost.n <- ce_cost(output = ff.sample.n$a[[3]], label = label.sample.1)
    
    grad.test[i,j] <- (cost.p-cost.n)/(2*epsilon)
  }
}

grad.bp <- backpropagation(ff=ff.sample, parameter.list = parameters.sample, label=label.sample.1)

mean(grad.test - grad.bp[[1]])
View(grad.bp[[1]][,100:200])
View(grad.test[,100:200])
