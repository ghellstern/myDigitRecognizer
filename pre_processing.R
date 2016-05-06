#===========================================================================================
# Full dataset
#===========================================================================================

# Read dataset
mnist <- read.csv("./input/train.csv")
table(mnist[,1])
prop.table(table(mnist[,1])) # Training examples are about equally split among classes

# Visualize examples
par(mfrow=c(10,10), mar=rep(0, times=4))
indexes <- sample(1:nrow(mnist))
for(digit in 0:9) {
  for( i in head(indexes[digit==mnist[indexes,1]], 10)) {
    ob <- matrix(as.integer(-mnist[i,-1]), nrow=28, ncol=28, byrow=TRUE)
    ob <- t(apply(ob, MARGIN = 2, FUN = rev))
    image(ob, col = grey(seq(0, 1, length = 255)), axes=FALSE)
  }
}

# Increase all pixel value by 1 to remove 0 & standardize to [0,1]
# "train" is the standardized version of "mnist" dataset
train <- mnist
train[,-1] <- (train[,-1]+1)/255

# Split 30,000 examples for training and 12,000 for validation
set.seed(41)
train.index <- sample(1:nrow(mnist), size=30000)
val <- train[-train.index,]
train <- train[train.index,]
prop.table(table(val[,1]))
prop.table(table(train[,1]))
# Note: 1st column of train/val contains the y value, i.e the digit


# Create label matrix for train dataset
label.train <- matrix(0, nrow=nrow(train), ncol=10)
for(i in 1:nrow(train)) {
  label.train[i, train[i,1]+1] <- 1
}

#========================================================
# Neural network size & parameters initialization (train)
#========================================================

# Set nn size
size<- c(784,100,10)

# Initialize weights and biases for hidden and output layers
source("initParameters.R")
set.seed(20)
parameters.train <- initParameters(size=size)


#==============================================================================================
# Sample set
#==============================================================================================

# Extract 3000 sample from train
set.seed(19)
sample.size = 3000
sample.index <- sample(1:nrow(train), size=sample.size)
train.sample <- train[sample.index, ]
prop.table(table(train.sample[,1]))

# Extract label matrix for sample set
label.sample <- label.train[sample.index, ]

#=========================================================
# Neural network size & parameters initialization (sample)
#=========================================================

# Set nn size
size <- c(784,20,10)

# Initialize weights and biases for hidden and output layers
source("initParameters.R")
set.seed(20)
parameters.sample <- initParameters(size=size)

