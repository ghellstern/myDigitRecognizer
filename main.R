# Mini-batch/batch gradient descent

#========
# Inputs
#========

# Training set 
input <- train[,-1]
label <- label.train
parameters <- parameters.train
true.y <- train[,1]

# Sample set
input <- train.sample[,-1]
label <- label.sample
parameters <- parameters.sample
true.y <- train.sample[,1]


#=================================================================================================
# Mini-batch gradient descent
#=================================================================================================

source("miniBatch_gradient_descent.R")

# Parameters
update_per_epoch = 200
b = nrow(input)/update_per_epoch
epoch = 30
learning_rate.mb = 1
lambda.mb = 0

set.seed(21)
trained_parameters <- miniBatch_gradient_descent(input=input, label=label, 
                                                 parameters=parameters, batch_size=b, 
                                                 epoch=epoch, learning_rate=learning_rate.mb,
                                                  lambda = lambda.mb)


#==============================================================================================
# Batch gradient descent
#==============================================================================================

source("batch_gradient_descent.R")

# Parameters
batch = 200
learning_rate.b = 1
lambda.b = 1

set.seed(21)
trained_parameters <- batch_gradient_descent(input=input, label=label, 
                                                 parameters=parameters, batch=batch, 
                                                 learning_rate=learning_rate.b,
                                                  lambda = lambda.b)


#============================================================================================
# Evaluation
#============================================================================================

#===================
# Training accuracy
#===================

# Predict digits and accuracy
source("predict_digit.R")
ff <- feedforward(input = input, parameter.list = trained_parameters)
pred.y <- predict_digit(ff$a[[3]])
paste("Training accuracy:", mean(pred.y==true.y))

#=====================
# Validation accuracy
#=====================

# Compute output values for validation set
ff.val <- feedforward(input = val[,-1], parameter.list = trained_parameters)
val.y <- predict_digit(ff.val$a[[3]])
paste("Validation accuracy:", mean(val.y==val[,1]))


