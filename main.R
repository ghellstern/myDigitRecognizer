# Mini-batch/batch gradient descent

#========
# Inputs
#========

# Training set 
input <- train[,-1]
y_label <- label.train
parameters <- parameters.train
true.y <- train[,1]

# Sample set
input <- train.sample[,-1]
y_label <- label.sample
parameters <- parameters.sample
true.y <- train.sample[,1]

# Mnist set 
input <- mnist.scaled[,-1]
y_label <- label
parameters <- parameters.train
true.y <- mnist.scaled[,1]


#=================================================================================================
# Mini-batch gradient descent
#=================================================================================================

source("miniBatch_gradient_descent.R")

# Parameters
update_per_epoch = 200
batch_size = nrow(input)/update_per_epoch
epoch = 30
learning_rate.mb = 1
lambda.mb = 10

set.seed(21)
trained_parameters <- miniBatch_gradient_descent(input=input, label=y_label, 
                                                 parameters=parameters, batch_size=batch_size, 
                                                 epoch=epoch, learning_rate=learning_rate.mb,
                                                  lambda = lambda.mb)


#==============================================================================================
# Batch gradient descent
#==============================================================================================

source("batch_gradient_descent.R")

# Parameters
batch = 200
learning_rate.b = 1
lambda.b = 0

set.seed(21)
trained_parameters <- batch_gradient_descent(input=input, label=y_label, 
                                                 parameters=parameters, batch=batch, 
                                                 learning_rate=learning_rate.b,
                                                  lambda = lambda.b)


#============================================================================================
# Evaluation
#============================================================================================

source("predict_digit.R")

#===================
# Training accuracy
#===================

# Predict digits and accuracy
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


#===================
# Predict test set
#===================

ff.test <- feedforward(input = test, parameter.list = trained_parameters)
test.y <- predict_digit(ff.test$a[[3]])
prop.table(table(test.y))
write.table(data.frame(ImageId=1:length(test.y), Label=test.y),
            file="./submission/run2.csv", 
            row.names=FALSE, sep = ",")


#================================
# Visualized trained parameters
#================================

par(mfrow=c(10,10), mar=rep(0, times=4))
for(i in 1:100) {
    ob <- matrix(-trained_parameters[[1]][i,-1], nrow=28, ncol=28, byrow=TRUE)
    ob <- t(apply(ob, MARGIN = 2, FUN = rev))
    image(ob, col = grey(seq(0, 1, length = 255)), axes=FALSE)
  }

par(mfrow=c(3,4), mar=rep(0, times=4))
for(i in 1:10) {
  ob <- matrix(-trained_parameters[[2]][i,-1], nrow=10, ncol=10, byrow=TRUE)
  ob <- t(apply(ob, MARGIN = 2, FUN = rev))
  image(ob, col = grey(seq(0, 1, length = 255)), axes=FALSE)
}
