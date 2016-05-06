initParameters <- function(size){
  # Title: 
    # Initialize weights & biases for a neural network(nn).
  
  # Argument:
    # size - int vector indicating size of nn
  
  # Value:
    # Return a list of matrices
  
  # Description:
    # Weights have mean 0 and standard deviation of 1/sqrt(n_in),
    # where n_in = no of neurons in previous layer.
    # Biases initialize with value 1.
    # Given size=c(n1, n2, n3), 3 matrices are returned in list format.
    # 1st column in each matrix is the bias
    # dim(matrix1) = [n2, n1+1]
    # dim(matrix2) = [n3, n2+1]
  
  # Example:
    # Given size=c(784, 100, 10),
    # where nn has 784 inputs,
    # 100 neurons in first hidden layer & 10 outputs
    # dim(matrix1) = [100, 785]
    # dim(matrix2) = [10, 101]
  
  matrix_list <- list()
  
  for(i in 1:(length(size)-1)) {
    # n_in = no of neurons in layer i
    n_in = size[i] 
    
    # n_out = no of neurons in layer i+1
    n_out = size[i+1]
    
    # matrx = matrix of weights/biases connecting layer i and i+1
    matrx <- matrix(data=rnorm(n=(n_in+1)*n_out,
                               mean=0, 
                               sd=sqrt(1/n_in)),
                    nrow=n_out, ncol=n_in+1)
    matrx[,1] <- 1
   
    # Store matrx in a list
    matrix_list[[i]] <- matrx
  }
  
  return(matrix_list)
}
