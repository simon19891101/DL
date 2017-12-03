#Keras is an API with tensorflow in the backend
#Better switch to admin mode for dependencies (rcpp,reticulate)
#install.packages("keras") or devtools::install_github("rstudio/keras")
#devtools::install_github("rstudio/tensorflow")
#devtools::install_github("rstudio/keras")
library(keras)
library(tensorflow)
library(dplyr)
library(magrittr)
#install keras with tensorflow as the backend
install_keras()
#for win OS this steps needs updating VS SDK as below:
#*I hate windows
#https://social.msdn.microsoft.com/Forums/vstudio/en-US/2f3d5130-f3f9-4771-a6f2-c5883aea0414/installing-windows-10-sdk-into-visual-studio?forum=vssetup
#install_tensorflow()
#useless here for my MAC with integrated graphic cards
#install_tensorflow(version = "gpu")

#same old MNIST handwritten digits
data <- dataset_mnist()
#check data structure
data %>% class
#contains 60000 28x28 figures
data$train$x %>% dim

#make train and test
train_x <- data$train$x
train_y <- data$train$y
test_x <- data$test$x
test_y <- data$test$y
#check balance. Pretty balanced and no minority.
prop.table(table(train_y))

#make input array (3D -> 2D: 60000*28*28 -> 60000*784) and normalized
train_x <- train_x %>% array(dim=c(dim(train_x)[1],prod(dim(train_x)[-1])))/255
test_x <- test_x %>% array(dim=c(dim(test_x)[1],prod(dim(test_x)[-1])))/255
#make target variable binary encoded (like dummy variables which only supports intergers)
train_y <- train_y %>% to_categorical()
test_y <- test_y %>% to_categorical()

#define a keras sequential model
model <- keras_model_sequential()
#dense layer: each neuron is connected to each neuron in the next layer
#units is the output dimension, input_shape is the input dimension (only applicable for the 1st layer)
model %>% 
  layer_dense(units = 784, input_shape=784) %>%
  layer_dropout(rate=0.4) %>%
  layer_activation(activation="relu") %>%
  layer_dense(units=10) %>%
  layer_activation(activation="softmax")

#615440 = 784input * 784 layer1_weights + 784dropout
summary(model)

#compile model：configure a model for training 
#loss:https://keras.io/losses/
#optimizer:https://keras.io/optimizers/
#metrics:
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy","mean_squared_error")
)

#training
model %>% fit(train_x,train_y,epochs=100,batch_size=128)

#validation
validation <- model %>% evaluate(test_x,test_y,batch_size=128)



