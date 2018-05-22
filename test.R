###########session1#######
#pros:no need for feature selection
#cons:computational complexity

#ANN:feed-forward ANN or recursive ANN
#Typical ANN with multi-layer perceptron: inputL -> hiddenL -> outputL
#the number of yeats = hidden layers + output layer
download.file("https://storage.googleapis.com/kaggle-competitions-data/kaggle/3004/train.csv?GoogleAccessId=competitions-data@kaggle-161607.iam.gserviceaccount.com&Expires=1502441388&Signature=KUtoJb%2B5nONmVuz2ayQlcfMfWILL0NGY3zmxjsCySVawg7%2Far30syFvwAJc5am5yAJGF4YCqKX7S5ZM0H702pV49dk7MenKolxWgIBDXQNZrJETe6w2vx5CnuX%2BSCwj56D3DklT%2FmA%2FN2V4pX5M6DJVsYA3WYRZQVs9wQS5Ge2q7y6w02WDZ354KFcbfgGyiJCpgLVh8wqHl3GO3pLjl3ph0Ts5Z3kGLiAONtVFdwYYNWEahx4i35qYX8KM0p71wxlEtAT7pXuUi14T%2BfklCdw9l%2FLlTYIvcI9scvwe8uOWOkbemtllwviPFW211Li7ulpuPWjL7R%2FJBjzNRueW3Ig%3D%3D","test.csv")
digits.data <- read.csv("test.csv",header = TRUE)

library(nnet)#only supports 2-layer ANN (input + hidden + output)
library(caret)
library(pipeR)
digits.data$label <- factor(digits.data$label)
i <- 1:5000
digits.sample <- digits.data[i,]
digits.model <- caret::train(label~., data=digits.sample, method="nnet", 
                      tuneGrid=expand.grid(.size=c(5),.decay=0.1),
                      trControl=trainControl(method="none"),
                      MaxNWts = 10000,
                      maxit = 100)
digits.pred <- predict(digits.model)#here newdata is the training data by default
confusionMatrix(digits.sample$label,digits.pred)
barplot(table(digits.sample$label))
barplot(table(digits.pred))

digits.pred1 <- predict(digits.model,newdata=digits.data[5000:6000,])
####how to check the output neuron's probbility for each sample:add type="prob"####
digits.pred2 <- predict(digits.model,newdata=digits.data[5000:6000,],type="prob")
###################################################################################
confusionMatrix(digits.pred1,digits.data[5000:6000,]$label)

###########session3#######
###################Multilayer ANN##############
#######caret requires RSNNS (Stuttgart Neural Network Simulator)###
mlp_grid = expand.grid(layer1 = 3,layer2 = 3,layer3=0)
digits.model <- caret::train(label~., data=digits.sample, method="mlpML", 
                             tuneGrid=mlp_grid,
                             trControl=trainControl(method="none"),
                             MaxNWts = 10000,
                             maxit = 100)


####H2O package is more flexible than NNET and CARET#######
library(h2o)
h2o.init()

train <- digits.data[1:5000,]
valid <- digits.data[5001:10000,]
test <- digits.data[10001:15000,]
train_h2o <- as.h2o(train)
test_h2o <- as.h2o(test)
model <- h2o.deeplearning(x=2:785,y=1,training_frame = train_h2o,hidden=c(5,10),seed=0)
preds <- h2o.predict(model,test_h2o)
h2o.confusionMatrix(model,test_h2o)

library(mlbench)#for cancer data
data(BreastCancer)
h2o.init()
summary(BreastCancer)
data <- subset(BreastCancer,select = -Id)
DT <- as.data.frame(sapply(data,as.numeric))#treat DF as a list and apply to every element
DT$Class <- as.factor(DT$Class)
splitSample <- sample(1:3,size=nrow(DT),prob = c(0.6,0.2,0.2),replace=TRUE)#sample and allocate index 1,2,3 for all rows
train_h2o <- as.h2o(DT[which(splitSample==1),])
valid_h2o <- as.h2o(DT[which(splitSample==2),])
test_h2o <- as.h2o(DT[which(splitSample==3),])
model <- h2o.deeplearning(x=1:9,y=10,training_frame = train_h2o,activation = "TanhWithDropout",input_dropout_ratio = 0.2,
                          balance_classes = TRUE,hidden=c(10,10),hidden_dropout_ratios = c(0.3,0.3),epochs = 10,seed=0)
#dropout ratio: at each layer, neurons may be dropped(deactivated) at a certain ratio as a way of regularization
#dropout randomly ignores neurons during training to avoid overfitting. Prob defines dropout rate at each layer"
#Epoch:How many times the dataset should be iterated (streamed), can be fractional
#Iteration:number of weight updates (forward propagation + backward propogtion)
#batch: number of data samples propagated each iteration.small batch (more iterations each with small data set) is better than large bacth
#each batch triggers one forward/backword propagation -> update weights
#cost function each time: average error for one batch this iteration
h2o.confusionMatrix(model)
h2o.confusionMatrix(model,valid_h2o)
h2o.confusionMatrix(model,test_h2o)

######hyper-parameters optimization########
hyper_params <- list(
  activation = c("Rectifier","Tanh","Maxout","TanhWithDropout","MaxoutWithDropout"),
  hidden=list(c(20,20),c(50,50),c(30,30,30),c(25,25,25,25)),
  input_dropout_ratio=c(0,0.05),
  l1=seq(0,1e-4,1e-6),
  l2=seq(0,1e-4,1e-6)
)

search_criteria = list(
  strategy = "RandomDiscrete",
  max_models = 100, seed=1234567,
  stopping_tolerance=1e-2
)

dl_random_grid <- h2o.grid(
  algorithm = "deeplearning",
  grid_id="dl_grid_random",
  training_frame = train_h2o,
  validation_frame = valid_h2o,
  x=1:9,
  y=10,
  epochs=1,
  stopping_metric="logloss",
  stopping_tolerance=1e-2,
  stopping_rounds=2,
  hyper_params = hyper_params,
  search_criteria = search_criteria
)

grid <- h2o.getGrid("dl_grid_random",sort_by="logloss",decreasing=FALSE)
grid@summary_table

best_model <- h2o.getModel(grid@model_ids[[1]])
best_model

###########session4#######
#################Convolutional Neural Net################ 
#Problems with regular ANN: full connectivity, many weights, inputs, overfitting
#CNN= conv layer, pooling layer, fully-connected layer
#In CNN, each layer contains 3 dimentions: width, height, depth
#so the number of neurons and weights are more flexible
#input -> convolutional layer (filters/kernels) -> poolling layer -> FC
#no need to select features. this is done by filters/kernels.
#very good for image recognition

#best practice: conv layer: 3x3 5x5; pool layer:2x2
#famous CNN:LeNet,AlexNet 
#CNN visualization: to understand why it works
#use MXNetR package (supports GPU)
# cran <- getOption("repos")
# cran["dmlc"] <- "https://s3.amazonaws.com/mxnet-r/"
# options(repos = cran)
# install.packages("mxnet")
library(mxnet)
library(caret)
#implement LeNet7
train <- digits.data[1:5000,]
test <- digits.data[5001:10000,]
#train<-read.csv('https://github.com/ozt-ca/tjo.hatenablog.samples/raw/master/r_samples/public_lib/jp/mnist_reproduced/short_prac_train.csv')
#test<-read.csv('https://github.com/ozt-ca/tjo.hatenablog.samples/raw/master/r_samples/public_lib/jp/mnist_reproduced/short_prac_test.csv')
train <- as.matrix(train)
test <- as.matrix(test)
train.x <- train[,-1]#except the first column
train.y <- train[,1]
test.x <- test[,-1]
test.y <- test[,1]
#normalize values very importnt for neural networks
train.x <- t(train.x/255)#transpose to accommodate MXNET: row:feature column:records
test.x <- t(test.x/255)#transpose to accommodate MXNET: row:feature column:records

############################my own###############
###implement LeNet7
#input layer
data <- mx.symbol.Variable('data')
#first conv
conv1 <- mx.symbol.Convolution(data=data,kernel=c(5,5),num_filter=20)
tanh1 <- mx.symbol.Activation(data=conv1,act_type="tanh")
pool1 <- mx.symbol.Pooling(data=tanh1,pool_type="max",kernel=c(2,2),stride=c(2,2))
#second conv
conv2 <- mx.symbol.Convolution(data=pool1,kernel=c(5,5),num_filter=50)
tanh2 <- mx.symbol.Activation(data=conv2,act_type="tanh")
pool2 <- mx.symbol.Pooling(data=tanh2,pool_type="max",kernel=c(2,2),stride=c(2,2))
#first fully connected layer
flatten <- mx.symbol.Flatten(data=pool2)#transfer pooled matrix to an array as input layer for fc
fc1 <- mx.symbol.FullyConnected(data=flatten,num_hidden=500)
tanh3 <- mx.symbol.Activation(data=fc1,act_type="tanh")
#second fully connected layer
fc2 <- mx.symbol.FullyConnected(data=tanh3,num_hidden=10)#num_hidden here = number of output classes
lenet <- mx.symbol.SoftmaxOutput(data=fc2)#softmax activation differentiable to train by gradient descent
###########################

#why???no difference in acc: because we need to specify input layer as (width, height, depth, etc)
#when depth =1 no need to specify
dim(train.x) <- c(28,28,1,ncol(train.x))#ncol due to t()
dim(test.x) <- c(28,28,1,ncol(test.x))#ncol due to t()

mx.set.seed(1)
tic <- proc.time()
model <- mx.model.FeedForward.create(lenet, X=train.x, y=train.y,
                                     ctx=mx.gpu(), num.round=30, array.batch.size=100,
                                     learning.rate=0.05, momentum=0.9, wd=0.00001,
                                     eval.metric=mx.metric.accuracy,
                                     epoch.end.callback=mx.callback.log.train.metric(100))
#momentum is the step of search in gradient descent

#epoch:times that total data passed through
#mx.gpu() or mx.cpu()
preds <- predict(model,test.x)
dim(preds)# col x row due to t(). contains output layer prob for all classes
# to find out the CNN decision that is the max(prob) among output classes
# -1 since classes start from 1 to 10
pred.label <- apply(preds,2,which.max)-1
pred.label <- max.col(t(preds))-1
caret::confusionMatrix(as.integer(pred.label),test.y)

#############practice cnn##########
#in practice we usually use pre-trained cnn as a feature extractor by removing the last fc layer
#or fine tune the pre-trained cnn by back propagation for our own applications
library(mxnet)
library(imager)

model = mx.model.load("Inception/Inception_BN",iteration=39)
mean.img = as.array(mx.nd.load("Inception/mean_224.nd"))[["mean_img"]]#to center data

im <- load.image("cat.jpg")
plot(im)

preproc.image <- function(im, mean.image) {
  # crop the image
  shape <- dim(im)
  short.edge <- min(shape[1:2])
  xx <- floor((shape[1] - short.edge) / 2)
  yy <- floor((shape[2] - short.edge) / 2)
  croped <- crop.borders(im, xx, yy)
  # resize to 224 x 224, needed by input of the model.
  resized <- imager::resize(croped, 224, 224)
  # convert to array (x, y, channel)
  arr <- as.array(resized) * 255
  dim(arr) <- c(224, 224, 3)
  # substract the mean
  normed <- arr - as.array(mean.img)#add "as.array to eliminte error"
  # Reshape to format needed by mxnet (width, height, channel, num)
  dim(normed) <- c(224, 224, 3, 1)
  return(normed)
}

normed <- preproc.image(im,mean.img)
probs <- predict(model,normed)
max.idx <- order(probs,decreasing = TRUE)[1:5]#return index

synsets <- readLines("Inception/synset.txt")
print(paste("Predicted top classes: ",synsets[max.idx],probs[max.idx]))


######Session 5########
#################Recurrent Neural Net###############
#design for time-related data to predict next moment (TS & language modelling)
##deal with correlated dependent data where input dat changes
##great for modelling short-term memory which means fast-changing netowkrs
#the current network state h_t = f_w(w_hh*h_t-1+w_xh*x_t); y_t = h_t*w_hy
#**above h_t, w_hh, x_t...etc are all vectors
#weight update:Backpropagation Through Time (BPTT)
#application: language transpation, speech recognition, stock price prediction, image captioning

#LSTM is designed to achieve long-term dependency
#http://colah.github.io/posts/2015-08-Understanding-LSTMs/
#the hidden layer h_t(w_hh*h_t-1,w_xh*x_t) is now composed of four hidden layers(4 gates + 1 cell state)
#every gate is a function with its own weight, every gate outputs a numeric to control the ratio of past/now moments
#0. cell state (memory, a vector like hidden layer) stores past and updates for every new input 
#1. forget gate layer (h_t-1,x_t), a sigmoid layer -> how much past to remember when we see new input?
#2. input gate layer (sigmoid layer)+tanh gate layer (h_t-1,x_t)-> what new to add?
#till now the cell state is updated from c_t-1 to c_t
#3. sigmoid layer (h_t-1,x_t) x tanh(c_t) -> filter(extract) c_t for this layer's h_t 
#final output for next moment: c_t and h_t (e.g. c_t:verb, h_t:singular)
#LSTM can manipulate gates to dynamically enhance/remove moments in the memory so it supports long-term dependency
#but original RNN treats every moment following the same pattern (x_t > x_t-1 > x_t-2...or something) due to same w_hh,w_xh,w_hy

#In R we can use RNN, MxNET, or TensorFlow for RNN implementation
#RNN for original RNN
library(rnn)
set.seed(10)

f <- 5
w <- 2*pi*f
t <- seq(0.005,2,0.005)
x <- sin(t*w)+rnorm(200,0,0.25)#if length(rnorm) < length(data), then random numbers will repeat
y <- cos(t*w)
y2 <- x[81:400]###if do TS forecast, need to do time shift for supervised learning (unlike ARIMA)

#Transfer the series to 10 (so nrow = 200/20=10)  timestamps seris for RNN
X <- matrix(x,nrow = 40)#each col is one time series
Y <- matrix(y,nrow=40)
Y2 <- matrix(y2,nrow=40)
#standrize/normalize data for neural networks
X <- (X-min(X))/(max(X)-min(X))
Y <- (Y-min(Y))/(max(Y)-min(Y))
Y2 <- (Y2-min(Y2))/(max(Y2)-min(Y2))
#transpose X,Y same as what happened for CNN
X <- t(X)#now each row is a trunk of TS
Y <- t(Y)
Y2 <- t(Y2)

#####train and test
train <- 1:8
test <- 9:10
test2 <- 3:5
test3 <- 5:7

###train model
model <- trainr(Y=Y[train,],X=X[train,],learningrate = 0.05,hidden_dim = 16,numepochs = 1500)
model2 <- trainr(Y=Y2,X=X[train,],learningrate = 0.05,hidden_dim = 16,numepochs = 1500)

###predict supervised time series
preds <- predictr(model,X[test,])
plot(as.vector(t(Y[test,])),type="l",col="blue")
lines(as.vector(t(preds)),type="l",col="red")

pred2 <- predictr(model2,X[test2,])
plot(as.vector(t(X[test3,])),type="l",col="blue")
lines(as.vector(t(pred2)),type="l",col="red")

###how to use R for remote json DL service
library(jsonlite)
res <- fromJSON('http://myserver.com/get_results')

####Use case for char-level english spelling by LSTM
#H2o for ANN, MxNet for CNN(LeNet) + RNN(LSTM)
library(mxnet)
batch.size = 32#in rnn, one batch containes many sequences (each = sentence)
seq.len = 32#each sequence contains # time stemps (char or word)
#all sequences in one batch have same length
#but each batch could have its own length for its sequences
#this enables batch padding to save memory
#after padding, each time, the n-th time steps of all sentences in one batch are fed into rnn in parallel

num.hidden = 200
num.embed = 16
num.lstm.layer = 1#layers of LSTM: number of cells between x_t and y_t (vertically)
num.round = 1
learning.rate = 0.1
wd=0.00001
clip_gradient=1
update.period = 1

download.data <- function(data_dir) {
  dir.create(data_dir, showWarnings = FALSE)
  if (!file.exists(paste0(data_dir,'input.txt'))) {
    download.file(url='https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tinyshakespeare/input.txt',
                  destfile=paste0(data_dir,'input.txt'))
  }
}

make.dict <- function(text, max.vocab=10000) {
  text <- strsplit(text, '')
  dic <- list()
  idx <- 1
  for (c in text[[1]]) {
    if (!(c %in% names(dic))) {
      dic[[c]] <- idx
      idx <- idx + 1
    }
  }
  if (length(dic) == max.vocab - 1)
    dic[["UNKNOWN"]] <- idx
  cat(paste0("Total unique char: ", length(dic), "\n"))
  return (dic)
}

make.data <- function(file.path, seq.len=32, max.vocab=10000, dic=NULL)      {
  fi <- file(file.path, "r")
  text <- paste(readLines(fi), collapse="\n")#transfer to string with \n #collapse vector -> string
  close(fi)
  
  if (is.null(dic))
    dic <- make.dict(text, max.vocab)
  lookup.table <- list()
  for (c in names(dic)) {
    idx <- dic[[c]]
    lookup.table[[idx]] <- c
  }
  
  char.lst <- strsplit(text, '')[[1]]#spli to charater-level
  num.seq <- as.integer(length(char.lst) / seq.len)#interger number of sequences
  char.lst <- char.lst[1:(num.seq * seq.len)]#truncate residual char
  data <- array(0, dim=c(seq.len, num.seq))
  idx <- 1
  for (i in 1:num.seq) {
    for (j in 1:seq.len) {
      if (char.lst[idx] %in% names(dic))
        data[j, i] <- dic[[ char.lst[idx] ]]-1
      else {
        data[j, i] <- dic[["UNKNOWN"]]-1
      }
      idx <- idx + 1
    }
  }#then each column is one sequence
  return (list(data=data, dic=dic, lookup.table=lookup.table))
}

drop.tail <- function(X, batch.size) {
  shape <- dim(X)
  nstep <- as.integer(shape[2] / batch.size)
  return (X[, 1:(nstep * batch.size)])
}#make total sentences be multiple integer of batch.size

get.label <- function(X) {
  label <- array(0, dim=dim(X))
  d <- dim(X)[1]
  w <- dim(X)[2]
  for (i in 0:(w-1)) {
    for (j in 1:d) {
      label[i*d+j] <- X[(i*d+j)%%(w*d)+1]#shift sequence by one character, last character goes back to the begining
    }
  }
  return (label)
}

download.data("./data/")
ret <- make.data("./data/input.txt",seq.len=seq.len)

X <- ret$data
dic <- ret$dic
lookup.table <- ret$lookup.table

vocab <- length(dic)

shape <- dim(X)
train.val.fraction <- 0.9
size <- shape[2]

X.train.data <- X[,1:as.integer(size*train.val.fraction)]#using 90% of the sequences for training
X.val.data <- X[,-(1:as.integer(size*train.val.fraction))]#using 10% of the sequences for training
X.train.data <- drop.tail(X.train.data,batch.size)#make training sentences be multiple integer of batch.size
X.val.data <- drop.tail(X.val.data,batch.size)

X.train.label <- get.label(X.train.data)#shift by one character 
X.val.label <- get.label(X.val.data)#shift by one character

X.train <- list(data=X.train.data, label=X.train.label)
X.val <- list(data=X.val.data, label=X.val.label)

model <- mx.lstm(X.train,X.val,
                 ctx=mx.cpu(),
                 num.round=num.round,
                 update.period = update.period,
                 num.lstm.layer = num.lstm.layer,
                 seq.len = seq.len,
                 num.hidden = num.hidden,
                 num.embed = num.embed,
                 num.label = vocab,
                 batch.size = batch.size,
                 input.size = vocab,
                 initializer = mx.init.uniform(0.1),
                 learning.rate=learning.rate,
                 wd=wd,
                 clip_gradient=clip_gradient)

cdf <- function(weights) {
  total <- sum(weights)
  result <- c()
  cumsum <- 0
  for (w in weights) {
    cumsum <- cumsum+w
    result <- c(result, cumsum / total)
  }
  return (result)
}

search.val <- function(cdf, x) {
  l <- 1
  r <- length(cdf)
  while (l <= r) {##this is binary search
    m <- as.integer((l+r)/2)
    if (cdf[m] < x) {
      l <- m+1
    } else {
      r <- m-1
    }
  }
  return (l)
}
choice <- function(weights) {
  cdf.vals <- cdf(as.array(weights))
  x <- runif(1)
  idx <- search.val(cdf.vals, x)#find an index of weight closest to x
  return (idx)
}
make.output <- function(prob, sample=FALSE) {
  if (!sample) {
    idx <- which.max(as.array(prob))
  }
  else {
    idx <- choice(prob)
  }
  return (idx)
  
}

infer.model <- mx.lstm.inference(num.lstm.layer=num.lstm.layer,
                                 input.size=vocab,
                                 num.hidden=num.hidden,
                                 num.embed=num.embed,
                                 num.label=vocab,
                                 arg.params=model$arg.params,
                                 ctx=mx.cpu())

start <- 'H'
seq.len <- 75
random.sample <- TRUE

last.id <- dic[[start]]
out <- "H"
for (i in (1:(seq.len-1))) {#expanding rnn
  input <- c(last.id-1)
  ret <- mx.lstm.forward(infer.model, input, FALSE)#h_t-1 and x_t
  infer.model <- ret$model
  prob <- ret$prob
  last.id <- make.output(prob, random.sample)
  out <- paste0(out, lookup.table[[last.id]])
}
cat (paste0(out, "\n"))

######Session 6########
############Autoencoder by ANN########
##a powerful tool to represent inputs for unsupervised learning
##outputs = inputs so backpropagation for unlabelled data
##hidden layer containes compressed information of inputs (like pca but pca only captures linearity)
##common packages: Autoencoder, H2O, MxNetR

##############restricted Boltzman Machine(RBM)#####
##a recurrent network with visible layer (data x) and hidden layer (h): no output 
##RBM models p(x,h) after N iterations backpropagtions

#############Deep Belief Network (DBN)
##can be regarded as multi-layer RBM###
##some layers become the visible layer to next
##packages:Deepnet, Darch, RcppDL, MxNetR

##RL learns a policy, Supervised ML learns a representation
##Deep RL = DL + RL: use DL to represent Q function (total reward) or policy in RL
##use stochastic gradient descent so find weights to max{Q}
##reason:traditional RL cannot handle many states/actions (too long to converge from iterations)
##so Deep RL is a subset of RL

#############Anomaly detection using Autoencoder##########
#1.human activcity recognition
library(ggplot2)
library(h2o)
h2o.init(nthreads = -1)

use.train.x <- read.table("data/UCI HAR Dataset/train/X_train.txt")
use.test.x <- read.table("data/UCI HAR Dataset/test/X_test.txt")

use.train.y <- read.table("data/UCI HAR Dataset/train/y_train.txt")[[1]]
use.test.y <- read.table("data/UCI HAR Dataset/test/y_test.txt")[[1]]

use.labels <- read.table("data/UCI HAR Dataset/activity_labels.txt")

h2oactivity.train <- as.h2o(
  use.train.x,
  destination_frame = "h2oactivitytrain"
)

h2oactivity.test <- as.h2o(
  use.test.x,
  destination_frame = "h2oactivitytest"
)

model <- h2o.deeplearning(
  x = colnames(h2oactivity.train),
  training_frame = h2oactivity.train,
  validation_frame = h2oactivity.test,
  activation = "RectifierWithDropout",
  autoencoder = TRUE,
  hidden = c(200),
  epochs = 30,
  sparsity_beta = 0,
  input_dropout_ratio = 0,
  hidden_dropout_ratios = c(0),
  l1=0,
  l2=0
)

#Next, we plot the reconstruction.MSE and see where the model grapples to reconstruct the original records. 
#This means that the model was not able to learn patterns for those records correctly which can be an anomaly.
error <- as.data.frame(h2o.anomaly(model,h2oactivity.train)[1:1000,])

ggplot(error,aes(Reconstruction.MSE))+geom_histogram(binwidth = 0.001)

i.anomolous <- error$Reconstruction.MSE >= quantile(error[[1]],probs = 0.99)
# for factors, we can use factor[index] to achieve the corresponding label
ggplot(as.data.frame(table(use.labels$V2[use.train.y[i.anomolous]])),aes(Var1,Freq))+geom_bar(stat = "identity")+
  theme_bw()+theme(axis.text.x = element_text(angle=45))

######Session 7 DL Application########
############Computer Vision via CNN########
#object recognition/detection in image and video
#realistic image generation
#BW coloring
#dense object detection (simoutaneously label many) via FRCNN
############NLP########
############Audio########
#WaveNet
############Multimodal Task########
#this combines language, audio, video...
#Automatic Image Captioning(CNN+RNN)
#visual translation
#reading lips
#adding sounds to silent movies
############Other APPs############
#recommendation engine
#cancer detction
#anomaly detection (autoencoder)
#market price forecasting(RNN)

######Session 8 Advanced topics########
#########Debugging DL system#########
#start with simple model and synthetic data
#########GPU & CPU###################
#Parallel data & model
#Specialized GPU & FPGA is the trend
#########Compare DL packages in R####
#For ANN and AutoEncoder use H2O
#For others use MxNet (similiar to Tensorflow)
#Tensor flow?to be checked.

###test
