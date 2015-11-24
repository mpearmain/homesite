require(xgboost)
require(methods)
require(randomForest)
library(Rtsne)
require(data.table)
options(scipen=999)
set.seed(1004)

train = fread('input/xtrain_mp1.csv',header=TRUE,data.table=F)
test = fread('input/xtest_mp1.csv',header=TRUE,data.table = F)
train = train[,-1]
test = test[,-1]

y = train[,ncol(train)]

x = rbind(train[,-ncol(train)],test)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))

# Run an XGB test prior to tnse features.
# Previous without was around 0.96800
trind = 1:length(y)
param <- list(objective = "binary:logistic", 
              booster = "gbtree",
              eval_metric = "auc",
              eta = 0.02,
              max_depth = 7, 
              subsample = 0.86,
              colsample_bytree = 0.68)

bst.cv = xgb.cv(param=param, data = x[trind,], label = y, nfold = 4, nrounds=1900)

# Lets run a TSNE model to see if creating extra features helps.
#x = 1/(1+exp(-sqrt(x)))

# Running this takes a LONG time.
tsne <- Rtsne(as.matrix(x), 
              check_duplicates = FALSE, 
              pca = FALSE, 
              perplexity=30, 
              theta=0.5, 
              dims=2)

# Add to the mix of features
x = cbind(x, tsne$Y[,1]) 
x = cbind(x, tsne$Y[,2])

trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)

trainX = x[trind,]
testX = x[teind,]

# Set necessary parameter
param <- list(objective = "binary:logistic", 
              booster = "gbtree",
              eval_metric = "auc",
              eta = 0.02,
              max_depth = 7, 
              subsample = 0.86,
              colsample_bytree = 0.68)

# Train the model
nround = 1900
bst = xgboost(param=param, data = trainX, label = y,nrounds=nround)

# Comparison of models.
# Without tnse
# With tnse


# Make prediction
pred = predict(bst,testX)
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)


tmpC = 1:240
tmpL = length(trind)
gtree = 200
for (z in tmpC) {
  print(z)
  tmpS1 = sample(trind,size=tmpL,replace=T)
  tmpS2 = setdiff(trind,tmpS1)
  
  tmpX2 = trainX[tmpS2,]
  tmpY2 = y[tmpS2]
  
  cst = randomForest(x=tmpX2, y=as.factor(tmpY2), replace=F, ntree=100, do.trace=T, mtry=7)
  
  tmpX1 = trainX[tmpS1,]
  tmpY1 = y[tmpS1]
  
  tmpX2 = predict(cst, tmpX1, type="prob")
  tmpX3 = predict(cst, testX, type="prob")
  
  bst = xgboost(param=param, data = cbind(tmpX1,tmpX2), label = tmpY1, column_subsample = 0.8, 
                nrounds=60, max.depth=11, eta=0.46, min_child_weight=10) 
  
  # Make prediction
  pred0 = predict(bst,cbind(testX,tmpX3))
  pred0 = matrix(pred0,9,length(pred0)/9)
  pred = pred + t(pred0)
}
pred = pred/(z+1)

pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
write.csv(pred,file='/home/mikeskim/Desktop/kaggle/otto/data/ottoHomeBagG4.csv', quote=FALSE,row.names=FALSE)