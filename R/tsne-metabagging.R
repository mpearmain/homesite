require(xgboost)
require(methods)
require(randomForest)
require(Rtsne)
require(data.table)
require(Metrics)

options(scipen=999)
set.seed(1004)

# Load T-SNE data set.
train = fread('input/xtrain_mp2.csv',header=TRUE,data.table=F)
test = fread('input/xtest_mp2.csv',header=TRUE,data.table = F)
submission = fread('input/sample_submission.csv')

# Remove Quotenumber col
train = train[,-1]
test = test[,-1]

#set and remove quote flag.
y = train[,ncol(train)]
train[,-ncol(train)]
##################################### Meta-Bagging models ####################################

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
write.csv(pred,file='ouptut/tsne-metabagging.csv', quote=FALSE,row.names=FALSE)