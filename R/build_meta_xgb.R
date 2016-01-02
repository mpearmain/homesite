## wd etc ####
require(readr)
require(xgboost)
require(stringr)

seed_value <- 1234
model_type <- "xgb"
todate <- str_replace_all(Sys.Date(), "-","")

## functions ####

msg <- function(mmm,...)
{
  cat(sprintf(paste0("[%s] ",mmm),Sys.time(),...)); cat("\n")
}


auc<-function (actual, predicted) {
  
  r <- as.numeric(rank(predicted))
  
  n_pos <- as.numeric(sum(actual == 1))
  n_neg <- as.numeric(length(actual) - n_pos)
  auc <- (sum(r[actual == 1]) - n_pos * (n_pos + 1)/2)/(n_pos *  n_neg)
  auc
  
}

## data ####
# division into folds: 5-fold
xfolds <- read_csv("./input/xfolds.csv"); xfolds$fold_index <- xfolds$fold5
xfolds <- xfolds[,c("QuoteNumber", "fold_index")]
nfolds <- length(unique(xfolds$fold_index))

## fit xgb models ####
# parameter grid
param_grid <- expand.grid(data  = c(paste("kb",c(3,4), sep = ""), paste("mp", 1:2,sep = "")),
                          eta = 0.01, nrounds = c(6000))

xsub <- read_csv("./submissions/xmix2_20151229.csv")

# storage structures 
mtrain <- array(0, c(nrow(xfolds), nrow(param_grid)))
mtest <- array(0, c(nrow(xsub), nrow(param_grid)))

# loop over parameters
for (ii in 1:nrow(param_grid))
{
  xtrain <- read_csv(paste("./input/xtrain_",as.character(param_grid[ii,1]),".csv", sep = ""))
  xtest <- read_csv(paste("./input/xtest_",as.character(param_grid[ii,1]),".csv", sep = ""))
  y <- xtrain$QuoteConversion_Flag; xtrain$QuoteConversion_Flag <- NULL
  id_train <- xtrain$QuoteNumber
  id_test <- xtest$QuoteNumber
  xtrain$QuoteNumber <- xtest$QuoteNumber <- NULL
  
  # loop over folds 
  for (jj in 1:nfolds)
  {
    isTrain <- which(xfolds$fold_index != jj)
    isValid <- which(xfolds$fold_index == jj)
    x0 <- xtrain[isTrain,]; x1 <- xtrain[isValid,]
    # convert to xgb format
 
    x0d <- xgb.DMatrix(as.matrix(x0), label = y[isTrain])
    x1d <- xgb.DMatrix(as.matrix(x1), label = y[isValid])
    watch <- list(valid = x0d)
    
    clf <- xgb.train(booster = "gbtree",
                     maximize = TRUE, 
                     print.every.n = 25,
                     early.stop.round = 25,
                   nrounds = 600,
                     eta = 0.01,
                     max.depth = 15, 
                     colsample_bytree = 0.85,
                     subsample = 0.8,
                     data = x1d, 
                     objective = "binary:logistic",
                     watchlist = watch, 
                     eval_metric = "auc")
    
    mtrain[isValid,ii] <- predict(clf, x1d)
  }
  
  # full version 
  x0d <- xgb.DMatrix(as.matrix(xtrain), label = y)
  x0d <- xgb.DMatrix(as.matrix(xtrain), label = y)
  
  clf <- xgb.train(booster = "gbtree",
                   maximize = TRUE, 
                   print.every.n = 25,
                   nrounds = 6000,
                   eta = 0.01,
                   max.depth = 12, 
                   colsample_bytree = 0.85,
                   subsample = 0.8,
                   data = x0d, 
                   objective = "binary:logistic",
                   eval_metric = "auc")
  
  mtest[,ii] <- predict(clf, x1d)
  
}

## store complete versions ####
mtrain <- data.frame(mtrain)
mtest <- data.frame(mtest)
mtrain$QuoteNumber <- id_train
mtest$QuoteNumber <- id_test
mtrain$QuoteConversion_Flag <- y

write_csv(mtrain, path = paste("./metafeatures/prval_",model_type,"_", todate, "_data", dataset_version, "_seed", seed_value, ".csv",sep = "" ))
write_csv(mtest, path = paste("./metafeatures/prfull_",model_type,"_", todate, "_data", dataset_version, "_seed", seed_value, ".csv",sep = "" ))

