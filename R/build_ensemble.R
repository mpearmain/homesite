## wd etc ####
require(readr)
require(stringr)
require(glmnet)
require(caret)
require(xgboost)

seed_value <- 132
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
# list the groups 
xlist_val <- dir("./metafeatures/", pattern = "prval", full.names = T)
xlist_full <- dir("./metafeatures/", pattern = "prfull", full.names = T)

# aggregate validation set
ii <- 1
mod_class <- str_split(xlist_val[[ii]], "_")[[1]][[2]]
xvalid <- read_csv(xlist_val[[ii]])
xcols <- colnames(xvalid)[1:(ncol(xvalid)-2)]
xcols <- paste(xcols , ii, sep = "")
colnames(xvalid)[1:(ncol(xvalid)-2)] <- xcols

for (ii in 2:length(xlist_val))
{
  mod_class <- str_split(xlist_val[[ii]], "_")[[1]][[2]]
  xval <- read_csv(xlist_val[[ii]])
  xcols <- colnames(xval)[1:(ncol(xval)-2)]
  xcols <- paste(xcols , ii, sep = "")
  colnames(xval)[1:(ncol(xval)-2)] <- xcols
  xvalid <- merge(xvalid, xval)
  msg(ii)
}

# aggregate test set
ii <- 1
mod_class <- str_split(xlist_full[[ii]], "_")[[1]][[2]]
xfull <- read_csv(xlist_full[[ii]])
xcols <- colnames(xfull)[1:(ncol(xfull)-1)]
xcols <- paste(xcols , ii, sep = "")
colnames(xfull)[1:(ncol(xfull)-1)] <- xcols

for (ii in 2:length(xlist_val))
{
  xval <- read_csv(xlist_full[[ii]])
  xcols <- colnames(xval)[1:(ncol(xval)-1)]
  xcols <- paste(xcols , ii, sep = "")
  colnames(xval)[1:(ncol(xval)-1)] <- xcols
  xfull <- merge(xfull, xval)
  msg(ii)
}

rm(xval)

## build ensemble model ####

y <- xvalid$QuoteConversion_Flag; xvalid$QuoteConversion_Flag <- NULL
id_valid <- xvalid$QuoteNumber; xvalid$QuoteNumber <- NULL
id_full <- xfull$QuoteNumber; xfull$QuoteNumber <- NULL

# folds for cv evaluation
# xfolds <- createDataPartition(y, 
#                     times = 20,
#                     p = 0.1,
#                     list = TRUE,
#                     groups = min(5, length(y)))
# 
# nfolds <- length(xfolds)
xfolds <- read_csv("./input/xfolds.csv"); xfolds$fold_index <- xfolds$fold5
xfolds <- xfolds[,c("QuoteNumber", "fold_index")]
nfolds <- length(unique(xfolds$fold_index))


storage_matrix <- array(0, c(nfolds, 6))

for (ii in 1:nfolds)
{
  isTrain <- which(xfolds$fold_index != ii)
  isValid <- which(xfolds$fold_index == ii)
  x0 <- xvalid[isTrain,];   x1 <- xvalid[isValid,]
  y0 <- y[isTrain];  y1 <- y[isValid]
#   x0 <- xvalid[-isValid,]; x1 <- xvalid[isValid,]
#   y0 <- y[-isValid]; y1 <- y[isValid]
  
  mod0 <- glmnet(x = as.matrix(x0), y = y0, alpha = 0)
  prx <- predict(mod0,as.matrix(x1))
  prx1 <- prx[,ncol(prx)]
  
  storage_matrix[ii,1] <- auc(y1,prx1)
 
  x0d <- xgb.DMatrix(as.matrix(x0), label = y0)
  x1d <- xgb.DMatrix(as.matrix(x1), label = y1)
  
  watch <- list(valid = x1d)
  
  clf <- xgb.train(booster = "gbtree",
                   maximize = TRUE, 
                   print.every.n = 25,
                   # early.stop.round = 25,
                   nrounds = 250,
                   eta = 0.01,
                   max.depth = 15, 
                   colsample_bytree = 0.85,
                   subsample = 0.8,
                   data = x0d, 
                   objective = "binary:logistic",
                   watchlist = watch, 
                   eval_metric = "auc",
                   gamma= 0.05)
  
  prx2 <- predict(clf, x1d)
  storage_matrix[ii,2] <- auc(y1,prx2)
  
  storage_matrix[ii,3] <- auc(y1, prx1 + prx2)
  storage_matrix[ii,4] <- auc(y1, rank(prx1) + rank(prx2))
  
  a <- apply(x1,2,function(s) auc(y1,s))
  storage_matrix[ii,5] <- max(a)
  storage_matrix[ii,6] <- which.max(a)
}

## build final prediction

mod0 <- glmnet(x = as.matrix(xvalid), y = y, alpha = 0)
prx <- predict(mod0,as.matrix(xfull))
prx1 <- prx[,ncol(prx)]

x0d <- xgb.DMatrix(as.matrix(xvalid), label = y)
x1d <- xgb.DMatrix(as.matrix(xfull))

clf <- xgb.train(booster = "gbtree",
                 maximize = TRUE, 
                 print.every.n = 25,
                 #early.stop.round = 25,
                 nrounds = 250,
                 eta = 0.01,
                 max.depth = 15, 
                 colsample_bytree = 0.85,
                 subsample = 0.8,
                 data = x0d, 
                 objective = "binary:logistic",
                 # watchlist = watch, 
                 eval_metric = "auc",
                 gamma= 0.05)


prx2 <- predict(clf, x1d)

# combine into the final forecast
xfor <- data.frame(QuoteNumber = id_full, QuoteConversion_Flag = 0.5 * (rank(prx1) + rank(prx2)))
write_csv(xfor, path = "./submissions/ens_20160101.csv")