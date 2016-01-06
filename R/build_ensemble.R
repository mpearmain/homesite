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

# build an ensemble, input = parameters(initSize,howMany,blendIt, blendProp),
# input x, input y (x0 / y0 in c-v)
# output = list(weight)
buildEnsemble <- function(parVec, xset, yvec)
{
  set.seed(20130912)
  # ensemble settings
  initSize <- parVec[1]; howMany <- parVec[2];
  blendIt <- parVec[3]; blendProp <- parVec[4]
  
  # storage matrix for blending coefficients
  arMat <- array(0, c(blendIt, ncol(xset)))
  colnames(arMat) <- colnames(xset)
  
  # loop over blending iterations
  dataPart <- createDataPartition(1:ncol(arMat), times = blendIt, p  = blendProp)
  for (bb in 1:blendIt)
  {
    idx <- dataPart[[bb]];    xx <- xset[,idx]
    
    # track individual scores
    trackScore <- apply(xx, 2, function(x) auc(yvec,x))
    
    # select the individual best performer - store the performance
    # and create the first column -> this way we have a non-empty ensemble
    bestOne <- which.max(trackScore)
    mastaz <- (rank(-trackScore) <= initSize)
    best.track <- trackScore[mastaz];    hillNames <- names(best.track)
    hill.df <- xx[,mastaz, drop = FALSE]
    
    # loop over adding consecutive predictors to the ensemble
    for(ee in 1 : howMany)
    {
      # add a second component
      trackScoreHill <- apply(xx, 2,
                              function(x) auc(yvec,rowMeans(cbind(x , hill.df))))
      
      best <- which.max(trackScoreHill)
      best.track <- c(best.track, max(trackScoreHill))
      hillNames <- c(hillNames,names(best))
      hill.df <- data.frame(hill.df, xx[,best])
    }
    
    ww <- summary(factor(hillNames))
    arMat[bb, names(ww)] <- ww
  }
  
  wgt <- colSums(arMat)/sum(arMat)
  
  return(wgt)
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


storage_matrix <- array(0, c(nfolds, 8))

xMed <- apply(xvalid,1,median)
xMin <- apply(xvalid,1,min)
xMax <- apply(xvalid,1,max)
xvalid$xmed <- xMed
xvalid$xmax <- xMax
xvalid$xmin <- xMin

xMed <- apply(xfull,1,median)
xMin <- apply(xfull,1,min)
xMax <- apply(xfull,1,max)
xfull$xmed <- xMed
xfull$xmax <- xMax
xfull$xmin <- xMin

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

  mod0 <- glmnet(x = as.matrix(x0), y = y0, alpha = 1)
  prx <- predict(mod0,as.matrix(x1))
  prx2 <- prx[,ncol(prx)]
  
  storage_matrix[ii,2] <- auc(y1,prx2)
  storage_matrix[ii,3] <- auc(y1,prx2 + prx1)
  
  x0d <- xgb.DMatrix(as.matrix(x0), label = y0)
  x1d <- xgb.DMatrix(as.matrix(x1), label = y1)
  
  watch <- list(valid = x1d)
  
  clf <- xgb.train(booster = "gbtree",
                   maximize = TRUE, 
                   print.every.n = 50,
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
  
  prx3 <- predict(clf, x1d)
  storage_matrix[ii,3] <- auc(y1,prx3)
  
  storage_matrix[ii,4] <- auc(y1, prx1 + prx2)
  storage_matrix[ii,5] <- auc(y1, prx1 + prx3)
  storage_matrix[ii,6] <- auc(y1, 0.5 * (prx1 + prx2) + prx3)
  
  a <- apply(x1,2,function(s) auc(y1,s))
  storage_matrix[ii,7] <- max(a)
  storage_matrix[ii,8] <- which.max(a)
}

## build final prediction

mod0 <- glmnet(x = as.matrix(xvalid), y = y, alpha = 0)
prx <- predict(mod0,as.matrix(xfull))
prx1 <- prx[,ncol(prx)]

mod0 <- glmnet(x = as.matrix(xvalid), y = y, alpha = 1)
prx <- predict(mod0,as.matrix(xfull))
prx2 <- prx[,ncol(prx)]

x0d <- xgb.DMatrix(as.matrix(xvalid), label = y)
x1d <- xgb.DMatrix(as.matrix(xfull))

clf <- xgb.train(booster = "gbtree",
                 maximize = TRUE, 
                 print.every.n = 50,
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


prx3 <- predict(clf, x1d)

# combine into the final forecast
xfor <- data.frame(QuoteNumber = id_full, QuoteConversion_Flag = 0.5 * (prx1 + prx2))
write_csv(xfor, path = "./submissions/ens_20160105.csv")