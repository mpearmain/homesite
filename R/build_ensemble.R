## wd etc ####
require(readr)
require(stringr)
require(glmnet)
require(caret)
require(xgboost)
require(nnet)
require(ranger)

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
      msg(ee)
    }
    
    ww <- summary(factor(hillNames))
    arMat[bb, names(ww)] <- ww
    msg(paste("blend: ",bb, sep = ""))
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

# prepare the data
y <- xvalid$QuoteConversion_Flag; xvalid$QuoteConversion_Flag <- NULL
id_valid <- xvalid$QuoteNumber; xvalid$QuoteNumber <- NULL
id_full <- xfull$QuoteNumber; xfull$QuoteNumber <- NULL

# folds for cv evaluation
xfolds <- read_csv("./input/xfolds.csv"); xfolds$fold_index <- xfolds$fold5
xfolds <- xfolds[,c("QuoteNumber", "fold_index")]
nfolds <- length(unique(xfolds$fold_index))

# storage for results
storage_matrix <- array(0, c(nfolds, 5))

# storage for level 2 forecasts 
xvalid2 <- array(0, c(nrow(xvalid),5))
xfull2 <- array(0, c(nrow(xfull),5))

# amend the data
xMed <- apply(xvalid,1,median); xMin <- apply(xvalid,1,min)
xMax <- apply(xvalid,1,max); xMad <- apply(xvalid,1,mad)
xvalid$xmed <- xMed; xvalid$xmax <- xMax; xvalid$xmin <- xMin; xvalid$xmad <- xMad
xMed <- apply(xfull,1,median); xMin <- apply(xfull,1,min)
xMax <- apply(xfull,1,max); xMad <- apply(xfull,1,mad)
xfull$xmed <- xMed; xfull$xmax <- xMax; xfull$xmin <- xMin; xfull$xmad <- xMad

for (ii in 1:nfolds)
{
  # mix with glmnet 
  isTrain <- which(xfolds$fold_index != ii)
  isValid <- which(xfolds$fold_index == ii)
  x0 <- xvalid[isTrain,];   x1 <- xvalid[isValid,]
  y0 <- y[isTrain];  y1 <- y[isValid]
  prx1 <- y1 * 0
  for (jj in 1:11)
  {
    mod0 <- glmnet(x = as.matrix(x0), y = y0, alpha = (jj-1) * 0.1)
    prx <- predict(mod0,as.matrix(x1))  
    prx <- prx[,ncol(prx)]
    # storage_matrix[ii,jj] <- auc(y1,prx1)
    prx1 <- prx1 + prx
  }
  storage_matrix[ii,1] <- auc(y1,prx1)
  xvalid2[isValid,1] <- prx1
  
  # mix with xgboost
  x0d <- xgb.DMatrix(as.matrix(x0), label = y0)
  x1d <- xgb.DMatrix(as.matrix(x1), label = y1)
  watch <- list(valid = x1d)
  clf <- xgb.train(booster = "gbtree",
                   maximize = TRUE, 
                   print.every.n = 50,
                   nrounds = 250, eta = 0.01,
                   max.depth = 15,  colsample_bytree = 0.85,
                   subsample = 0.8,
                   data = x0d, objective = "binary:logistic",
                   watchlist = watch,  eval_metric = "auc",
                   gamma= 0.05)
  
  prx2 <- predict(clf, x1d)
  storage_matrix[ii,2] <- auc(y1,prx2)
  xvalid2[isValid,2] <- prx2
  
  # mix with nnet 
  net0 <- nnet(factor(y0) ~ ., data = x0, size = 10, MaxNWts = 10000)
  prx3 <- predict(net0, x1)
  storage_matrix[ii,3] <- auc(y1,prx3)
  xvalid2[isValid,3] <- prx3
  
  # mix with hillclimbing
  par0 <- buildEnsemble(c(1,8,5,0.3), x0,y0)
  prx4 <- as.matrix(x1) %*% as.matrix(par0)
  storage_matrix[ii,4] <- auc(y1,prx4)
  xvalid2[isValid,4] <- prx4
  
  # mix with random forest
  rf0 <- ranger(factor(y0) ~ ., data = x0, 
         mtry = 25, num.trees = 250,
         write.forest = T, probability = T,
         min.node.size = 10, seed = seed_value
  )
  prx5 <- predict(rf0, x1)$predictions[,2]
  xvalid2[isValid,5] <- prx5
  
  msg(ii)
}

## build prediction on full set
# glmnet
prx1 <- rep( 0, nrow(xfull))
for (jj in 1:11)
{
  mod0 <- glmnet(x = as.matrix(xvalid), y = y, alpha = (jj-1) * 0.1)
  prx <- predict(mod0,as.matrix(xfull))  
  prx <- prx[,ncol(prx)]
  # storage_matrix[ii,jj] <- auc(y1,prx1)
  prx1 <- prx1 + prx
}
prx1 <- rank(prx1)/length(prx1)
xfull2[,1] <- prx1

# xgboost
x0d <- xgb.DMatrix(as.matrix(xvalid), label = y)
x1d <- xgb.DMatrix(as.matrix(xfull))
clf <- xgb.train(booster = "gbtree",
                 maximize = TRUE, 
                 print.every.n = 50,
                 #early.stop.round = 25,
                 nrounds = 250,  eta = 0.01,
                 max.depth = 15, colsample_bytree = 0.85,
                 subsample = 0.8, data = x0d, 
                 objective = "binary:logistic",
                 # watchlist = watch, 
                 eval_metric = "auc",gamma= 0.05)

prx2 <- predict(clf, x1d)
prx2 <- rank(prx2)/length(prx2)
xfull2[,2] <- prx2


# mix with nnet 
net0 <- nnet(factor(y) ~ ., data = xvalid, size = 10, MaxNWts = 10000)
prx3 <- predict(net0, xfull)
xfull2[,3] <- prx3

# mix with hillclimbing
par0 <- buildEnsemble(c(1,8,5,0.3), xvalid,y)
prx4 <- as.matrix(xfull) %*% as.matrix(par0)
xfull2[,4] <- prx4

# mix with ranger
rf0 <- ranger(factor(y) ~ ., data = xvalid, 
              mtry = 25, num.trees = 250,
              write.forest = T, probability = T,
              min.node.size = 10, seed = seed_value
)
prx5 <- predict(rf0, xfull)$predictions[,2]
xfull2[,5] <- prx5

## store the final ensemble forecasts ####
# SFSG # 

# find the best combination of mixers

# evaluate performance across folds

# construct forecast
# xfor <- data.frame(QuoteNumber = id_full, QuoteConversion_Flag = 0.5 * (prx1 + prx2))

# store
# write_csv(xfor, path = paste("./submissions/ens_",todate,".csv", sep = ""))