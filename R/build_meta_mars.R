## wd etc ####
require(readr)
require(earth)
require(stringr)

dataset_version <- "kb4"
seed_value <- 1901
model_type <- "mars"
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
# read actual data
xtrain <- read_csv(paste("./input/xtrain_",dataset_version,".csv", sep = ""))
xtest <- read_csv(paste("./input/xtest_",dataset_version,".csv", sep = ""))
y <- xtrain$QuoteConversion_Flag; xtrain$QuoteConversion_Flag <- NULL
id_train <- xtrain$QuoteNumber
id_test <- xtest$QuoteNumber
xtrain$QuoteNumber <- xtest$QuoteNumber <- NULL
xtrain$SalesField8 <- xtest$SalesField8 <- NULL

# division into folds: 5-fold
xfolds <- read_csv("./input/xfolds.csv"); xfolds$fold_index <- xfolds$fold5
xfolds <- xfolds[,c("QuoteNumber", "fold_index")]
nfolds <- length(unique(xfolds$fold_index))

# SFSG # 

## fit models ####
# parameter grid
param_grid <- expand.grid(deg = c(2,3,4))

# storage structures 
mtrain <- array(0, c(nrow(xtrain), nrow(param_grid)))
mtest <- array(0, c(nrow(xtest), nrow(param_grid)))

# loop over parameters
for (ii in 1:nrow(param_grid))
{
  
  # loop over folds 
  for (jj in 1:nfolds)
  {
    isTrain <- which(xfolds$fold_index != jj)
    isValid <- which(xfolds$fold_index == jj)
    x0 <- xtrain[isTrain,]; x1 <- xtrain[isValid,]
    y0 <- factor(y)[isTrain]; y1 <- factor(y)[isValid]
    
    mars.model <- earth(x = x0, y = y0, degree = param_grid$deg[ii], glm=list(family=binomial))
    
    pred_valid <- predict(mars.model, x1, type = "response")
    mtrain[isValid,ii] <- pred_valid
  }
  
  # full version 
  mars.model <- earth(x = x0, y = y0, degree = param_grid$deg[ii], glm=list(family=binomial))
  
  pred_full <- predict(mars.model, xtest, type = "response")
  mtest[,ii] <- pred_full
  msg(ii)
}

## store complete versions ####
mtrain <- data.frame(mtrain)
mtest <- data.frame(mtest)
colnames(mtrain) <- colnames(mtest) <- paste(model_type, 1:ncol(mtrain), sep = "")
mtrain$QuoteNumber <- id_train
mtest$QuoteNumber <- id_test
mtrain$QuoteConversion_Flag <- y

write_csv(mtrain, path = paste("./metafeatures/prval_",model_type,"_", todate, "_data", dataset_version, "_seed", seed_value, ".csv",sep = "" ))
write_csv(mtest, path = paste("./metafeatures/prfull_",model_type,"_", todate, "_data", dataset_version, "_seed", seed_value, ".csv",sep = "" ))


