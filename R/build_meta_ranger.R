## wd etc ####
require(readr)
require(ranger)
require(stringr)

dataset_version <- "kb3"
seed_value <- 132
model_type <- "ranger"
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

## fit dl models ####
# parameter grid
param_grid <- expand.grid(ntree = c(500, 1250),
                          mtry = c(10,15,25),
                          nsize = c(1,5))

# storage structures 
mtrain <- array(0, c(nrow(xtrain), nrow(param_grid)))
mtest <- array(0, c(nrow(xtest), nrow(param_grid)))
xrange <- 1:(ncol(xtrain) - 1)

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
    
    ranger.model <- ranger(y0 ~ ., data = x0, 
                           mtry = param_grid$mtry[ii],
                           num.trees = param_grid$ntree[ii],
                           write.forest = T,
                           probability = T,
                           min.node.size = param_grid$nsize[ii],
                           seed = seed_value
                           )
    
    
    pred_valid <- predict(ranger.model, x1)$predictions[,2]
    mtrain[isValid,ii] <- pred_valid
  }
  
  # full version 
  ranger.model <- ranger(y0 ~ ., data = x0, 
                         mtry = param_grid$mtry[ii],
                         num.trees = param_grid$ntree[ii],
                         write.forest = T,
                         probability = T,
                         min.node.size = param_grid$nsize[ii],
                         seed = seed_value
  )
  
  pred_full <- predict(ranger.model, xtest)$predictions[,2]
  mtest[,ii] <- pred_full
  
  
}

## store complete versions ####
mtrain <- data.frame(mtrain)
mtest <- data.frame(mtest)
mtrain$QuoteNumber <- id_train
mtest$QuoteNumber <- id_test
mtrain$QuoteConversion_Flag <- y

write_csv(mtrain, path = paste("./metafeatures/prval_",model_type,"_", todate, "_data", dataset_version, "_seed", seed_value, ".csv",sep = "" ))
write_csv(mtest, path = paste("./metafeatures/prfull_",model_type,"_", todate, "_data", dataset_version, "_seed", seed_value, ".csv",sep = "" ))

