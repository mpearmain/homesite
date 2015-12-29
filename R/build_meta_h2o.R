## wd etc ####
require(readr)
require(h2o)
require(stringr)

h2oServer <- h2o.init(nthreads=-1, max_mem_size = "14g")

dataset_version <- "kb5"
seed_value <- 1234
model_type <- "h2o"
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
param_grid <- expand.grid(size1 = c(400, 200),
                          size2 = c(200, 100),
                          rate_dec = c(1))

xtrain$target <- factor(y)

# storage structures 
mtrain <- array(0, c(nrow(xtrain), nrow(param_grid)))
mtest <- array(0, c(nrow(xtest), nrow(param_grid)))
xrange <- 1:(ncol(xtrain) - 1)

# loop over parameters
for (ii in 1:nrow(param_grid))
{
  size1 <- param_grid$size1[ii]
  size2 <- param_grid$size2[ii]
  rate_dec <- param_grid$rate_dec[ii]
  
  # loop over folds 
  for (jj in 1:nfolds)
  {
    isTrain <- which(xfolds$fold_index != jj)
    isValid <- which(xfolds$fold_index == jj)
    x0 <- xtrain[isTrain,]; x1 <- xtrain[isValid,-297]
    # convert to H2O format
    x0.hex <- as.h2o(x0); x1.hex <- as.h2o(x1)
    xseed <- seed_value
    
    
    dl.model <- h2o.deeplearning(
      # data specifications
      x = xrange, y = max(xrange)+1, training_frame = x0.hex, 
      autoencoder = FALSE, 
      # network structure: activation and geometry
      activation = "RectifierWithDropout",
      hidden = c(size1, size2), epochs = 25, 
      input_dropout_ratio = 0.05, hidden_dropout_ratios = c(0.05, 0.02), 
      # parameters of the optimization process
      rho = 0.99, epsilon = 1e-08, rate = 0.005,
      rate_annealing = 1e-06, rate_decay = rate_dec, momentum_start = 0.5,
      l1 = 0, l2 = 0,  loss = c("CrossEntropy")
    )
    
    pred_valid <- as.data.frame(predict(dl.model, x1.hex))$p1
    mtrain[isValid,ii] <- pred_valid
  }
 
  # full version 
  x0.hex <- as.h2o(xtrain); x1.hex <- as.h2o(xtest)
  xseed <- seed_value
  
  dl.model <- h2o.deeplearning(
    # data specifications
    x = xrange, y = max(xrange)+1, training_frame = x0.hex, 
    autoencoder = FALSE, 
    # network structure: activation and geometry
    activation = "RectifierWithDropout",
    hidden = c(size1, size2), epochs = 25, 
    input_dropout_ratio = 0.05, hidden_dropout_ratios = c(0.05, 0.02), 
    # parameters of the optimization process
    rho = 0.99, epsilon = 1e-08, rate = 0.005,
    rate_annealing = 1e-06, rate_decay = rate_dec, momentum_start = 0.5,
    l1 = 0, l2 = 0,  loss = c("CrossEntropy")
  )
  
  pred_full <- as.data.frame(predict(dl.model, x1.hex))$p1
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

