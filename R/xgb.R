## wd etc ####
require(readr)
require(xgboost)

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
# read
xtrain <- read_csv("./input/xtrain_kb5.csv")
xtest <- read_csv("./input/xtest_kb5.csv")

## fit model ####
# separate into training and validation
xfolds <- read_csv("./input/xfolds.csv")
isValid <- which(xfolds$valid == 1)
y <- xtrain$QuoteConversion_Flag; xtrain$QuoteConversion_Flag <- NULL
xtrain$QuoteNumber <- xtest$QuoteNumber <- NULL

# map to distances from kmeans clusters

# setup xgb
# xgboost format
dval<-xgb.DMatrix(data=data.matrix(xtrain[isValid, ]),label=y[isValid])
dtrain<-xgb.DMatrix(data=data.matrix(xtrain[-isValid,]),label=y[-isValid ])
# setup
watchlist<-list(val=dval)
param <- list(  objective           = "binary:logistic", 
                booster = "gbtree",
                eta                 = 0.05,
                max_depth           = 25, 
                subsample           = 0.9, 
                colsample_bytree    = 0.7,
                eval_metric = "auc",
                gamma = 0.005
)

# fit the model
clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 15000,
                    verbose             = 0,
                    early.stop.round    = 25,
                    watchlist           = watchlist,
                    maximize            = TRUE
                    )

pred_valid_xgb <- predict(clf, dval)
pred_full_xgb <- predict(clf, data.matrix(xtest))

## generate prediction ####
xtest <- read_csv("./input/xtest_kb1.csv")
xsub <- data.frame(QuoteNumber = xtest$QuoteNumber, QuoteConversion_Flag = pred_full_xgb)
xsub$QuoteConversion_Flag <- rank(pred1)/nrow(xsub)
write_csv(xsub, "./submissions/xgb_datakb5_v1.csv")

