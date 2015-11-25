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
xtrain <- read_csv("./input/xtrain_kb4.csv")
xtest <- read_csv("./input/xtest_kb4.csv")

## fit model ####
# separate into training and validation
xfolds <- read_csv("./input/xfolds.csv")
isValid <- which(xfolds$valid == 1)
y <- xtrain$QuoteConversion_Flag; xtrain$QuoteConversion_Flag <- NULL
xtrain$QuoteNumber <- xtest$QuoteNumber <- NULL
xtrain$SalesField8 <- xtest$SalesField8 <- NULL

# map to distances from kmeans clusters
nof_centers <- 20
km0 <- kmeans(xtrain, centers = nof_centers)
dist_matrix <- array(0, c(nrow(xtrain), nof_centers))
for (ii in 1:nof_centers)
{
  dist_matrix[,ii] <- apply(xtrain,1,function(s) sd(s - km0$centers[ii,]))
  msg(ii)
}


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

# re-fit on complete data


## generate prediction ####
pred1 <- expm1(predict(clf, data.matrix(xtest)))
xsub <- data.frame(QuoteNumber = xtest$QuoteNumber, QuoteConversion_Flag = pred1)
xsub$QuoteConversion_Flag <- rank(pred1)/nrow(xsub)
write_csv(xsub, "./submissions/btb_data_kb4.csv")

