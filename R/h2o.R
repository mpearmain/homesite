## wd etc ####
require(readr)
require(xgboost)
require(h2o)

h2oServer <- h2o.init(nthreads=-1, max_mem_size = "12g")

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
y <- xtrain$QuoteConversion_Flag; xtrain$QuoteConversion_Flag <- NULL
xtrain$QuoteNumber <- xtest$QuoteNumber <- NULL
xtrain$SalesField8 <- xtest$SalesField8 <- NULL

## fit an h2o model ####
xfolds <- read_csv("./input/xfolds.csv")
isValid <- which(xfolds$valid == 1); isTrain <- which(xfolds$valid == 0)

xtrain$target <- factor(y)

xvalid <- xtrain[isValid,]; xtrain <- xtrain[isTrain,]

size1 <- ncol(xtrain)-1; size2 <- round(0.5 * size1)
xrange <- 1:size1

train.hex <- as.h2o(xtrain); valid.hex <- as.h2o(xvalid)
xseed <- 123; test.hex <- as.h2o(xtest)

rm(xtrain, xtest, xvalid)

## generate prediction ####
# build a dl.model using rectifierWithDropout activation
dl.model <- h2o.deeplearning(
  # data specifications
  x = xrange, y = max(xrange)+1, training_frame = train.hex, 
  autoencoder = FALSE, 
  # network structure: activation and geometry
  activation = "RectifierWithDropout",
  hidden = c(size1, size2), epochs = 20, 
  input_dropout_ratio = 0.05, hidden_dropout_ratios = c(0.05, 0.02), 
  # parameters of the optimization process
  rho = 0.99, epsilon = 1e-08, rate = 0.005,
  rate_annealing = 1e-06, rate_decay = 1, momentum_start = 0.5,
  
  l1 = 0, l2 = 0,  loss = c("CrossEntropy")
)

pred_valid_h2o_v2 <- as.data.frame(predict(dl.model, valid.hex))$p1
pred_full_h2o_v2 <- as.data.frame(predict(dl.model, test.hex))$p1

pred1 <- pred_full
x <- read_csv("./input/xtest_kb4.csv")
xsub <- data.frame(QuoteNumber = x$QuoteNumber, QuoteConversion_Flag = pred1)
xsub$QuoteConversion_Flag <- rank(pred1)/nrow(xsub)
write_csv(xsub, "./submissions/h2o_kb5_v1.csv")

