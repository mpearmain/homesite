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
xtrain <- read_csv("./input/xtrain_kb4.csv")
xtest <- read_csv("./input/xtest_kb4.csv")

## create kmeans-based dataset ####
xfolds <- read_csv("./input/xfolds.csv")
isValid <- which(xfolds$valid == 1)
y <- xtrain$QuoteConversion_Flag; xtrain$QuoteConversion_Flag <- NULL
xtrain$QuoteNumber <- xtest$QuoteNumber <- NULL
xtrain$SalesField8 <- xtest$SalesField8 <- NULL

# map to distances from kmeans clusters
nof_centers <- 40
km0 <- kmeans(xtrain, centers = nof_centers)
dist1 <- array(0, c(nrow(xtrain), nof_centers))
for (ii in 1:nof_centers)
{
  dist1[,ii] <- apply(xtrain,1,function(s) sd(s - km0$centers[ii,]))
  msg(ii)
}
dist2 <- array(0, c(nrow(xtest), nof_centers))
for (ii in 1:nof_centers)
{
  dist2[,ii] <- apply(xtest,1,function(s) sd(s - km0$centers[ii,]))
  msg(ii)
}

## model fitting ####
# load split
xfolds <- read_csv("./input/xfolds.csv")
dist1 <- data.frame(dist1)
dist2 <- data.frame(dist2)
dist1$target <- factor(y)



