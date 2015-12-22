## wd etc ####
require(readr)
require(h2o)
require(stringr)

h2oServer <- h2o.init(nthreads=-1, max_mem_size = "14g")

dataset_version <- "kb4"
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

## extract deep features ####

xtrain$target <- factor(y)

size1 <- 400; size2 <- 200
xrange <- 1:(ncol(xtrain) - 1)

xtrain1 <- data.frame(array(0, c(nrow(xtrain), size1)))
xtest1 <- data.frame(array(0, c(nrow(xtest), size1)))
xtrain2 <- data.frame(array(0, c(nrow(xtrain), size2)))
xtest2 <- data.frame(array(0, c(nrow(xtest), size2)))

# loop over folds 
for (ii in 1:nfolds)
{
  isTrain <- which(xfolds$fold_index != ii)
  isValid <- which(xfolds$fold_index == ii)
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
    rate_annealing = 1e-06, rate_decay = 1, momentum_start = 0.5,
    l1 = 0, l2 = 0,  loss = c("CrossEntropy")
  )
  
  # extract layer 1 and 2 
  xl1 <- as.data.frame(h2o.deepfeatures(dl.model, x1.hex, layer = 1))
  xl2 <- as.data.frame(h2o.deepfeatures(dl.model, x1.hex, layer = 2))
  
  xtrain1[isValid,] <- xl1
  xtrain2[isValid,] <- xl2
  
  msg(ii)  
}

# full dataset
xtrain.hex <- as.h2o(xtrain); xtest.hex <- as.h2o(xtest)
dl.model <- h2o.deeplearning(
  # data specifications
  x = xrange, y = max(xrange)+1, training_frame = xtrain.hex, 
  autoencoder = FALSE, 
  # network structure: activation and geometry
  activation = "RectifierWithDropout",
  hidden = c(size1, size2), epochs = 25, 
  input_dropout_ratio = 0.05, hidden_dropout_ratios = c(0.05, 0.02), 
  # parameters of the optimization process
  rho = 0.99, epsilon = 1e-08, rate = 0.005,
  rate_annealing = 1e-06, rate_decay = 1, momentum_start = 0.5,
  l1 = 0, l2 = 0,  loss = c("CrossEntropy")
)

# extract layer 1 and 2 
xtest1 <- h2o.deepfeatures(dl.model, xtest.hex, layer = 1)
xtest1 <- as.data.frame(xtest1)
xtest2 <- as.data.frame(h2o.deepfeatures(dl.model, xtest.hex, layer = 2))


## store complete versions ####
xtrain1$QuoteConversion_Flag <- y
xtrain2$QuoteConversion_Flag <- y
xtrain1$QuoteNumber <- id_train
xtrain2$QuoteNumber <- id_train
xtest1$QuoteNumber <- id_test
xtest2$QuoteNumber <- id_test

write_csv(xtrain1, "./input/xtrain_kb8.csv")
write_csv(xtest1, "./input/xtest_kb8.csv")
write_csv(xtrain2, "./input/xtrain_kb9.csv")
write_csv(xtest2, "./input/xtest_kb9.csv")


