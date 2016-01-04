## wd etc ####
require(readr)
require(stringr)
require(caret)
require(gbm)

dataset_version <- "kb5"
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

## data ####
# read actual data
xtrain <- read_csv(paste("./input/xtrain_",dataset_version,".csv", sep = ""))
xtest <- read_csv(paste("./input/xtest_",dataset_version,".csv", sep = ""))
y <- xtrain$QuoteConversion_Flag; xtrain$QuoteConversion_Flag <- NULL
id_train <- xtrain$QuoteNumber
id_test <- xtest$QuoteNumber
xtrain$QuoteNumber <- xtest$QuoteNumber <- NULL


xfolds <- createDataPartition(y, 
                    times = 20,
                    p = 0.1,
                    list = TRUE,
                    groups = min(5, length(y)))

## importance ####
relev_mat <- array(0, c(ncol(xtrain), length(xfolds)))
for (ii in 1:length(xfolds))
{
  idx <- xfolds[[ii]]
  x0 <- xtrain[-idx,]; y0 <- y[-idx]
  x1 <- xtrain[idx,]; y1 <- y[idx]
  
  mod0 <- gbm.fit(x= x1, y = y1, interaction.depth = 20, shrinkage = 0.01, n.trees = 200,
                  distribution = "bernoulli", verbose = T)
  relev_mat[,ii] <- summary(mod0, order= F)[,2]
}
# store
write_csv(data.frame(relev_mat), path = paste("./input/selection_", dataset_version, ".csv", sep = ""))


# more than 0.005 relevance
idx <- which(rowMeans(relev_mat) > 0.005)
xtrain1 <- xtrain[,idx]
xtest1 <- xtest[,idx]

xtrain1$QuoteNumber <- id_train
xtest1$QuoteNumber <- id_test
xtrain1$QuoteConversion_Flag <- y

write_csv(xtrain1, path = paste("./input/xtrain_kb5r1.csv",sep = "" ))
write_csv(xtest1, path = paste("./input/xtest_kb5r1.csv",sep = "" ))

# more than 0.01 relevance
idx <- which(rowMeans(relev_mat) > 0.01)
xtrain1 <- xtrain[,idx]
xtest1 <- xtest[,idx]

xtrain1$QuoteNumber <- id_train
xtest1$QuoteNumber <- id_test
xtrain1$QuoteConversion_Flag <- y

write_csv(xtrain1, path = paste("./input/xtrain_kb5r2.csv",sep = "" ))
write_csv(xtest1, path = paste("./input/xtest_kb5r2.csv",sep = "" ))

# more than 0.05 relevance
idx <- which(rowMeans(relev_mat) > 0.05)
xtrain1 <- xtrain[,idx]
xtest1 <- xtest[,idx]

xtrain1$QuoteNumber <- id_train
xtest1$QuoteNumber <- id_test
xtrain1$QuoteConversion_Flag <- y

write_csv(xtrain1, path = paste("./input/xtrain_kb5r3.csv",sep = "" ))
write_csv(xtest1, path = paste("./input/xtest_kb5r3.csv",sep = "" ))

