## wd etc ####
require(Metrics)
require(caret)
require(readr)
require(stringr)


## extra functions ####
# print a formatted message
msg <- function(mmm,...)
{
  cat(sprintf(paste0("[%s] ",mmm),Sys.time(),...)); cat("\n")
}

## create folds ####
# load data

xtrain <- read_csv(file = "./input/train.csv")
id_train <- xtrain$QuoteNumber; xtrain$QuoteNumber <- NULL
y <- xtrain$QuoteConversion_Flag; xtrain$QuoteConversion_Flag <- NULL


set.seed(20150817)
idFix5 <- createFolds(y, k = 5, list = T)
idFix10 <- createFolds(y, k = 10, list = T)
val_size <- 14000
subrange <- sample(nrow(xtrain), size = val_size)

# populate the different indices into columns
xfolds <- array(0, c(length(y), 4))
# column 1  = train id
xfolds[,1] <- id_train
# column 2 = 5-fold split, where each row gets the {1,2,3,4,5} id assigned
# this is R counting convention => starts from 1!
for (ii in seq(idFix5)) {
  xfolds[ idFix5[[ii]],2] <- ii
}
# column 3 = 10-fold split, where each row gets the {1,...,10} id assigned
# this is R counting convention => starts from 1!
for (ii in seq(idFix10)) {
  xfolds[ idFix10[[ii]],3] <- ii
}
# column 4 = train/valid split (0 = train, 1 = valid)
xfolds[subrange,4] <- 1

# store 
xfolds <- data.frame(xfolds)
colnames(xfolds) <- c("QuoteNumber", "fold5", "fold10", "valid")
write_csv(xfolds, path = "./input/xfolds.csv")
