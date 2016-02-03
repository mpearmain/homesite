library(xgboost)
library(data.table)
library(caret)

#Load dataset
load("dataset.RData")

# add the NA / zeros count
nb_NA <- apply(subset(dataset, select = - c(QuoteConversion_Flag, type)), 1, function(x) sum(x == -1))
nb_0 <- apply(subset(dataset, select = - c(QuoteConversion_Flag, type)), 1, function(x) sum(x == 0))
dataset <- cbind(dataset, nb_NA = nb_NA, nb_0 = nb_0)
rm(nb_NA, nb_0)
gc()

#replace missing values by -1
dataset[is.na(dataset)] <- -1
dataset <- subset(dataset, select = -c(QuoteNumber, PropertyField6, GeographicField10A, Original_Quote_Date))

y <- dataset$QuoteConversion_Flag[dataset$type == 1 ]

##create training set and validation set
set.seed(55555)
ind_training <- createFolds(y = y, k = 5)

#create the training dataset
training <- dataset[Reduce(c, ind_training[1:4]),]
y_training <- y[Reduce(c, ind_training[1:4])]

#remove useless features
training <- subset(training, select = -c(QuoteConversion_Flag, type))
training <- data.matrix(training)

indices <- combn(x = 300, 2)
res <- matrix(0, nrow = nrow(training), ncol = dim(indices)[2] + 300)
for (i in 1:dim(indices)[2]) {
  print(i)
  res[,i] <- training[,indices[2,i]] / training[,indices[1,i]]
}

col.names <- matrix(colnames(training)[indices], ncol = 2, byrow = TRUE)
col.names <- apply(col.names, 1, function(x) paste0(x, collapse = "("))
col.names <- c(col.names, colnames(training))
res[,44851:45150] <- training
colnames(res) <- col.names

res[!is.finite(res)] <- 0

parametres <- list(objective = "binary:logistic", booster = "gbtree", eval_metric = "auc",
                   eta              = 0.01,
                   max_depth        = 6, 
                   subsample        = 0.5,
                   colsample_bytree = 0.05,
                   nthread = 32)

model_interacdiv2 <- xgboost(data = res, label = y_training, params = parametres,
                            nrounds = 5000, verbose = 1, print.every.n = 10L)
featImp_interacdiv2 <- xgb.importance(feature_names = colnames(res), model = model_interacdiv2)

save(featImp_interacdiv2, file = "./Dropbox/featImp_interacdiv2.RData")
