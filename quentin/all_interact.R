# All Interactions
install.packages("xgboost")
install.packages("data.table")
install.packages("caret")
install.packages("stringr")

library("xgboost")
library("data.table")
library("caret")
library("stringr")


#Load dataset
load("dataset.RData")
load("names_feat_interac.RData")
load("featImp_interacadd.RData")
load("featImp_interacdiv.RData")
load("featImp_interacdiv2.RData")
load("featImp_interacsub.RData")
load("featImp_interacsub2.RData")


# add the NA / zeros count
nb_NA <- apply(subset(dataset, select = - c(QuoteConversion_Flag, type)), 1, function(x) sum(x == -1))
nb_0 <- apply(subset(dataset, select = - c(QuoteConversion_Flag, type)), 1, function(x) sum(x == 0))
dataset <- cbind(dataset, nb_NA = nb_NA, nb_0 = nb_0)
rm(nb_NA, nb_0)
gc()

#replace missing values by -1
dataset[is.na(dataset)] <- -1
dataset <- subset(dataset, select = -c(QuoteNumber, PropertyField6, GeographicField10A, Original_Quote_Date))

dataset2 <- data.matrix(dataset)

# Compute quadratic interactions
features <- names_feat_interac[2:301,]
interac_feat <- matrix(0, ncol = nrow(features), nrow = nrow(dataset))
for (i in 1:nrow(features)) {
  tmp <- dataset2[, as.character(features[i, 1])] * dataset2[, as.character(features[i, 2])]
  interac_feat[, i] <- as.matrix(tmp)
}
colnames(interac_feat) <- paste(features[,1], features[,2], sep = ":")

# Compute subtraction interactions
sub_list <- strsplit(featImp_interacsub$Feature[1:500], split = "-")
sub_feat <- Reduce(rbind, sub_list[lapply(sub_list, length) == 2])
sub_feat <- sub_feat[1:300,]
interac_feat_sous <- matrix(0, ncol = nrow(sub_feat), nrow = nrow(dataset2))
for (i in 1:nrow(sub_feat)) {
  tmp <- dataset2[, as.character(sub_feat[i, 1])] - dataset2[, as.character(sub_feat[i, 2])]
  interac_feat_sous[, i] <- as.matrix(tmp)
}
colnames(interac_feat_sous) <- paste(sub_feat[,1], sub_feat[,2], sep = "-")

# Compute division interactions
div_list <- strsplit(featImp_interacdiv$Feature[1:500], split = "/")
div_feat <- Reduce(rbind, div_list[lapply(div_list, length) == 2])
div_feat <- div_feat[1:300,]

interac_feat_div <- matrix(0, ncol = nrow(div_feat), nrow = nrow(dataset2))
for (i in 1:nrow(div_feat)) {
  tmp <- dataset2[, as.character(div_feat[i, 1])] / dataset2[, as.character(div_feat[i, 2])]
  interac_feat_div[, i] <- as.matrix(tmp)
}
interac_feat_div[!is.finite(interac_feat_div)] <- 0
colnames(interac_feat_div) <- paste(div_feat[,1], div_feat[,2], sep = "/")

# Compute second way of division interactions
div2_list <- strsplit(featImp_interacdiv2$Feature[1:500], split = "\\(")
div2_feat <- Reduce(rbind, div2_list[lapply(div2_list, length) == 2])
div2_feat <- div2_feat[1:300,]

interac_feat_div2 <- matrix(0, ncol = nrow(div2_feat), nrow = nrow(dataset2))

for (i in 1:nrow(div2_feat)) {
  tmp <- dataset2[, as.character(div2_feat[i, 2])] / dataset2[, as.character(div2_feat[i, 1])] 
  interac_feat_div2[, i] <- as.matrix(tmp)
}
interac_feat_div2[!is.finite(interac_feat_div2)] <- 0
colnames(interac_feat_div2) <- paste(div2_feat[,1], div2_feat[,2], sep = "(")

# Compute second way of subtraction interactions
sub2_list <- strsplit(featImp_interacsub2$Feature[1:500], split = "\\(-")
sub2_feat <- Reduce(rbind, sub2_list[lapply(sub2_list, length) == 2])
sub2_feat <- sub2_feat[1:300,]

interac_feat_sous2 <- matrix(0, ncol = nrow(sub2_feat), nrow = nrow(dataset2))

for (i in 1:nrow(sub2_feat)) {
  tmp <- dataset2[, as.character(sub2_feat[i, 2])] - dataset2[, as.character(sub2_feat[i, 1])] 
  interac_feat_sous2[, i] <- as.matrix(tmp)
}

colnames(interac_feat_sous2) <- paste(sub2_feat[,1], sub2_feat[,2], sep = "(-")


# Compute addition interactions
add_list <- strsplit(featImp_interacadd$Feature[1:500], split = "\\+")
add_feat <- Reduce(rbind, add_list[lapply(add_list, length) == 2])
add_feat <- add_feat[1:300,]

interac_feat_add <- matrix(0, ncol = nrow(add_feat), nrow = nrow(dataset2))

for (i in 1:nrow(add_feat)) {
  tmp <- dataset2[, as.character(add_feat[i, 2])] - dataset2[, as.character(add_feat[i, 1])] 
  interac_feat_add[, i] <- as.matrix(tmp)
}
colnames(interac_feat_add) <- paste(add_feat[,1], add_feat[,2], sep = "+")


#######################################
## Create dataset with dummy features #
#######################################

## find nominal and ordinal variables
to_include <- colnames(dataset)[!(colnames(dataset) %in% c("Original_Quote_Date","PropertyField6", "GeographicField10A", "year","month", "weekday", "day", "type", "QuoteConversion_Flag", "QuoteNumber"))]
cols_ordinal <- to_include[sapply(to_include,function(x){str_sub(x, start= -1)}) %in% c("A","B")]
cols_ordinal <- c(cols_ordinal, "nb_0")
cols_nominal <- to_include [!to_include %in% cols_ordinal]

#create dummy variable for each nominal column
dataset_nominal_factor <- dataset[,cols_nominal, with = F]
dataset_nominal_factor <- dataset_nominal_factor[, lapply(.SD, as.factor)]
dataset_nominal_factor <- subset(dataset_nominal_factor, select = - SalesField8)
dim(dataset_nominal_factor)
dataset_dummy <- model.matrix(~.-1, data=dataset_nominal_factor) 
dim(dataset_dummy)

#remove cols_nominal from dataset
dataset_no_nominal <- dataset[,!(colnames(dataset) %in% cols_nominal), with = F]
dim(dataset_no_nominal)

#create a new dataset with no nominal feature but dummy variables
dataset_with_dummy <- cbind(dataset_no_nominal,dataset_dummy)
dim(dataset_with_dummy)

### end dummy

##create training set and validation set
set.seed(55555)
ind_training <- createFolds(y = y, k = 5)
test <- dataset_with_dummy[dataset_with_dummy$type == 0,]
#create the validation dataset
validation <- dataset_with_dummy[ind_training$Fold5, ]
y_validation <- y[ind_training$Fold5]

#create the training dataset
training <- dataset_with_dummy[Reduce(c, ind_training[1:4]),]
y_training <- y[Reduce(c, ind_training[1:4])]

#remove useless features
training <- subset(training, select = -c(QuoteConversion_Flag, type))
validation <- subset(validation, select = -c(QuoteConversion_Flag, type))
test <- subset(test, select = -c(QuoteConversion_Flag, type))


training <- cbind(training, 
                  interac_feat[Reduce(c, ind_training[1:4]),],
                  interac_feat_add[Reduce(c, ind_training[1:4]),],
                  interac_feat_div[Reduce(c, ind_training[1:4]),],
                  interac_feat_div2[Reduce(c, ind_training[1:4]),],
                  interac_feat_sous[Reduce(c, ind_training[1:4]),],
                  interac_feat_sous2[Reduce(c, ind_training[1:4]),])

validation <- cbind(validation, 
                    interac_feat[ind_training$Fold5,],
                    interac_feat_add[ind_training$Fold5,],
                    interac_feat_div[ind_training$Fold5,],
                    interac_feat_div2[ind_training$Fold5,],
                    interac_feat_sous[ind_training$Fold5,],
                    interac_feat_sous2[ind_training$Fold5,])

test <- cbind(test, 
              interac_feat[dataset$type == 0, ],
              interac_feat_add[dataset$type == 0, ],
              interac_feat_div[dataset$type == 0, ],
              interac_feat_div2[dataset$type == 0, ],
              interac_feat_sous[dataset$type == 0, ],
              interac_feat_sous2[dataset$type == 0, ])

dval <- xgb.DMatrix(data = data.matrix(validation), label = y_validation)
dtrain <- xgb.DMatrix(data = data.matrix(training), label = y_training)

watchlist <- list(val = dval, train = dtrain)

set.seed(559655)
parametres <- list(objective = "binary:logistic", booster = "gbtree", eval_metric = "auc",
                   eta              = 0.01,
                   max_depth        = 6, 
                   subsample        = 0.7,
                   colsample_bytree = 0.77,
                   nthread = 40)

clf <- xgb.train(params = parametres, data = dtrain, nrounds = 5000, verbose = 1, watchlist = watchlist, print.every.n = 10)

pred_val_allinterac <- predict(clf, dval)
save(pred_val_allinterac_with_models, file = "pred_val_allinterac.RData")

set.seed(559655)
dall <- xgb.DMatrix(data = data.matrix(rbind(training, validation)), label = c(y_training, y_validation))
clf_all <- xgboost(params = parametres, data = dall, nrounds = 5000, verbose = 1, print.every.n = 10)

dtest <- xgb.DMatrix(data = data.matrix(test))
pred_test_allinterac <- predict(clf_all, dtest)
save(pred_test_allinterac, file = "pred_test_allinterac_with_model_5000.RData")
