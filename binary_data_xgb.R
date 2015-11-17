# Lets try FTRL proximal on the data.. very niave approach.
library(data.table)
library(FeatureHashing)
library(xgboost)

train_full <- fread('input/xtrain_full.csv')
train <- fread('input/xtrain.csv')
valid <- fread('input/xvalid.csv')
test <- fread('input/xtest.csv')


train_full[, QuoteNumber := NULL]
train[, QuoteNumber := NULL]
valid[, QuoteNumber := NULL]
test[, QuoteNumber := NULL]

# Carve out things like log dates and salesField 28 and 'join' to the 
# hashed model matrix to have a combination of binaries and floats.
numeric_cols <- c('Field8',
                  'Field9', 
                  'Field11', 
                  'SalesField8',
                  'daysDurOrigQuote', 
                  'logDaysDurOrigQuote',
                  'logdaysDurOrigQuote2', 
                  'logdaysDurOrigQuote3')

numeric_mat_train_full <- train_full[, numeric_cols, with = F]
numeric_mat_train <- train[, numeric_cols, with = F]
numeric_mat_valid <- valid[, numeric_cols, with = F]
numeric_mat_test <- test[, numeric_cols, with = F]

# Remove the numeric cols from the data to be hashed.
train_full <- train_full[, -numeric_cols, with = F]
train <- train[, -numeric_cols, with = F]
valid <- valid[, -numeric_cols, with = F]
test <- test[, -numeric_cols, with = F]

y_train_full <- train_full[, QuoteConversion_Flag]
y_train <- train[, QuoteConversion_Flag]
y_valid <- valid[, QuoteConversion_Flag]

train_full[, QuoteConversion_Flag := NULL]
train[, QuoteConversion_Flag := NULL]
valid[, QuoteConversion_Flag := NULL]

# OK now lets cast into hashed map.
hash_full <- hashed.model.matrix(~., train_full, 2^16, create.mapping = TRUE)
# Rate of collision = 0 (i.e no collisions on 2^16)
mapping1 <- hash.mapping(hash_full)
mean(duplicated(mapping1))
hash_train <- hashed.model.matrix(~., train, 2^16)
hash_valid <- hashed.model.matrix(~., valid, 2^16)
hash_test <- hashed.model.matrix(~., test, 2^16)

# Should export data here -> useful for NN as wellas bins

dtrain <- xgb.DMatrix(hash_train, label = y_train)
dvalid <- xgb.DMatrix(hash_valid, label = y_valid)
watch <- list(valid = dvalid, train = dtrain)

m1 <- xgb.train(booster = "gbtree",
                maximize = TRUE, 
                print.every.n = 25,
                early.stop.round = 50,
                nrounds = 2000,
                eta = 0.028,
                max.depth = 50, 
                colsample_bytree = 0.9,
                subsample = 0.85,
                data = dtrain, 
                objective = "binary:logistic",
                watchlist = watch, 
                eval_metric = "auc")


