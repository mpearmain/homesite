# Attempt at glmnet ensemble for multiple predictions made from classify_bayes.py
library(data.table)
library(glmnet)
library(Metrics)

# Load the train dataset -> forgot to add the QuoteConversionFlag to output :-S
# Lets merge.
train <- fread('input/train.csv', select = c('QuoteNumber', 'QuoteConversion_Flag'))

# Lets grab all the valid and test files and join them.
valid.data <- lapply(list.files(path='./submission', 
                                pattern = "^pred[Valid]",
                                full.names = T),
                     fread)
# Merge all the dataset together to ensemble.
valid.data <- Reduce(function(x, y) merge(x, y, by='QuoteNumber'), valid.data)
valid.data <- merge(valid.data, train, by='QuoteNumber')
valid.data <- valid.data[, QuoteNumber := NULL]
y = as.factor(valid.data[, QuoteConversion_Flag])
valid.data[, QuoteConversion_Flag := NULL]

### Load and build test data.
# Lets grab all the valid and test files and join them.
test.data <- lapply(list.files(path='./submission', 
                                pattern = "^pred[Test]",
                                full.names = T),
                     fread)
# Merge all the dataset together to ensemble.
test.data <- Reduce(function(x, y) merge(x, y, by='QuoteNumber'), test.data)
test.QuoteNumber <- test.data[, QuoteNumber]
test.data[, QuoteNumber := NULL]
######################################################################
######################################################################
## Model 
cvfit <- cv.glmnet(as.matrix(valid.data), y, family = "binomial", type.measure = "auc")
submission <- predict(cvfit, newx = as.matrix(test.data), s='lambda.min', type="response")

submission <- as.data.table(list("QuoteNumber" = test.QuoteNumber, 
                                 'QuoteConversion_Flag' = submission))

write.csv(submission, 'output/glmnet_test.csv', row.names = F)





