# Attempt at glmnet ensemble for multiple predictions made from classify_bayes.py
library(data.table)
library(glmnet)
library(Metrics)
library(caret)
library(e1071)
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
## Model. Use Caret to find best alpha and lambda
# First remove duplicate cols that come from LR and DTrees.
#keep.cols <- valid.data[,colnames(unique(as.matrix(valid.data), MARGIN=2))]
#valid.data <- valid.data[, valid.data[,colnames(unique(as.matrix(valid.data), MARGIN=2))], with = F]

eGrid <- expand.grid(.alpha = (1:10) * 0.1, 
                     .lambda = (1:10) * 0.1)
Control <- trainControl(method = "repeatedcv", 
                        number = 10,
                        repeats = 10,
                        verboseIter =TRUE,
                        classProbs = TRUE,
                        summaryFunction=twoClassSummary)

netFit <- train(x = as.matrix(valid.data), 
                y = as.factor(make.names(y)),
                method = "glmnet",
                tuneGrid = eGrid,
                trControl = Control,
                family = "binomial",
                metric = "ROC")

auc(y, predict(netFit, valid.data, type = "prob"))


submission <- predict(netFit, test.data, type = "prob")
submission <- as.data.table(list("QuoteNumber" = test.QuoteNumber, 
                                 'QuoteConversion_Flag' = submission))

write.csv(submission, 'output/glmnet_test.csv', row.names = F)





