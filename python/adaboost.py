__author__ = 'michael.pearmain'

import pandas as pd
import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score as auc


print('Loading Full Train data set')
x_train_full = pd.read_csv('input/xtrain_full.csv')
print('Loading Train-valid data set')
x_train = pd.read_csv('input/xtrain.csv')
print('Loading Valid data set')
x_valid = pd.read_csv('input/xvalid.csv')

print('Loading Test data set')
test = pd.read_csv('input/xtest.csv')

sample = pd.read_csv('input/sample_submission.csv')

y_train_full = x_train_full.QuoteConversion_Flag.values
x_train_full = x_train_full.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)

y_train = x_train.QuoteConversion_Flag.values
x_train = x_train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)

y_valid = x_valid.QuoteConversion_Flag.values
x_valid = x_valid.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)

test = test.drop('QuoteNumber', axis=1)

x_train_full = x_train_full.fillna(-1)
x_train = x_train.fillna(-1)
x_valid = x_valid.fillna(-1)
test = test.fillna(-1)

# Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8),
                         algorithm="SAMME",
                         n_estimators=200)

bdt.fit(x_train, y_train)
#preds = bdt.predict_proba(test)[:,1]
pred_valid = bdt.predict_proba(x_valid)[:,1]
print 'AUC full features = ', auc(y_valid, pred_valid)

sample.QuoteConversion_Flag = preds
file_name = 'adaboost_trees_' + time.strftime("%d_%m_%Y") + '_mp.csv'
sample.to_csv('output/'+file_name, index=False)

