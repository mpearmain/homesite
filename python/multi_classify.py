__author__ = 'michael.pearmain'

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score as auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# Got a little bored so decided to just build a loop over a whole bunch of classifiers
# add them back and then through in xgb for final ensemble.

# Read the data files in - Train, test, and xfolds.
print('Loading Full Train data set')
x_train = pd.read_csv('input/xtrain_mp2.csv')
print('Loading X-folds data set')
x_folds = pd.read_csv('input/xfolds.csv')
print('Loading Test data set')
test = pd.read_csv('input/xtest_mp2.csv')
sample = pd.read_csv('input/sample_submission.csv')

# Make sure final set has no na's (should be solved in data_preparation.R)
x_train = x_train.fillna(-1)
test = test.fillna(-1)

# Lets do a basic run to test methodology using folds5
# Folds 1:4 To build the xgb + meta predictors
# Fold 5 validation of final model.
# REM current best xgb had (not totally compariable) roc 0.9806

print 'Splitting data sets for meta features, and validiation'
meta_quoteNum = x_folds[x_folds['fold10'].isin(range(1,2))].QuoteNumber
meta_train = x_train[x_train['QuoteNumber'].isin(meta_quoteNum)]
meta_y = meta_train.QuoteConversion_Flag.values
meta_train = meta_train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
meta_names = list(meta_train)
print 'Shape of meta data', meta_train.shape
del meta_quoteNum

xgb_quoteNum = x_folds[x_folds['fold10'].isin(range(2,9))].QuoteNumber
xgb_train = x_train[x_train['QuoteNumber'].isin(xgb_quoteNum)]
xgb_y = xgb_train.QuoteConversion_Flag.values
xgb_train = xgb_train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
del xgb_quoteNum

valid_quoteNum = x_folds[x_folds['fold10'].isin([10])].QuoteNumber
valid_train = x_train[x_train['QuoteNumber'].isin(valid_quoteNum)]
valid_y = valid_train.QuoteConversion_Flag.values
valid_train = valid_train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
del valid_quoteNum
print 'Done...'

test = test.drop('QuoteNumber', axis=1)

seed = 260681
names = ["Random Forest", "Extra Trees", "Decision Tree", "Logistic Regression", "Naive Bayes", "Linear SVM", "RBF SVM"]
col_names =["predRF", "predET", "predDT", "predLR", "predNB", "predSVM", "predRBF"]
classifiers = [RandomForestClassifier(max_depth=10, n_estimators=500, n_jobs=-1, random_state=seed),
               ExtraTreesClassifier(max_depth=10, n_estimators=500, n_jobs=-1, random_state=seed),
               DecisionTreeClassifier(max_depth=7, random_state=seed),
               LogisticRegression(random_state=seed),
               GaussianNB(),
               SVC(kernel="linear", C=0.025, probability=True,cache_size=2000),
               SVC(gamma=2, C=1, probability=True,cache_size=2000)]

# Now the fun part, build 'weak' classifiers on the above models - easy to add more
# check the AUC on the validation set
# Append prediction to xgb and valid dataset and run xgb model on top

# iterate over classifiers to generate metafeatures.
for name, col_name, clf in zip(names, col_names, classifiers):
    print 'Building', name, 'model'
    clf.fit(meta_train, meta_y)
    # Predict values for validation check
    pred_valid = clf.predict_proba(valid_train[meta_names])[:,1]
    valid_train[col_name] = clf.predict_proba(valid_train[meta_names])[:,1]
    xgb_train[col_name] = clf.predict_proba(xgb_train[meta_names])[:,1]
    test[col_name] = clf.predict_proba(test[meta_names])[:,1]
    print 'AUC for classifier', name, '=', auc(valid_y, pred_valid)

pred_average = True
no_bags = 3
for k in range(no_bags):
    print 'Building bag:', k
    clf = xgb.XGBClassifier(n_estimators=1900,
                            nthread=-1,
                            max_depth=7,
                            learning_rate=0.02,
                            silent=True,
                            subsample=0.86,
                            colsample_bytree=0.68,
                            seed=k*100+22)
    clf.fit(xgb_train, xgb_y, eval_metric="auc",eval_set=[(valid_train, valid_y)], early_stopping_rounds=25)
    preds = clf.predict_proba(test)[:,1]
    if type(pred_average) == bool:
        pred_average = preds.copy()/no_bags
    else:
        pred_average += preds/no_bags

sample.QuoteConversion_Flag = pred_average
sample.to_csv('output/xgb_homesite_3bag_mp2_meta_test_30112015.csv', index=False)
