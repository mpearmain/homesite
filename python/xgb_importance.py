# coding: utf-8
__author__ = 'mpearmain'


import pandas as pd
import xgboost as xgb
from python.importance_splits import generate_feature_labels
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

pred_average = True
no_bags = 1
for k in range(no_bags):
    clf = xgb.XGBClassifier(n_estimators=338,
                            nthread=-1,
                            max_depth=9,
                            learning_rate=0.045827517804649449,
                            silent=True,
                            subsample=0.795364790,
                            colsample_bytree=0.57238046827515454,
                            seed=k*100+22)
    xgb_model = clf.fit(x_train, y_train, eval_metric="auc")
    preds = clf.predict_proba(test)[:,1]
    pred_valid = clf.predict_proba(x_valid)[:,1]
    if type(pred_average) == bool:
        pred_average = preds.copy()/no_bags
        validation_pred_average = pred_valid.copy()/no_bags
    else:
        pred_average += preds/no_bags
        validation_pred_average += pred_valid/no_bags
print 'AUC full features = ', auc(y_valid, validation_pred_average)

sample.QuoteConversion_Flag = pred_average
sample.to_csv('output/xgb_homesite_full_bagged_test_13112015.csv', index=False)

# Now lets re-run for sets of features based on ranked importance from the above model.
importance_labels = generate_feature_labels(clf._Booster, 2)

importance_pred_average = True
feature_sets = len(importance_labels)
for k in range(feature_sets):
    features = importance_labels[k]
    clf = xgb.XGBClassifier(n_estimators=338,
                            nthread=-1,
                            max_depth=9,
                            learning_rate=0.045827517804649449,
                            silent=True,
                            subsample=0.795364790,
                            colsample_bytree=0.57238046827515454,
                            seed=k*100+22)
    xgb_model = clf.fit(x_train[features], y_train, eval_metric="auc")
    preds = clf.predict_proba(test[features])[:,1]
    pred_valid = clf.predict_proba(x_valid[features])[:,1]
    print 'Running for feature set ', k, ' - AUC value = ', auc(y_valid, pred_valid)
    if type(importance_pred_average) == bool:
        importance_pred_average = preds.copy()/feature_sets
        validation_pred_average = pred_valid.copy()/feature_sets
    else:
        importance_pred_average += preds/feature_sets
        validation_pred_average += pred_valid/feature_sets

print 'AUC combined features = ', auc(y_valid, validation_pred_average)


sample.QuoteConversion_Flag = importance_pred_average
sample.to_csv('output/xgb_homesite_full_10bag_13112015.csv', index=False)








