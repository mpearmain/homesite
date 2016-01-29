from __future__ import print_function
from __future__ import division

__author__ = 'michael.pearmain'

import pandas as pd
import datetime
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score as auc
from bayesian_optimization import BayesianOptimization

def xgboostcv(max_depth,
              learning_rate,
              n_estimators,
              subsample,
              colsample_bytree,
              gamma,
              min_child_weight,
              silent=True,
              nthread=-1,
              seed=1234):

    clf = XGBClassifier(max_depth=int(max_depth),
                        learning_rate=learning_rate,
                        n_estimators=int(n_estimators),
                        silent=silent,
                        nthread=nthread,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        gamma=gamma,
                        min_child_weight = min_child_weight,
                        seed=seed,
                        objective="binary:logistic")

    clf.fit(x0, y0, eval_metric="auc", eval_set=[(x1, y1)], early_stopping_rounds = 25)
    roc =  auc(y1, clf.predict_proba(x1)[:,1])

    return roc

if __name__ == "__main__":
    ## settings
    projPath = '/Users/konrad/Documents/projects/homesite/'
    dataset_version = "kb9"
    model_type = "xgb"
    seed_value = 5610
    todate = datetime.datetime.now().strftime("%Y%m%d")

    ## data
    # read the training and test sets
    # xtrain = pd.read_csv(projPath + 'input/xvalid_20160127.csv')
    xtrain = pd.read_csv(projPath + 'input/xtrain_' + dataset_version + '.csv')
    id_train = xtrain.QuoteNumber
    ytrain = xtrain.QuoteConversion_Flag
    xtrain.drop('QuoteNumber', axis = 1, inplace = True)
    xtrain.drop('QuoteConversion_Flag', axis = 1, inplace = True)

    # xtest = pd.read_csv(projPath + 'input/xfull_20160127.csv')
    xtest = pd.read_csv(projPath + 'input/xtest_' + dataset_version + '.csv')
    id_test = xtest.QuoteNumber
    xtest.drop('QuoteNumber', axis = 1, inplace = True)

    # Get rid of incorrect names for xgboost (scv-rbf) cannont handle '-'
    xtrain = xtrain.rename(columns=lambda x: x.replace('-', ''))
    xtest = xtest.rename(columns=lambda x: x.replace('-', ''))

    # folds
    xfolds = pd.read_csv(projPath + 'input/xfolds.csv')
    # work with validation split
    idx0 = xfolds[xfolds.valid == 0].index
    idx1 = xfolds[xfolds.valid == 1].index
    x0 = xtrain[xtrain.index.isin(idx0)]
    x1 = xtrain[xtrain.index.isin(idx1)]
    y0 = ytrain[ytrain.index.isin(idx0)]
    y1 = ytrain[ytrain.index.isin(idx1)]

    xgboostBO = BayesianOptimization(xgboostcv,
                                     {'max_depth': (int(6), int(10)),
                                      'learning_rate': (0.03, 0.005),
                                      'n_estimators': (int(350), int(750)),
                                      'subsample': (0.8, 0.9),
                                      'colsample_bytree': (0.8, 0.9),
                                      'gamma': (0.0001, 0.0007),
                                      'min_child_weight': (int(1), int(20))  
                                     })

    xgboostBO.maximize(init_points=7, restarts=250, n_iter=20)
    print('-' * 53)

    print('Final Results')
    print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])
