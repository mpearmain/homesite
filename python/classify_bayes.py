__author__ = 'michael.pearmain'

''' Simple set of functions to run bayes optimization on simple models'''

import pandas as pd
from sklearn.metrics import roc_auc_score as auc
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from bayesian_optimization import BayesianOptimization
from sklearn.cross_validation import cross_val_score

def rfccv(n_estimators, min_samples_split, max_features):
    clf = RFC(n_estimators=int(n_estimators),
              min_samples_split=int(min_samples_split),
              max_features=min(max_features, 0.999),
              random_state=1234,
              n_jobs=-1)
    # Predict values for validation check
    clf.fit(meta_train, meta_y)
    pred_valid = clf.predict_proba(valid_train)[:,1]
    return auc(valid_y, pred_valid)

def etccv(n_estimators, min_samples_split, max_features):
    clf = ETC(n_estimators=int(n_estimators),
              min_samples_split=int(min_samples_split),
              max_features=min(max_features, 0.999),
              random_state=1234,
              n_jobs=-1)
    # Predict values for validation check
    clf.fit(meta_train, meta_y)
    pred_valid = clf.predict_proba(valid_train)[:,1]
    return auc(valid_y, pred_valid)

if __name__ == "__main__":
    # Read the data files in - Train, test, and xfolds.
    print('Loading Full Train data set')
    x_train = pd.read_csv('input/xtrain_mp1.csv')
    print('Loading X-folds data set')
    x_folds = pd.read_csv('input/xfolds.csv')

    # Make sure final set has no na's (should be solved in data_preparation.R)
    x_train = x_train.fillna(-1)

    print 'Splitting data sets for meta features, and validiation'
    meta_quoteNum = x_folds[x_folds['fold5'].isin(range(1,5))].QuoteNumber
    meta_train = x_train[x_train['QuoteNumber'].isin(meta_quoteNum)]
    meta_y = meta_train.QuoteConversion_Flag.values
    meta_train = meta_train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
    meta_names = list(meta_train)
    del meta_quoteNum

    valid_quoteNum = x_folds[x_folds['fold5'].isin([5])].QuoteNumber
    valid_train = x_train[x_train['QuoteNumber'].isin(valid_quoteNum)]
    valid_y = valid_train.QuoteConversion_Flag.values
    valid_train = valid_train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
    del valid_quoteNum
    print 'Done...'


    print 'Running Random Forest Optimization'
    rfcBO = BayesianOptimization(rfccv, {'n_estimators': (int(500), int(1500)),
                                         'min_samples_split': (int(1), int(25)),
                                         'max_features': (0.1, 1)})
    print('-'*53)
    rfcBO.maximize(restarts=50)
    print('RFC: %f' % rfcBO.res['max']['max_val'])

    print 'Running Extra Trees Optimization'
    etcBO = BayesianOptimization(rfccv, {'n_estimators': (int(500), int(1500)),
                                         'min_samples_split': (int(1), int(25)),
                                         'max_features': (0.1, 1)})
    print('-'*53)
    etcBO.maximize(restarts=50)
    print('RFC: %f' % etcBO.res['max']['max_val'])