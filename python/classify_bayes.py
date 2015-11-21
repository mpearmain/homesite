__author__ = 'michael.pearmain'

''' Simple set of functions to run bayes optimization on simple models'''

import pandas as pd
from sklearn.metrics import roc_auc_score as auc
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from bayesian_optimization import BayesianOptimization
from sklearn.cross_validation import cross_val_score

def rfccv(n_estimators, min_samples_split, max_features):
    return cross_val_score(RFC(n_estimators=int(n_estimators),
                               min_samples_split=int(min_samples_split),
                               max_features=min(max_features, 0.999),
                               random_state=1234,
                               n_jobs=-1),
                           x_train, y_train, 'roc_auc', cv=3).mean()

def etccv(n_estimators, min_samples_split, max_features):
    return cross_val_score(ETC(n_estimators=int(n_estimators),
                               min_samples_split=int(min_samples_split),
                               max_features=min(max_features, 0.999),
                               random_state=1234,
                               n_jobs=-1),
                           x_train, y_train, 'roc_auc', cv=3).mean()

if __name__ == "__main__":
    print('Loading Train data set')
    x_train = pd.read_csv('input/xtrain_mp1.csv')
    print('Loading Test data set')
    test = pd.read_csv('input/xtest_mp1.csv')

    sample = pd.read_csv('input/sample_submission.csv')

    y_train = x_train.QuoteConversion_Flag.values
    x_train = x_train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
    test = test.drop('QuoteNumber', axis=1)

    x_train = x_train.fillna(-1)
    test = test.fillna(-1)


    print 'Running Random Forest Optimization'
    rfcBO = BayesianOptimization(rfccv, {'n_estimators': (int(750), int(2500)),
                                         'min_samples_split': (int(2), int(25)),
                                         'max_features': (0.1, 1)})
    print('-'*53)
    rfcBO.maximize(restarts=100)
    print('RFC: %f' % rfcBO.res['max']['max_val'])

    print 'Running Extra Trees Optimization'
    etcBO = BayesianOptimization(rfccv, {'n_estimators': (int(750), int(2500)),
                                         'min_samples_split': (int(2), int(25)),
                                         'max_features': (0.1, 1)})
    print('-'*53)
    etcBO.maximize(restarts=100)
    print('RFC: %f' % etcBO.res['max']['max_val'])