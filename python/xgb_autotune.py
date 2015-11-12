from __future__ import print_function
from __future__ import division

__author__ = 'michael.pearmain'

import pandas as pd
from sklearn.metrics import roc_auc_score as auc
from xgboost import XGBClassifier
from bayesian_optimization import BayesianOptimization

def xgboostcv(max_depth,
              learning_rate,
              n_estimators,
              subsample,
              colsample_bytree,
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
                        seed=seed,
                        objective="binary:logistic")

    xgb_model = clf.fit(x_train, y_train, eval_metric="auc", eval_set=[(x_valid, y_valid)], early_stopping_rounds=25)

    print('xgb best round = ', xgb_model.best_iteration)
    # Because our objective function is correct we can make use of the early stopping much easier an just set it very
    # high
    return xgb_model.best_score

if __name__ == "__main__":
    print('Loading Train data set')
    x_train = pd.read_csv('input/xtrain.csv')
    print('Loading Valid data set')
    x_valid = pd.read_csv('input/xvalid.csv')
    print('Loading Test data set')
    test = pd.read_csv('input/xtest.csv')

    y_train = x_train.QuoteConversion_Flag.values
    y_valid = x_valid.QuoteConversion_Flag.values
    x_train = x_train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
    x_valid = x_valid.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
    test = test.drop('QuoteNumber', axis=1)

    x_train = x_train.fillna(-1)
    x_valid = x_valid.fillna(-1)
    test = test.fillna(-1)

    xgboostBO = BayesianOptimization(xgboostcv,
                                     {'max_depth': (int(5), int(40)),
                                      'learning_rate': (0.05, 0.01),
                                      'n_estimators': (int(5000), int(5000)),
                                      'subsample': (0.65, 0.9),
                                      'colsample_bytree': (0.5, 0.9)
                                     })

    xgboostBO.maximize(init_points=7, restarts=100, n_iter=50)
    print('-' * 53)

    print('Final Results')
    print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])

    #
    # # Build and Run on the full data set K-fold times for bagging
    #
    # pred_average = True
    # seeds = [1234, 2606081, 998466, 229883, 8838120]
    # for seed_bag in seeds:
    #     clf = XGBClassifier(max_depth=int(xgboostBO.res['max']['max_params']['max_depth']),
    #                         learning_rate=xgboostBO.res['max']['max_params']['learning_rate'],
    #                         n_estimators=int(xgboostBO.res['max']['max_params']['n_estimators']),
    #                         subsample=xgboostBO.res['max']['max_params']['subsample'],
    #                         colsample_bytree=xgboostBO.res['max']['max_params']['colsample_bytree'],
    #                         seed=seed_bag,
    #                         objective="binary:logistic")
    #
    #     clf.fit(x_train, y_train, eval_metric="auc", eval_set=[(x_valid, y_valid)], early_stopping_rounds=25)
    #     print('Prediction Complete')
    #     preds = clf.predict_proba(test)[:, 1]
    #     if type(pred_average) == bool:
    #         pred_average = preds.copy()/len(seeds)
    #     else:
    #         pred_average += preds/len(seeds)
    #
    #     submission = pd.read_csv('input/sample_submission.csv')
    #     submission.QuoteConversion_Flag = pred_average
    #     outfile_seed = 'output/xgb_v2_bag_' + len(seed_bag) + '.csv'
    #     submission.to_csv(outfile_seed)
    #
    #
