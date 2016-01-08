# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 00:20:32 2015

@author: konrad
"""

import numpy as np
import pandas as pd
import datetime
import xgboost as xgb


if __name__ == '__main__':

    ## settings
    projPath = './'
    dataset_version = "mp3"
    model_type = "xgb"
    seed_value = 123
    todate = datetime.datetime.now().strftime("%Y%m%d")

    ## data
    # read the training and test sets
    xtrain = pd.read_csv(projPath + 'input/xtrain_'+ dataset_version + '.csv')
    id_train = xtrain.QuoteNumber
    ytrain = xtrain.QuoteConversion_Flag
    xtrain.drop('QuoteNumber', axis = 1, inplace = True)
    xtrain.drop('QuoteConversion_Flag', axis = 1, inplace = True)

    xtest = pd.read_csv(projPath + 'input/xtest_'+ dataset_version + '.csv')
    id_test = xtest.QuoteNumber
    xtest.drop('QuoteNumber', axis = 1, inplace = True)

    # folds
    xfolds = pd.read_csv(projPath + 'input/xfolds.csv')
    # work with 5-fold split
    fold_index = xfolds.fold5
    fold_index = np.array(fold_index) - 1
    n_folds = len(np.unique(fold_index))

    ## model
    # parameter grids: LR + range of training subjects to subset to
    '''
    Staying with index convention.
        child_weight = 0
        max_depth = 1
        colsample = 2
        rowsample = 3
        gamma_val = 4
        eta_val = 5
        ntrees = 6
    '''
    param_grid = [
        # (10, 4, 0.93, 0.856, 0.1, 0.02, 150),
        # (1, 40, 0.73, 0.756, 0.001, 0.02, 1500),
        # (4, 8, 0.87, 0.798, 0.0007, 0.043, 400),
        # (10, 6, 0.60, 0.90, 0, 0.0432, 380),
        # (3, 8, 0.87, 0.7890, 0, 0.09087, 350),
        # (1, 3, 0.789, 0.97, 0, 0.012, 1100),
        # (1, 7, 0.74, 0.64, 0.00015, 0.008, 6200),
        # (7, 9, 0.53, 0.879, 0.005, 0.019, 900),
        # (7, 6, 0.80, 0.54, 0, 0.0215, 1300),
        # (7, 3, 0.50, 0.66, 0, 0.08732, 300),
        # (1, 6, 0.89, 0.694, 0, 0.05421, 700),
        # (1, 10, 0.74, 0.908, 0, 0.0121, 1750),
        # (1, 15, 0.7890, 0.890643, 0.231, 0.21, 900),
        # (15, 15, 0.74231, 0.8634, 0.003832, 0.03, 900),
        # (1, 19, 0.78, 0.97453, 0, 0.001, 3900),
        # (2, 6, 0.91, 0.795342, 0, 0.032, 800),
        # (2, 8, 0.53, 0.53429, 0, 0.0213, 1500),
        # (8, 6, 0.87, 0.54238, 0, 0.0215, 1500),
        # (5, 60, 0.99, 0.98, 0.0001, 0.00821, 2300),
        # (3, 23, 0.564, 0.89534, 0.002, 0.025, 900),
        # (9, 5, 0.7, 0.98032, 0, 0.0289, 900),
        # (5, 9, 0.73, 0.7649, 0, 0.0187, 900)
        #(1, 6, 0.77, 0.83, 0, 0.023, 1800),
        #(1, 8, 0.77, 0.83, 0.001, 0.03, 900)
    ]

    # storage structure for forecasts
    mvalid = np.zeros((xtrain.shape[0],len(param_grid)))
    mfull = np.zeros((xtest.shape[0],len(param_grid)))

    ## build 2nd level forecasts
    for i in range(len(param_grid)):
        print "processing parameter combo:", param_grid[i]
        # configure model with j-th combo of parameters
        x = param_grid[i]
        clf = xgb.XGBClassifier(n_estimators=x[6],
                                nthread=-1,
                                max_depth=x[1],
                                min_child_weight=x[0],
                                learning_rate=x[5],
                                silent=True,
                                subsample=x[3],
                                colsample_bytree=x[2],
                                gamma=x[2],
                                seed=seed_value)

        # loop over folds - Keeping as pandas for ease of use with xgb wrapper
        for j in range(1 ,n_folds+1):
            idx0 = xfolds[xfolds.fold5 != j].index
            idx1 = xfolds[xfolds.fold5 == j].index
            x0 = xtrain[xtrain.index.isin(idx0)]
            x1 = xtrain[xtrain.index.isin(idx1)]
            y0 = ytrain[ytrain.index.isin(idx0)]
            y1 = ytrain[ytrain.index.isin(idx1)]

            # fit the model on observations associated with subject whichSubject in this fold
            clf.fit(x0, y0, eval_metric="auc", eval_set=[(x1, y1)])
            mvalid[idx1,i] = clf.predict_proba(x1)[:,1]

        # fit on complete dataset
        bst = xgb.XGBClassifier(n_estimators=x[6],
                                nthread=-1,
                                max_depth=x[1],
                                min_child_weight=x[0],
                                learning_rate=x[5],
                                silent=True,
                                subsample=x[3],
                                colsample_bytree=x[2],
                                gamma=x[2],
                                seed=seed_value)
        bst.fit(xtrain, ytrain, eval_metric="auc")
        mfull[:,i] = bst.predict_proba(xtest)[:,1]


    ## store the results
    # add indices etc
    mvalid = pd.DataFrame(mvalid)
    mvalid.columns = [model_type + str(i) for i in range(0, mvalid.shape[1])]
    mvalid['QuoteNumber'] = id_train
    mvalid['QuoteConversion_Flag'] = ytrain

    mfull = pd.DataFrame(mfull)
    mfull.columns = [model_type + str(i) for i in range(0, mfull.shape[1])]
    mfull['QuoteNumber'] = id_test


    # save the files
    mvalid.to_csv(projPath + 'metafeatures/prval_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False, header = True)
    mfull.to_csv(projPath + 'metafeatures/prfull_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False, header = True)