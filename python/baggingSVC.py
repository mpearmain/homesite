
import pandas as pd
from sklearn.metrics import roc_auc_score as auc
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
import time


DATASETS_TRAIN = ['input/xtrain_mp1.csv']
DATASETS_TEST = ['input/xtest_mp1.csv']
SEEDS = [1234]

print('Loading X-folds data set')
x_folds = pd.read_csv('input/xfolds.csv')

# Read submission file for later to construct testPredict.
submission = pd.read_csv('input/sample_submission.csv')

# Create the loop structure.
for i in xrange(len(DATASETS_TRAIN)):
    for seed in SEEDS:
        # Read the data files in - Train, test, and xfolds.
        print 'Loading', DATASETS_TRAIN[i], 'data set'
        x_train = pd.read_csv(DATASETS_TRAIN[i])
        print 'Loading', DATASETS_TEST[i], 'data set'
        test = pd.read_csv(DATASETS_TEST[i])
        test = test.drop('QuoteNumber', axis=1)

        # Make sure final set has no na's (should be solved in data_preparation.R)
        x_train = x_train.fillna(-1)

        print 'Splitting data sets for train, and validiation'
        meta_quoteNum = x_folds[x_folds['valid'] == 0 ].QuoteNumber
        meta_train = x_train[x_train['QuoteNumber'].isin(meta_quoteNum)]
        meta_y = meta_train.QuoteConversion_Flag.values
        meta_train = meta_train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
        meta_names = list(meta_train)
        del meta_quoteNum

        valid_quoteNum = x_folds[x_folds['valid'] == 1 ].QuoteNumber
        valid_train = x_train[x_train['QuoteNumber'].isin(valid_quoteNum)]
        valid_y = valid_train.QuoteConversion_Flag.values
        valid_train = valid_train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
        valid_quoteNum
        print 'Done...'

        start = time.time()
        n_estimators = 20
        clf = BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='auto'),
                                max_samples=1.0 / n_estimators,
                                n_estimators=n_estimators,
                                n_jobs=-1)
        clf.fit(meta_train, meta_y)
        end = time.time()
        print "Bagging SVC", end - start
        pred_valid = clf.predict_proba(valid_train)
        print 'AUC for Bagging SVC =', auc(valid_y, pred_valid)



