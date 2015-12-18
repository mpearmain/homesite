__author__ = 'michael.pearmain'

'''
    Quite a long script and will take some time to run:
    Idea is to run bayesian optimization on RF, and Extra tree to find optimal params from the train set.
    We then run a models for:
    RF, ET, XGB, DTC, NB, and LR for the dataset in questions and output the prediction on the validation set,
    and the output on the test set.

    All Models are stored in ./submission/
    with format:
        validation: predValid_modeltype_dataset_seed_params.csv
        test: testValid_modeltype_dataset_seed_params.csv


'''
import pandas as pd
from sklearn.metrics import roc_auc_score as auc
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression

from bayesian_optimization import BayesianOptimization
from sklearn.tree import DecisionTreeClassifier as DTC


def rfccv(n_estimators, min_samples_split, max_features):
    clf = BaggingClassifier(RFC(n_estimators=int(n_estimators),
                                min_samples_split=int(min_samples_split),
                                max_features=min(max_features, 0.999),
                                random_state=1234,
                                n_jobs=-1),
                            random_state=1234)
    # Predict values for validation check
    clf.fit(meta_train, meta_y)
    pred_valid = clf.predict_proba(valid_train)[:,1]
    return auc(valid_y, pred_valid)

def etccv(n_estimators, min_samples_split, max_features):
    clf = BaggingClassifier(ETC(n_estimators=int(n_estimators),
                                min_samples_split=int(min_samples_split),
                                max_features=min(max_features, 0.999),
                                random_state=1234,
                                n_jobs=-1),
                            random_state=1234)
    # Predict values for validation check
    clf.fit(meta_train, meta_y)
    pred_valid = clf.predict_proba(valid_train)[:,1]
    return auc(valid_y, pred_valid)

def dtcv(max_depth):
    clf = BaggingClassifier(DTC(max_depth=int(max_depth),
                                random_state=1234),
                            random_state=1234,
                            n_jobs=-1)
    # Predict values for validation check
    clf.fit(meta_train, meta_y)
    pred_valid = clf.predict_proba(valid_train)[:,1]
    return auc(valid_y, pred_valid)



########################################################################################################################
####### RUN ########

if __name__ == "__main__":
    DATASETS_TRAIN = ['input/xtrain_mp1.csv', 'input/xtrain_kb4.csv']
    DATASETS_TEST = ['input/xtest_mp1.csv', 'input/xtest_kb4.csv']
    SEEDS = [1234, 5678, 9101112]

    print('Loading X-folds data set')
    x_folds = pd.read_csv('input/xfolds.csv')

    # Read submission file for later to construct testPredict.
    submission = pd.read_csv('input/sample_submission.csv')

    # Create the loop structure.
    for i in xrange(len(DATASETS_TRAIN)):

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

        train_quoteNum = x_train.QuoteNumber.values
        train_y = x_train.QuoteConversion_Flag.values
        x_train = x_train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)

        print 'Done...'

        print 'Running Decision Trees Optimization'
        dtBO = BayesianOptimization(dtcv, {'max_depth': (int(5), int(25))})
        print('-'*53)
        dtBO.maximize(init_points=5, restarts=250, n_iter=5)
        print('DT: %f' % dtBO.res['max']['max_val'])

        print 'Running Random Forest Optimization'
        rfcBO = BayesianOptimization(rfccv, {'n_estimators': (int(250), int(500)),
                                             'min_samples_split': (int(15), int(25)),
                                             'max_features': (0.05, 0.31)})
        print('-'*53)
        rfcBO.maximize(init_points=5, restarts=250, n_iter=5)
        print('RFC: %f' % rfcBO.res['max']['max_val'])

        print 'Running Extra Trees Optimization'
        etcBO = BayesianOptimization(etccv, {'n_estimators': (int(250), int(500)),
                                             'min_samples_split': (int(15), int(25)),
                                             'max_features': (0.05, 0.31)})
        print('-'*53)
        etcBO.maximize(init_points=5, restarts=250, n_iter=7)
        print('ETC: %f' % etcBO.res['max']['max_val'])



        ##################################################################################################
        #
        # Now predict on valid and Test
        for seed in SEEDS:
            names = ["RandomForest", "ExtraTrees", "DecisionTree", "LogisticRegression"]
            classifiers = [BaggingClassifier(RFC(n_estimators=int(rfcBO.res['max']['max_params']['n_estimators']),
                                                 min_samples_split=int(rfcBO.res['max']['max_params']['min_samples_split']),
                                                 max_features=rfcBO.res['max']['max_params']['max_features'],
                                                 n_jobs=-1,
                                                 random_state=seed),
                                             random_state=seed),
                           BaggingClassifier(ETC(n_estimators=int(etcBO.res['max']['max_params']['n_estimators']),
                                                 min_samples_split=int(etcBO.res['max']['max_params']['min_samples_split']),
                                                 max_features=etcBO.res['max']['max_params']['max_features'],
                                                 n_jobs=-1,
                                                 random_state=seed),
                                             random_state=seed),
                           BaggingClassifier(DTC(max_depth=int(dtBO.res['max']['max_params']['max_depth']),
                                                 random_state=seed),
                                             random_state=seed,
                                             n_jobs=-1),
                           BaggingClassifier(LogisticRegression(random_state=seed),
                                             random_state=seed)]

            # iterate over classifiers to generate predictions..
            for name, clf in zip(names, classifiers):
                print 'Building', name, 'model'
                clf.fit(x_train, train_y)
                # Predict values for validation check
                print 'Writing bagging prediction to file...'
                pred_train = clf.predict_proba(x_train)[:,1]
                d = {'QuoteNumber': train_quoteNum, name+DATASETS_TRAIN[i][6:]+str(seed): train_valid}
                df = pd.DataFrame(data=d, index=None)
                build_path = './submission/predTrain_' + name + '_' + str(seed) + '_' + DATASETS_TRAIN[i][6:]
                df.to_csv(build_path, index=None)

                print 'Writing Test submission file...'
                pred_test = clf.predict_proba(test)[:, 1]
                submission = pd.read_csv('input/sample_submission.csv')
                test_quoteNum = submission.QuoteNumber
                d = {'QuoteNumber': test_quoteNum, name+DATASETS_TRAIN[i][6:]+str(seed): pred_test}
                df = pd.DataFrame(data=d, index=None)
                build_path = './submission/predTest_' + name + '_' + str(seed) + '_' + DATASETS_TRAIN[i][6:]
                df.to_csv(build_path, index=None)


