__author__ = 'michael.pearmain'

import pandas as pd
from sklearn.metrics import roc_auc_score as auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA

# Got a little bored so decided to just build a loop over a whole bunch of classifiers
# who knows, might be good on the ensemble.

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

names = ["Nearest Neighbors", "Random Forest", "Extra Trees", "AdaBoost", "Naive Bayes",
         "Decision Tree", "Bagging Classifier", "Passive Aggressive", "Linear Discriminant Analysis",
         "Quadratic Discriminant Analysis"]
classifiers = [
    KNeighborsClassifier(10000),
    RandomForestClassifier(max_depth=12, n_estimators=500, n_jobs=-1),
    ExtraTreesClassifier(max_depth=12, n_estimators=500, n_jobs=-1),
    AdaBoostClassifier(RandomForestClassifier(max_depth=12,n_jobs=-1),algorithm="SAMME",n_estimators=500),
    GaussianNB(),
    DecisionTreeClassifier(max_depth=12),
    BaggingClassifier(n_estimators=500, n_jobs=-1),
    PassiveAggressiveClassifier(n_iter=500, n_jobs=-1),
    LDA(),
    QDA()]
# iterate over classifiers
for name, clf in zip(names, classifiers):
    clf.fit(x_train, y_train)
    pred_valid = clf.predict_proba(x_valid)[:,1]
    print 'AUC for classifier', name, '=', auc(y_valid, pred_valid)