__author__ = 'michael.pearmain'

import pandas as pd
from sklearn.metrics import roc_auc_score as auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
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

names = ["Nearest Neighbors", "Random Forest", "AdaBoost", "Naive Bayes",
         "Linear SVM", "RBF SVM", "Decision Tree", "Linear Discriminant Analysis",
         "Quadratic Discriminant Analysis"]
classifiers = [
    KNeighborsClassifier(1000),
    RandomForestClassifier(max_depth=12, n_estimators=500, max_features=1,n_jobs=-1),
    AdaBoostClassifier(RandomForestClassifier(max_depth=12,n_jobs=-1),algorithm="SAMME",n_estimators=50),
    GaussianNB(),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    LDA(),
    QDA()]
# iterate over classifiers
for name, clf in zip(names, classifiers):
    clf.fit(x_train, y_train)
    pred_valid = clf.predict_proba(x_valid)[:,1]
    print 'AUC for classifier', name, '=', auc(y_valid, pred_valid)