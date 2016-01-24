# Homesite contest

This is the description of the approach taken by team New Model Army (Michael Pearmain and Konrad Banachewicz) in the Homesite Quote Conversion contest hosted by Kaggle:

https://www.kaggle.com/c/homesite-quote-conversion

## Data analysis

### Datasets
The original datasets (training and test) are processed to create several versions of train/test:
* kb1:
* kb2: 
* kb3: 

Those datasets are used as input to the metafeature creation (see below).

### Folds

The file *folds_prep.R* generates a split of the training set into 5 folds, 10 folds and a train/validation split. The output is a dataframe xfolds (and a file *xfolds.csv*), which is used in subsequent analysis. This way we can ensure consistency across different models - each time the same folds are used, so there is no leakage.

## Metafeatures 


## Ensembling

As per previous comps.

Data manips are in R to create 4 data sets

1. Full training set
2. Full test set
3. Training - Validation (90%)
4. Validation - (10%)
 
To get the current best score i run the data\_prep in R and hten run 10-bag xgb\_benchmark.py

##TODO:
Add more feature engineering.

Setup glmnet for ensembles script including CV.

I want to build models using the part-train data and check local CV

Re-train on full data for the model and produce predict on test

Ensemble using validation sets.

This way we get the benefit of using all data to train model on, and a clever way to ensemble prediction.


As a test i read in a forum that taking feature importance of top model, then doing say mod(3) on the feature in order of importance and rebuilding 3 new xgboost models with same params makes a big difference - This will be my next check to investigate
