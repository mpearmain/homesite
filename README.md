# Homesite contest

This is the description of the approach taken by team New Model Army (Michael Pearmain and Konrad Banachewicz) in the Homesite Quote Conversion contest hosted by Kaggle:

https://www.kaggle.com/c/homesite-quote-conversion

The general idea is a two-level architecture:
1. create multiple transformations of the train/test data
2. generate stacked predictions using multiple models trained on versions of the data (metafeatures)
3. create a stacked ensemble of the metafeatures

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

Python code exists to run 5fold second level models using a variety of different classification models;
XGB, NN, RF, ET, DT, LogReg 

## Ensembling
Second level features are ensembled via hill climbing

