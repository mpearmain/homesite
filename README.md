# homesite

As per previous comps.

Data manips are in R to create 4 data sets

1. Full training set
2. Full test set
3. Training - Validation (90%)
4. Validation - (10%)
 
To get the current best score i run the data\_prep in R and hten run 10-bag xgb\_benchmark.py

##TODO:

Setup glmnet for ensembles script including CV.
I want to build models using the part-train data and check local CV
Re-train on full data for the model and produce predict on test
Ensemble using validation sets.

This way we get the benefit of using all data to train model on, and a clever way to ensemble prediction.
