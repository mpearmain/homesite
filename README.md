# Kaggle Homesite contest 3rd Place solution (New Model Army part)

This is the description of the approach taken by team New Model Army (Michael Pearmain and Konrad Banachewicz) in the Homesite Quote Conversion contest hosted by Kaggle:

https://www.kaggle.com/c/homesite-quote-conversion

The general idea is a two-level architecture:
1. Create multiple transformations of the train/test data
2. Generate stacked predictions using multiple models trained on versions of the data (metafeatures)
3. Create a stacked ensemble of the metafeatures

### System and Dependenices  
To replicate the full solution will require 16GB RAM min, and depending on the number of cores and machines available will take in excess of 10days to run - (This is due to teh heavy workload of producing meta-level features for 5-fold splits on multiple datasets for multiple models)

The code is written as a combination of python (2.7.6) with the following libraries:
  * numpy
  * pandas
  * keras
  * sklearn
  * datetime
  * itertools
  * xgboost
  
and R (3.2.1) with the following libraries:
  * data.table
  * caret
  * stringr
  * readr
  * lubridate
  * Rtnse
  * lme4
  * chron
  * h2o
  * earth
  * ranger
  * glmnet
  * Metrics
  * e1071
  * nnet
  * xgboost


## Solution Replication
To replicate the top NMA (New Model Army) submission all should be run at teh top level dir, it is expected that in `./input` are the train test and sample submission files for the contest. It also expects a `metafeature` folder, and a `submissions` folder.

### Datasets
Dataset creation was achieved in R by running the `~/R/data_preparation.R` script
Those datasets are used as input to the metafeature creation.

WARNING:- The creation of the datasets may take in excess of 2 days.

### Folds

The file `~/R/folds_prep.R` generates a split of the training set into 5 folds, 10 folds and a train/validation split. The output is a dataframe xfolds (and a file `xfolds.csv`), which is used in subsequent analysis. This way we can ensure consistency across different models - each time the same folds are used, so there is no leakage, and allows us to compare performace of all new models.

### Metafeatures 

This is the main workhorse of the solution where we create fold second level models using a variety of different classification models using `R` and `python`.  Within the `~/R/` and `~/python` directories are numerous `build_meta_XXX` files.  Each of these files must be run and the output is saved into the `~/metafeatures` subdirectory

### Second level Meta Ensembling.
Once we have a variety of models prediction as meta features, we then build a second stage set of meta features using `build_ensemble.R`. This script firstly removes any linear combinations of metafeatures that may have been produced, and then runs five more classification stack prediction, (nnet, xgboost, ranger, HillClimbing (which is our implementation of the "library of models" approach of Caruana et al), and glmnet

### Third Level Meta Ensemble.
This was a final blend of the models using our hand crafted hillclimbing model.


### Ensembling
Second level features are ensembled via hill climbing

