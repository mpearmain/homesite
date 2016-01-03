# Data Creation Script
# These output can then be used in all subsequent models, and the prediction can ben ensembled from
# the validation using ridge regression (say).

## packages loading ####
library(data.table)
library(caret)
library(stringr)
library(readr)
library(lubridate)
library(Rtsne)
require(lme4)
require(chron)

set.seed(260681)

## functions ####

msg <- function(mmm,...)
{
  cat(sprintf(paste0("[%s] ",mmm),Sys.time(),...)); cat("\n")
}

## MP set v1 ####

BuildMP1 <- function() {
  # Wrapping the builds in functions to make it easier to call just one Dataset Build.
  # Working Dir should be top level with folders ./R, ./input, ./output
  train <- fread('input/train.csv')
  test <- fread('input/test.csv')
  
  # Lets first align the datasets for equal vars to work with.
  y <- train[, QuoteConversion_Flag]
  train[, c('QuoteConversion_Flag') := NULL]
  train[, dset := 0]
  test[, dset := 1]
  
  #Quick check to align data dimensions.
  stopifnot(dim(test)[2] ==dim(train)[2])
  # Join the datasets for simple manipulations.
  bigD <- rbind(train, test)
  rm(list = c('train', 'test'))
  
  ## Data Manipulations 
  # Lets solve the Field10 issue: 1,165 => 1165.
  bigD[, Field10 := as.numeric(gsub(",", "", Field10))]
  
  # First lets work with the dates and remove "Original_Quote_Date" field.
  bigD[, Date := parse_date_time(Original_Quote_Date, "%Y%m%d")]
  bigD[, Original_Quote_Date := NULL]
  
  # Lets get the year, month, day of month, weekday,
  bigD[, year := as.factor(year(Date))]
  bigD[, month := as.factor(month(Date))]
  bigD[, monthday := as.factor(mday(Date))]
  bigD[, weekday := as.factor(wday(Date))]
  # Adding year day - people rebuy the same year?
  bigD[, yearday := as.factor(yday(Date))]
  
  # Fake duration date - How long since the start of the comp (Mon 9 Nov 2015)
  # Taking logs as magnitude is an order difference 175 min 1042 max
  # Checking distributions between train and test proves equal distributions. --> Good to know for CV.
  bigD[, daysDurOrigQuote := as.integer(parse_date_time("2015-11-09", "%Y%m%d") - Date)]
  bigD[, logDaysDurOrigQuote := log(as.integer(parse_date_time("2015-11-09", "%Y%m%d") - Date))]
  bigD[, Date := NULL]
  
  # Lets model temporal effects in time
  # http://www.rochester.edu/College/PSC/signorino/research/Carter_Signorino_2010_PA.pdf
  # Needs more work for anything useful
  #bigD[, logdaysDurOrigQuote2 := log((daysDurOrigQuote)^2 - daysDurOrigQuote)]
  #bigD[, logdaysDurOrigQuote3 := log((daysDurOrigQuote)^3 - daysDurOrigQuote)]
  
  bigD[is.na(bigD)] <- -1
  
  # Count -1's across the data set
  bigD[, CoverageNeg1s := rowSums(.SD == -1), .SDcols = grep("CoverageField", names(bigD))]
  bigD[, SalesNeg1s := rowSums(.SD == -1), .SDcols = grep("SalesField", names(bigD))]
  bigD[, PropertyNeg1s := rowSums(.SD == -1), .SDcols = grep("PropertyField", names(bigD))]
  bigD[, GeoNeg1s := rowSums(.SD == -1), .SDcols = grep("GeographicField", names(bigD))]
  bigD[, PersonalNeg1s := rowSums(.SD == -1), .SDcols = grep("PersonalField", names(bigD))]
  # Finally Total across all.
  bigD[,
       TotalNeg1s := rowSums(.SD),
       .SDcols = c("CoverageNeg1s", "SalesNeg1s", "PropertyNeg1s", "GeoNeg1s", "PersonalNeg1s")]
  
  
  # Catch factor columns
  fact_cols <- which(lapply(bigD, class) == "character")
  # Map all categoricals into numeric.
  cat("Assuming text variables are categorical & replacing them with numeric ids\n")
  for (f in colnames(bigD)) {
    if (class(bigD[[f]])=="character") {
      print(f)
      levels <- unique(bigD[[f]])
      bigD[[f]] <- as.integer(factor(bigD[[f]], levels=levels))
    }
  }
  
  ## Split files & Export 
  xtrain <- bigD[dset == 0, ]
  xtest <- bigD[dset == 1, ]
  rm(bigD)
  
  xtrain[, dset := NULL]
  xtest[, dset := NULL]
  
  xtrain[, QuoteConversion_Flag := y]
  
  write.csv(xtrain, 'input/xtrain_mp1.csv', row.names = F)
  write.csv(xtest, 'input/xtest_mp1.csv', row.names = F)
  
  rm(xtrain)
  rm(xtest)
  return(cat("MP1 dataset built"))
}

BuildMP1()

## MP set v2 ####
  # Same as V1 only adding tnse features to the mix.#
BuildMP2 <- function() {
  
  train = fread('input/xtrain_mp1.csv', header=TRUE, data.table = F)
  test = fread('input/xtest_mp1.csv', header=TRUE, data.table = F)
  submission = fread('input/sample_submission.csv')
  
  train.Qnumber <- as.data.frame(train[,1])
  setnames(train.Qnumber, 'QuoteNumber')
  test.Qnumber <- as.data.frame(test[,1])
  setnames(test.Qnumber, 'QuoteNumber')
  
  train = train[,-1]
  test = test[,-1]
  
  y = as.data.frame(train[,ncol(train)])
  setnames(y, 'QuoteConversion_Flag')
  train = train[,-ncol(train)]
  
  train.names <- names(train)
  test.names <- names(test)
  
  x = rbind(train,test)
  x = as.matrix(x)
  x = matrix(as.numeric(x),nrow(x),ncol(x))
  
  # Running this takes a LONG time. -> Need min 16GB RAM spare
  tsne <- Rtsne(as.matrix(x), 
                check_duplicates = FALSE, 
                pca = FALSE, 
                perplexity=30, 
                theta=0.5, 
                dims=2)
  
  # Add to the mix of features -> cbind because its a matrix
  x = cbind(x, tsne$Y[,1]) 
  x = cbind(x, tsne$Y[,2])
  
  # Get index of train and test set to split when training
  trind = 1:dim(y)[1]
  teind = (nrow(train)+1):nrow(x)
  
  trainX = as.data.frame(x[trind,])
  testX = as.data.frame(x[teind,])
  
  setnames(trainX, c(train.names, 'tnse1', 'tnse2'))
  setnames(testX, c(test.names, 'tnse1', 'tnse2'))
  
  trainX <- cbind(train.Qnumber, trainX)
  trainX <- cbind(trainX, y)
  testX <- cbind(test.Qnumber, testX)
  
  
  # Output tnse train and test files to save re-running.
  write.csv(trainX, 'input/xtrain_mp2.csv', row.names = F, quote = F)
  write.csv(testX, 'input/xtest_mp2.csv', row.names = F, quote = F)
  
  rm(list=c(teind, trind, testX, train.Qnumber, test.Qnumber, trainX, x, y, tsne))
  return(cat("MP2 dataset built"))
}

BuildMP2()

BuildMP3 <- function() {
  # Wrapping the builds in functions to make it easier to call just one Dataset Build.
  # Working Dir should be top level with folders ./R, ./input, ./output
  train <- fread('input/train.csv')
  test <- fread('input/test.csv')
  
  # Lets first align the datasets for equal vars to work with.
  y <- train[, QuoteConversion_Flag]
  train[, c('QuoteConversion_Flag') := NULL]
  train[, dset := 0]
  test[, dset := 1]
  
  #Quick check to align data dimensions.
  stopifnot(dim(test)[2] ==dim(train)[2])
  # Join the datasets for simple manipulations.
  bigD <- rbind(train, test)
  rm(list = c('train', 'test'))
  
  ## Data Manipulations 
  # Lets solve the Field10 issue: 1,165 => 1165.
  bigD[, Field10 := as.numeric(gsub(",", "", Field10))]
  
  # First lets work with the dates and remove "Original_Quote_Date" field.
  bigD[, Date := parse_date_time(Original_Quote_Date, "%Y%m%d")]
  bigD[, Original_Quote_Date := NULL]
  
  # Lets get the year, month, day of month, weekday,
  bigD[, year := as.factor(year(Date))]
  bigD[, month := as.factor(month(Date))]
  bigD[, monthday := as.factor(mday(Date))]
  bigD[, weekday := as.factor(wday(Date))]
  # Adding year day - people rebuy the same year?
  bigD[, yearday := as.factor(yday(Date))]
  
  # Fake duration date - How long since the start of the comp (Mon 9 Nov 2015)
  # Taking logs as magnitude is an order difference 175 min 1042 max
  # Checking distributions between train and test proves equal distributions. --> Good to know for CV.
  bigD[, daysDurOrigQuote := as.integer(parse_date_time("2015-11-09", "%Y%m%d") - Date)]
  bigD[, logDaysDurOrigQuote := log(as.integer(parse_date_time("2015-11-09", "%Y%m%d") - Date))]
  bigD[, Date := NULL]
  
  # Lets model temporal effects in time
  # http://www.rochester.edu/College/PSC/signorino/research/Carter_Signorino_2010_PA.pdf
  # Needs more work for anything useful
  #bigD[, logdaysDurOrigQuote2 := log((daysDurOrigQuote)^2 - daysDurOrigQuote)]
  #bigD[, logdaysDurOrigQuote3 := log((daysDurOrigQuote)^3 - daysDurOrigQuote)]
  
  bigD[is.na(bigD)] <- -1
  
  # Count -1's across the data set
  bigD[, CoverageNeg1s := rowSums(.SD == -1), .SDcols = grep("CoverageField", names(bigD))]
  bigD[, SalesNeg1s := rowSums(.SD == -1), .SDcols = grep("SalesField", names(bigD))]
  bigD[, PropertyNeg1s := rowSums(.SD == -1), .SDcols = grep("PropertyField", names(bigD))]
  bigD[, GeoNeg1s := rowSums(.SD == -1), .SDcols = grep("GeographicField", names(bigD))]
  bigD[, PersonalNeg1s := rowSums(.SD == -1), .SDcols = grep("PersonalField", names(bigD))]
  # Finally Total across all.
  bigD[,
       TotalNeg1s := rowSums(.SD),
       .SDcols = c("CoverageNeg1s", "SalesNeg1s", "PropertyNeg1s", "GeoNeg1s", "PersonalNeg1s")]
  
  # Count 0's across the data set
  bigD[, Coverage0s := rowSums(.SD == 0), .SDcols = grep("CoverageField", names(bigD))]
  bigD[, Sales0s := rowSums(.SD == 0), .SDcols = grep("SalesField", names(bigD))]
  bigD[, Property0s := rowSums(.SD == 0), .SDcols = grep("PropertyField", names(bigD))]
  bigD[, Geo0s := rowSums(.SD == 0), .SDcols = grep("GeographicField", names(bigD))]
  bigD[, Personal0s := rowSums(.SD == 0), .SDcols = grep("PersonalField", names(bigD))]
  # Finally Total across all.
  bigD[,
       Total0s := rowSums(.SD),
       .SDcols = c("Coverage0s", "Sales0s", "Property0s", "Geo0s", "Personal0s")]
  
  # Catch factor columns
  fact_cols <- which(lapply(bigD, class) == "character")
  # Map all categoricals into numeric.
  cat("Assuming text variables are categorical & replacing them with numeric ids\n")
  for (f in colnames(bigD)) {
    if (class(bigD[[f]])=="character") {
      print(f)
      levels <- unique(bigD[[f]])
      bigD[[f]] <- as.integer(factor(bigD[[f]], levels=levels))
    }
  }
  
  ## Split files & Export 
  xtrain <- bigD[dset == 0, ]
  xtest <- bigD[dset == 1, ]
  rm(bigD)
  
  xtrain[, dset := NULL]
  xtest[, dset := NULL]
  
  xtrain[, QuoteConversion_Flag := y]
  
  write.csv(xtrain, 'input/xtrain_mp3.csv', row.names = F)
  write.csv(xtest, 'input/xtest_mp3.csv', row.names = F)
  
  rm(xtrain)
  rm(xtest)
  return(cat("MP1 dataset built"))
}

BuildMP3()



## KB set v1: factors mapped to integers ####
# map everything to integers
# read
xtrain <- read_csv("./input/train.csv")
xtest <- read_csv("./input/test.csv")

# process
xtrain[is.na(xtrain)]   <- 0; xtest[is.na(xtest)]   <- 0

# convert categorical ones to numeric
for (f in colnames(xtrain)) {
  if (class(xtrain[[f]])=="character") {
    levels <- unique(c(xtrain[[f]], xtest[[f]]))
    xtrain[[f]] <- as.integer(factor(xtrain[[f]], levels=levels))
    xtest[[f]]  <- as.integer(factor(xtest[[f]],  levels=levels))
  }
  msg(f)
}

# time-based features
xtrain$year <- lubridate::year(xtrain$Original_Quote_Date)
xtest$year <- lubridate::year(xtest$Original_Quote_Date)

xtrain$month <- lubridate::month(xtrain$Original_Quote_Date)
xtest$month <- lubridate::month(xtest$Original_Quote_Date)

xtrain$Original_Quote_Date <- xtest$Original_Quote_Date <- NULL

# store the files
write_csv(xtrain, path = "./input/xtrain_kb1.csv")
write_csv(xtest, path = "./input/xtest_kb1.csv")

## KB set v2: v1 + pairwise factors mapped to integers ####
# - create more factors (pairwise combinations)
# - map everything to integers
# read
xtrain <- read_csv("./input/train.csv")
xtest <- read_csv("./input/test.csv")

# process
xtrain[is.na(xtrain)]   <- 0; xtest[is.na(xtest)]   <- 0

# add combinations of character columns
which_char <- colnames(xtrain)[which(sapply(xtrain, class) == "character")]
xcomb <- combn(length(which_char),2)
for (ff in 1:ncol(xcomb))
{
  i1 <- xcomb[1,ff]; i2 <- xcomb[2,ff]
  xcol1 <- which_char[i1]; xcol2 <- which_char[i2]
  xcol <- paste(xtrain[,xcol1], xtrain[,xcol2], sep = "")
  xname <- paste(xcol1, xcol2, sep = "")
  xtrain[,xname] <- xcol
  xcol <- paste(xtest[,xcol1], xtest[,xcol2], sep = "")
  xtest[,xname] <- xcol
  msg(xname)
  
}
# convert categorical ones to numeric
for (f in colnames(xtrain)) {
  if (class(xtrain[[f]])=="character") {
    levels <- unique(c(xtrain[[f]], xtest[[f]]))
    xtrain[[f]] <- as.integer(factor(xtrain[[f]], levels=levels))
    xtest[[f]]  <- as.integer(factor(xtest[[f]],  levels=levels))
  }
  msg(f)
}

# time-based features
xtrain$year <- lubridate::year(xtrain$Original_Quote_Date)
xtest$year <- lubridate::year(xtest$Original_Quote_Date)

xtrain$month <- lubridate::month(xtrain$Original_Quote_Date)
xtest$month <- lubridate::month(xtest$Original_Quote_Date)

xtrain$Original_Quote_Date <- xtest$Original_Quote_Date <- NULL

# store the files
write_csv(xtrain, path = "./input/xtrain_kb2.csv")
write_csv(xtest, path = "./input/xtest_kb2.csv")

## KB set v3: v1, but factors replaced by response rates ####
# read
xtrain <- read_csv("./input/train.csv")
xtest <- read_csv("./input/test.csv")

# process
xtrain[is.na(xtrain)]   <- 0; xtest[is.na(xtest)]   <- 0

xfold <- read_csv(file = "./input/xfolds.csv")
idFix <- list()
for (ii in 1:10)
{
  idFix[[ii]] <- which(xfold$fold10 == ii)
}
rm(xfold,ii)  

# time-based features => treat as factors
xtrain$year <- as.character(year(xtrain$Original_Quote_Date))
xtest$year <- as.character(year(xtest$Original_Quote_Date))

xtrain$month <- as.character(month(xtrain$Original_Quote_Date))
xtest$month <- as.character(month(xtest$Original_Quote_Date))

mdy <- chron::month.day.year(xtrain$Original_Quote_Date)
xtrain$dow <- as.character(day.of.week(mdy$month, mdy$day, mdy$year))

mdy <- chron::month.day.year(xtest$Original_Quote_Date)
xtest$dow <- as.character(day.of.week(mdy$month, mdy$day, mdy$year))

xtrain$Original_Quote_Date <- xtest$Original_Quote_Date <- NULL

# grab factor variables
factor_vars <- colnames(xtrain)[which(sapply(xtrain, class) == "character")]

# loop over factor variables, create a response rate version for each
for (varname in factor_vars)
{
  # placeholder for the new variable values
  x <- rep(NA, nrow(xtrain))
  for (ii in seq(idFix))
  {
    # separate ~ fold
    idx <- idFix[[ii]]
    x0 <- xtrain[-idx, factor_vars]; x1 <- xtrain[idx, factor_vars]
    y0 <- xtrain$QuoteConversion_Flag[-idx]; y1 <- xtrain$QuoteConversion_Flag[idx]
    # take care of factor lvl mismatches
    x0[,varname] <- factor(as.character(x0[,varname]))
    # fit LMM model
    myForm <- as.formula (paste ("y0 ~ (1|", varname, ")"))
    myLME <- lmer (myForm, x0, REML=FALSE, verbose=F)
    myFixEf <- fixef (myLME); myRanEf <- unlist (ranef (myLME))
    # table to match to the original
    myLMERDF <- data.frame (levelName = as.character(levels(x0[,varname])), myDampVal = myRanEf+myFixEf)
    rownames(myLMERDF) <- NULL
    x[idx] <- myLMERDF[,2][match(xtrain[idx, varname], myLMERDF[,1])]
    x[idx][is.na(x[idx])] <- mean(y0)
  }
  rm(x0,x1,y0,y1, myLME, myLMERDF, myFixEf, myRanEf)
  # add the new variable
  xtrain[,paste(varname, "dmp", sep = "")] <- x
  
  # create the same on test set
  xtrain[,varname] <- factor(as.character(xtrain[,varname]))
  y <- xtrain$QuoteConversion_Flag
  x <- rep(NA, nrow(xtest))
  # fit LMM model
  myForm <- as.formula (paste ("y ~ (1|", varname, ")"))
  myLME <- lmer (myForm, xtrain[,factor_vars], REML=FALSE, verbose=F)
  myFixEf <- fixef (myLME); myRanEf <- unlist (ranef (myLME))
  # table to match to the original
  myLMERDF <- data.frame (levelName = as.character(levels(xtrain[,varname])), myDampVal = myRanEf+myFixEf)
  rownames(myLMERDF) <- NULL
  x <- myLMERDF[,2][match(xtest[, varname], myLMERDF[,1])]
  x[is.na(x)] <- mean(y)
  xtest[,paste(varname, "dmp", sep = "")] <- x
  msg(varname)
}

# drop the factors
ix <- which(colnames(xtrain) %in% factor_vars)
xtrain <- xtrain[,-ix]
ix <- which(colnames(xtest) %in% factor_vars)
xtest <- xtest[,-ix]

# store the files
write_csv(xtrain, path = "./input/xtrain_kb3.csv")
write_csv(xtest, path = "./input/xtest_kb3.csv")

## KB set v4: ~ v1, but create (almost) everything as factors ####
# read
xtrain <- read_csv("./input/train.csv")
xtest <- read_csv("./input/test.csv")

# process
xtrain[is.na(xtrain)]   <- 0; xtest[is.na(xtest)]   <- 0

xfold <- read_csv(file = "./input/xfolds.csv")
idFix <- list()
for (ii in 1:10)
{
  idFix[[ii]] <- which(xfold$fold10 == ii)
}
rm(xfold,ii)  

# time-based features => treat as factors
xtrain$year <- as.character(year(xtrain$Original_Quote_Date))
xtest$year <- as.character(year(xtest$Original_Quote_Date))

xtrain$month <- as.character(month(xtrain$Original_Quote_Date))
xtest$month <- as.character(month(xtest$Original_Quote_Date))

mdy <- chron::month.day.year(xtrain$Original_Quote_Date)
xtrain$dow <- as.character(day.of.week(mdy$month, mdy$day, mdy$year))

mdy <- chron::month.day.year(xtest$Original_Quote_Date)
xtest$dow <- as.character(day.of.week(mdy$month, mdy$day, mdy$year))

xtrain$Original_Quote_Date <- xtest$Original_Quote_Date <- NULL

# count number of distinct values
nof_vals <- sapply(xtrain, function(s) nlevels(factor(s)))

# grab factor variables
factor_vars <- colnames(xtrain)[which( (nof_vals < 100) & (nof_vals > 1))]
factor_vars <- setdiff(factor_vars, "QuoteConversion_Flag")

# loop over factor variables, create a response rate version for each
for (varname in factor_vars)
{
  # placeholder for the new variable values
  x <- rep(NA, nrow(xtrain))
  for (ii in seq(idFix))
  {
    # separate ~ fold
    idx <- idFix[[ii]]
    x0 <- xtrain[-idx, factor_vars]; x1 <- xtrain[idx, factor_vars]
    y0 <- xtrain$QuoteConversion_Flag[-idx]; y1 <- xtrain$QuoteConversion_Flag[idx]
    # take care of factor lvl mismatches
    x0[,varname] <- factor(as.character(x0[,varname]))
    # fit LMM model
    myForm <- as.formula (paste ("y0 ~ (1|", varname, ")"))
    myLME <- lmer (myForm, x0, REML=FALSE, verbose=F)
    myFixEf <- fixef (myLME); myRanEf <- unlist (ranef (myLME))
    # table to match to the original
    myLMERDF <- data.frame (levelName = as.character(levels(x0[,varname])), myDampVal = myRanEf+myFixEf)
    rownames(myLMERDF) <- NULL
    x[idx] <- myLMERDF[,2][match(xtrain[idx, varname], myLMERDF[,1])]
    x[idx][is.na(x[idx])] <- mean(y0)
  }
  rm(x0,x1,y0,y1, myLME, myLMERDF, myFixEf, myRanEf)
  # add the new variable
  xtrain[,paste(varname, "dmp", sep = "")] <- x
  
  # create the same on test set
  xtrain[,varname] <- factor(as.character(xtrain[,varname]))
  y <- xtrain$QuoteConversion_Flag
  x <- rep(NA, nrow(xtest))
  # fit LMM model
  myForm <- as.formula (paste ("y ~ (1|", varname, ")"))
  myLME <- lmer (myForm, xtrain[,factor_vars], REML=FALSE, verbose=F)
  myFixEf <- fixef (myLME); myRanEf <- unlist (ranef (myLME))
  # table to match to the original
  myLMERDF <- data.frame (levelName = as.character(levels(xtrain[,varname])), myDampVal = myRanEf+myFixEf)
  rownames(myLMERDF) <- NULL
  x <- myLMERDF[,2][match(xtest[, varname], myLMERDF[,1])]
  x[is.na(x)] <- mean(y)
  xtest[,paste(varname, "dmp", sep = "")] <- x
  msg(varname)
}

# drop the original factors
for (xn in factor_vars)
{
  xtrain[,xn] <- NULL
  xtest[,xn] <- NULL
}

# drop constant columns 
xsd <- apply(xtrain,2,sd); xnames <- names(which(xsd == 0))
for (xn in xnames)
{
  xtrain[,xn] <- NULL
  xtest[,xn] <- NULL
}
# store the files
write_csv(xtrain, path = "./input/xtrain_kb4.csv")
write_csv(xtest, path = "./input/xtest_kb4.csv")

## KB set v5: ~v2 + treat almost everything as factors ####
# read
xtrain <- read_csv("./input/train.csv")
xtest <- read_csv("./input/test.csv")

# process
xtrain[is.na(xtrain)]   <- 0; xtest[is.na(xtest)]   <- 0

# time-based features => treat as character
xtrain$year <- as.character(year(xtrain$Original_Quote_Date))
xtest$year <- as.character(year(xtest$Original_Quote_Date))

xtrain$month <- as.character(month(xtrain$Original_Quote_Date))
xtest$month <- as.character(month(xtest$Original_Quote_Date))

mdy <- chron::month.day.year(xtrain$Original_Quote_Date)
xtrain$dow <- as.character(day.of.week(mdy$month, mdy$day, mdy$year))

mdy <- chron::month.day.year(xtest$Original_Quote_Date)
xtest$dow <- as.character(day.of.week(mdy$month, mdy$day, mdy$year))

xtrain$Original_Quote_Date <- xtest$Original_Quote_Date <- NULL


# add combinations of character columns
which_char <- colnames(xtrain)[which(sapply(xtrain, class) == "character")]
xcomb <- combn(length(which_char),2)
for (ff in 1:ncol(xcomb))
{
  i1 <- xcomb[1,ff]; i2 <- xcomb[2,ff]
  xcol1 <- which_char[i1]; xcol2 <- which_char[i2]
  xcol <- paste(xtrain[,xcol1], xtrain[,xcol2], sep = "")
  xname <- paste(xcol1, xcol2, sep = "")
  xtrain[,xname] <- xcol
  xcol <- paste(xtest[,xcol1], xtest[,xcol2], sep = "")
  xtest[,xname] <- xcol
  msg(xname)
  
}

xfold <- read_csv(file = "./input/xfolds.csv")
idFix <- list()
for (ii in 1:10)
{
  idFix[[ii]] <- which(xfold$fold10 == ii)
}
rm(xfold,ii)  


# count number of distinct values
nof_vals <- sapply(xtrain, function(s) nlevels(factor(s)))

# grab factor variables
factor_vars <- colnames(xtrain)[which( (nof_vals < 100) & (nof_vals > 1))]
factor_vars <- setdiff(factor_vars, "QuoteConversion_Flag")

# loop over factor variables, create a response rate version for each
for (varname in factor_vars)
{
  # placeholder for the new variable values
  x <- rep(NA, nrow(xtrain))
  for (ii in seq(idFix))
  {
    # separate ~ fold
    idx <- idFix[[ii]]
    x0 <- xtrain[-idx, factor_vars]; x1 <- xtrain[idx, factor_vars]
    y0 <- xtrain$QuoteConversion_Flag[-idx]; y1 <- xtrain$QuoteConversion_Flag[idx]
    # take care of factor lvl mismatches
    x0[,varname] <- factor(as.character(x0[,varname]))
    # fit LMM model
    myForm <- as.formula (paste ("y0 ~ (1|", varname, ")"))
    myLME <- lmer (myForm, x0, REML=FALSE, verbose=F)
    myFixEf <- fixef (myLME); myRanEf <- unlist (ranef (myLME))
    # table to match to the original
    myLMERDF <- data.frame (levelName = as.character(levels(x0[,varname])), myDampVal = myRanEf+myFixEf)
    rownames(myLMERDF) <- NULL
    x[idx] <- myLMERDF[,2][match(xtrain[idx, varname], myLMERDF[,1])]
    x[idx][is.na(x[idx])] <- mean(y0)
  }
  rm(x0,x1,y0,y1, myLME, myLMERDF, myFixEf, myRanEf)
  # add the new variable
  xtrain[,paste(varname, "dmp", sep = "")] <- x
  
  # create the same on test set
  xtrain[,varname] <- factor(as.character(xtrain[,varname]))
  y <- xtrain$QuoteConversion_Flag
  x <- rep(NA, nrow(xtest))
  # fit LMM model
  myForm <- as.formula (paste ("y ~ (1|", varname, ")"))
  myLME <- lmer (myForm, xtrain[,factor_vars], REML=FALSE, verbose=F)
  myFixEf <- fixef (myLME); myRanEf <- unlist (ranef (myLME))
  # table to match to the original
  myLMERDF <- data.frame (levelName = as.character(levels(xtrain[,varname])), myDampVal = myRanEf+myFixEf)
  rownames(myLMERDF) <- NULL
  x <- myLMERDF[,2][match(xtest[, varname], myLMERDF[,1])]
  x[is.na(x)] <- mean(y)
  xtest[,paste(varname, "dmp", sep = "")] <- x
  msg(varname)
}

# drop the original factors
for (xn in factor_vars)
{
  xtrain[,xn] <- NULL
  xtest[,xn] <- NULL
}

# drop constant columns 
xsd <- apply(xtrain,2,sd); xnames <- names(which(xsd == 0))
for (xn in xnames)
{
  xtrain[,xn] <- NULL
  xtest[,xn] <- NULL
}

# quick fix re-run: take care of the factors missed the first 
# time around due to larger number of unique values
factor_vars <- names(which(sapply(xtrain, class) == "character"))
# loop over factor variables, create a response rate version for each
for (varname in factor_vars)
{
  # placeholder for the new variable values
  x <- rep(NA, nrow(xtrain))
  for (ii in seq(idFix))
  {
    # separate ~ fold
    idx <- idFix[[ii]]
    x0 <- xtrain[-idx, factor_vars]; x1 <- xtrain[idx, factor_vars]
    y0 <- xtrain$QuoteConversion_Flag[-idx]; y1 <- xtrain$QuoteConversion_Flag[idx]
    # take care of factor lvl mismatches
    x0[,varname] <- factor(as.character(x0[,varname]))
    # fit LMM model
    myForm <- as.formula (paste ("y0 ~ (1|", varname, ")"))
    myLME <- lmer (myForm, x0, REML=FALSE, verbose=F)
    myFixEf <- fixef (myLME); myRanEf <- unlist (ranef (myLME))
    # table to match to the original
    myLMERDF <- data.frame (levelName = as.character(levels(x0[,varname])), myDampVal = myRanEf+myFixEf)
    rownames(myLMERDF) <- NULL
    x[idx] <- myLMERDF[,2][match(xtrain[idx, varname], myLMERDF[,1])]
    x[idx][is.na(x[idx])] <- mean(y0)
  }
  rm(x0,x1,y0,y1, myLME, myLMERDF, myFixEf, myRanEf)
  # add the new variable
  xtrain[,paste(varname, "dmp", sep = "")] <- x
  
  # create the same on test set
  xtrain[,varname] <- factor(as.character(xtrain[,varname]))
  y <- xtrain$QuoteConversion_Flag
  x <- rep(NA, nrow(xtest))
  # fit LMM model
  myForm <- as.formula (paste ("y ~ (1|", varname, ")"))
  myLME <- lmer (myForm, xtrain[,factor_vars], REML=FALSE, verbose=F)
  myFixEf <- fixef (myLME); myRanEf <- unlist (ranef (myLME))
  # table to match to the original
  myLMERDF <- data.frame (levelName = as.character(levels(xtrain[,varname])), myDampVal = myRanEf+myFixEf)
  rownames(myLMERDF) <- NULL
  x <- myLMERDF[,2][match(xtest[, varname], myLMERDF[,1])]
  x[is.na(x)] <- mean(y)
  xtest[,paste(varname, "dmp", sep = "")] <- x
  
  # clean up 
  # xtrain[,varname] <- NULL;   xtest[,varname] <- NULL
  msg(varname)
}

# clean up 
for (xn in factor_vars)
{
  xtrain[,xn] <- NULL
  xtest[,xn] <- NULL
}

xtrain$Field6PersonalField16 <- xtest$Field6PersonalField16 <- NULL

# store the files
write_csv(xtrain, path = "./input/xtrain_kb5.csv")
write_csv(xtest, path = "./input/xtest_kb5.csv")

## KB set v6: ~kmeans based on v4  ####
xtrain <- read_csv("./input/xtrain_kb4.csv")
xtest <- read_csv("./input/xtest_kb4.csv")

## create kmeans-based dataset 
xfolds <- read_csv("./input/xfolds.csv")
isValid <- which(xfolds$valid == 1)
y <- xtrain$QuoteConversion_Flag; xtrain$QuoteConversion_Flag <- NULL
xtrain$SalesField8 <- xtest$SalesField8 <- NULL
train_QuoteNumber <- xtrain$QuoteNumber
test_QuoteNumber <- xtest$QuoteNumber
xtrain$QuoteNumber <- xtest$QuoteNumber <- NULL


# map to distances from kmeans clusters
nof_centers <- 50
km0 <- kmeans(xtrain, centers = nof_centers)
dist1 <- array(0, c(nrow(xtrain), nof_centers))
for (ii in 1:nof_centers)
{
  dist1[,ii] <- apply(xtrain,1,function(s) sd(s - km0$centers[ii,]))
  msg(ii)
}
dist2 <- array(0, c(nrow(xtest), nof_centers))
for (ii in 1:nof_centers)
{
  dist2[,ii] <- apply(xtest,1,function(s) sd(s - km0$centers[ii,]))
  msg(ii)
}

# storage
dist1 <- data.frame(dist1)
dist2 <- data.frame(dist2)
dist1$QuoteConversion_Flag <- factor(y)
dist1$QuoteNumber <- train_QuoteNumber
dist2$QuoteNumber <- test_QuoteNumber

write_csv(dist1, "./input/xtrain_kb6.csv")
write_csv(dist2, "./input/xtest_kb6.csv")

## KB set v7: kmeans based on v5  ####
xtrain <- read_csv("./input/xtrain_kb5.csv")
xtest <- read_csv("./input/xtest_kb5.csv")

## create kmeans-based dataset
xfolds <- read_csv("./input/xfolds.csv")
isValid <- which(xfolds$valid == 1)
y <- xtrain$QuoteConversion_Flag; xtrain$QuoteConversion_Flag <- NULL
xtrain$SalesField8 <- xtest$SalesField8 <- NULL
train_QuoteNumber <- xtrain$QuoteNumber
test_QuoteNumber <- xtest$QuoteNumber
xtrain$QuoteNumber <- xtest$QuoteNumber <- NULL


# map to distances from kmeans clusters
nof_centers <- 100
km0 <- kmeans(xtrain, centers = nof_centers)
dist1 <- array(0, c(nrow(xtrain), nof_centers))
for (ii in 1:nof_centers)
{
  dist1[,ii] <- apply(xtrain,1,function(s) sd(s - km0$centers[ii,]))
  msg(ii)
}
dist2 <- array(0, c(nrow(xtest), nof_centers))
for (ii in 1:nof_centers)
{
  dist2[,ii] <- apply(xtest,1,function(s) sd(s - km0$centers[ii,]))
  msg(ii)
}

# storage
dist1 <- data.frame(dist1)
dist2 <- data.frame(dist2)
dist1$QuoteConversion_Flag <- factor(y)
dist1$QuoteNumber <- train_QuoteNumber
dist2$QuoteNumber <- test_QuoteNumber

write_csv(dist1, "./input/xtrain_kb7.csv")
write_csv(dist2, "./input/xtest_kb7.csv")
