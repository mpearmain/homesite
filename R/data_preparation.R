# Data Creation Script.
# Aim is to produce 3 files:
# Train
# Validation (for blending later) -> also for basic CV (LB feedback looks reasonable, i.e stable data)
# Test
#
# These output can then be used in all subsequent models, and the prediction can ben ensembled from
# the validation using ridge regression (say).

library(data.table)
library(caret)
library(stringr)
library(lubridate)

set.seed(260681)

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

################################ Data Manipulations ##############################################
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

# plotting Sales field 


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

# Next big step is to make binary values of the factor cols, and response rates
# for the factor cols.


############################## Split files & Export ##########################################
xtrain <- bigD[dset == 0, ]
xtest <- bigD[dset == 1, ]
rm(bigD)

xtrain[, dset := NULL]
xtest[, dset := NULL]

xtrain[, QuoteConversion_Flag := y]


write.csv(xtrain, 'input/xtrain_mp1.csv', row.names = F)
write.csv(xtest, 'input/xtest_mp1.csv', row.names = F)




