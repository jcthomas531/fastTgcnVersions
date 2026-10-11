#here::i_am("rugaeDetect/machShop2.R")
library(here)
library(MachineShop)
library(dplyr)
library(doSNOW)
registerDoSNOW(makeCluster(64))
library(data.table)
library(stringr)


seed = 826
set.seed(seed)


#testing with scenario:
#density reduction: remesh
#density: 8500
#normal alpha: 0.012
#descriptor mult: 1.3
#feature: PFH

###################
#helper functions


#make a list where each entry has a data frame for a single hold out patient
#and a data frame for the rest of the patients
patSplitter <- function(x) {

  #unique patient numbers
  uniqPatNums <- str_extract(x$patId, "[0-9]{3}") |>
    unique()

  #split the data into a list on patient id
  allSplit <- split(x, x$patId)
  
  #loop thru each patient number and create a list that has a df
  #that contains that patients pre and post data and a df that
  #contains all other patient pre and post data
  leftOutList <- vector(mode = "list", length = length(uniqPatNums))
  names(leftOutList) <- paste0("pat", uniqPatNums)
  for (i in 1:length(uniqPatNums)) {
    pati <- uniqPatNums[i]
    logicInd <- str_detect(names(allSplit), pati)
    #extract postions of lists with pati and non pati data
    patiList <- allSplit[logicInd]
    nonPatiList <- allSplit[!logicInd]
    #concatenate lists
    patiDf <- do.call(rbind, patiList)
    nonPatiDf <- do.call(rbind, nonPatiList)
    #put into list
    leftOutList[[i]] <- list(pat = patiDf, nonPat = nonPatiDf)
  }

  return(leftOutList)
}


#intersection over union metric
iouPoint <- MLMetric(
  function(observed, predicted, ...) {
    #logicals for classification
    pred1 <- predicted == 1
    obs1 <- observed == 1
    #creating metric
    numer <- sum(pred1 & obs1)
    denom <- sum(pred1 | obs1)
    return(numer/denom)
  },
  name = "iouPoint",
  label = "Intersection over union for point classification",
  maximize = TRUE
)


###################
print(getwd())
#read in data
# prePath <- "../../../../../Shared/gb_lss/Thomas/iowaExpTest/rugaePipeOptim/combinedDescriptors/pre/remesh8500/norm0.012_desc1.3/combinedDesc.csv"
# postPath <- "../../../../../Shared/gb_lss/Thomas/iowaExpTest/rugaePipeOptim/combinedDescriptors/post/remesh8500/norm0.012_desc1.3/combinedDesc.csv"
prePath <- "K:/iowaExpTest/rugaePipeOptim/combinedDescriptors/pre/remesh8500/norm0.012_desc1.3/combinedDesc.csv"
postPath <- "K:/iowaExpTest/rugaePipeOptim/combinedDescriptors/post/remesh8500/norm0.012_desc1.3/combinedDesc.csv"
#read in only columns needed for current model
#both the pre and the post csvs should have the same column names so arbitrarily selecting from pre
allCols <- names(fread(prePath, nrows = 0))
pfhCols <- allCols[startsWith(allCols, "pfh")]
#read
preDat <- fread(
  prePath, 
  select = c("patId", "label", "x", "y", "z", "nx", "ny", "nz", pfhCols)
)
postDat <- fread(
  postPath, 
  select = c("patId", "label", "x", "y", "z", "nx", "ny", "nz", pfhCols)
)
#combine pre and post
dat <- rbind(
  preDat |>
    mutate(period = "pre"),
  postDat |>
    mutate(period = "post")
)


#FOR TESTING ONLY
#SUBSETTING DATA TO MORE REASONABLE SIZE
# dat <- dat |>
#   slice_sample(n = 10000)
dat <- dat |>
  filter(patId %in% c("pat001Pre", "pat007Pre", "pat012Post"))
###############


#number of patients
nPats =  length(unique(dat$patId))

#basic data manipulations
dat <- dat |>
  mutate(label = factor(label, levels = c(0,1)))

#create formula for model
covariates <- setdiff(names(dat), c("label", "patId", "period"))
form <- reformulate(covariates, response = "label")

#set up data for rotating patient validation
splitDat <- patSplitter(dat)

#loop thru each patient being held out for validation
outList <- vector(mode = "list", length = length(splitDat))
for (i in 1:length(splitDat)) {
  spliti <- splitDat[[i]]
  #training and test data
  ttDat <- spliti$nonPat
  #validation data
  vDat <- spliti$pat

  #model frame
  mf <- ModelFrame(
    formula = form,
    data = ttDat,
    strata = ttDat$label
  )

  #cross validation
  #to be adjusted later
  cv <- CVControl(folds = 5, repeats = 2, seed = seed)

  #tuning
  #to be adjusted later
  #XGBModel is just the generic framework for all extreme gradient boosting
  #XGBTreeModel is the conventional implementation
  xgbTune <- TunedModel(
    object = XGBTreeModel,
    grid = c(
      nrounds = 2,
      eta = 1,
      gamma = 1,
      max_depth = 1
    )
  )
  
  #model specification
  xgbSpec <- ModelSpecification(
    input = mf,
    model = xgbTune,
    control = cv,
    metrics = c(
      "iouPoint",
      "roc_auc",
      "auc", 
      "accuracy",
      "roc_index", 
      "fnr", 
      "fpr", 
      "tnr", 
      "tpr", 
      "pr_auc", 
      "precision", 
      "recall",
      "sensitivity",
      "specificity"
    )
  )
  
  #fitting
  xgbFit <- fit(xgbSpec, verbose = TRUE)
  #model object
  xgbMod <- as.MLModel(xgbFit)
  
  #response for validation
  valResp <- response(xgbFit, newdata = vDat)
  #predict on validation
  valPred <- predict(xgbFit, newdata = vDat)
  #validation performance
  valPerf <- performance(
    valResp,
    valPred,
    metrics = c(
      "iouPoint",
      "roc_auc",
      "auc", 
      "accuracy",
      "roc_index", 
      "fnr", 
      "fpr", 
      "tnr", 
      "tpr", 
      "pr_auc", 
      "precision", 
      "recall",
      "sensitivity",
      "specificity"
    )
  )

  #
  outList[[i]] <- list(
    performance = valPerf
  )
  

  print(i)
}


save(outList, file = "performance.RData")
