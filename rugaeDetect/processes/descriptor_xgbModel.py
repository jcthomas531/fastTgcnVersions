import sys
sys.path.append("tools/rugaeDetect")
from patSplitter import patSplitter
import iouPoint as iouP

import re
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import pickle
import time
rSeed = 826

#testing
# descPrefix = "^pfh_"
# resultsOut = "K:/iowaExpTest/testDir/results_test.csv"
# summaryOut = "K:/iowaExpTest/testDir/summary_test.pkl"
# inFiles = [
#     "K:/iowaExpTest/rugaePipeOptim/combinedDescriptors/pre/remesh8500/norm0.012_desc1.3/combinedDesc.csv",
#     "K:/iowaExpTest/rugaePipeOptim/combinedDescriptors/post/remesh8500/norm0.012_desc1.3/combinedDesc.csv"
# ]

#variables from snakemake
descPrefix = sys.argv[1] #regular expression
resultsOut = sys.argv[2]
summaryOut = sys.argv[3]
inFiles = sys.argv[4:]

#get infiles
prePath = inFiles[0]
postPath = inFiles[1]

#read in data, only the columns we need
colKeepLogical = lambda col: bool(re.match(descPrefix, col)) or col in ["label", "patId", "nx", "ny", "z", "x", "nz", "y"]
datPre = pd.read_csv(prePath, usecols=colKeepLogical)
datPost = pd.read_csv(postPath, usecols=colKeepLogical)
dat = pd.concat([datPre, datPost], axis = 0, ignore_index=True)
print("Data read")


#applying a very conservative low variance filter
feats = dat.loc[:, dat.columns.str.match(descPrefix)]
featVar = feats.var()
dat = dat.drop(columns = featVar[featVar < 0.0001].index)


#unique patients ids
uniqPats = dat["patId"].str.extract(r"(pat[0-9]{3})")[0].unique()

#rotating hold out patient data frames
splitDat = patSplitter(dat)

#loop thru hold out groups
print("Begin model fitting")
resDict = {}
for i in range(len(uniqPats)):
    pati = uniqPats[i]

    #test/training and validation data
    ttDat = splitDat[pati]["nonPat"]
    vDat = splitDat[pati]["pat"]

    #separate x and y
    ttDatX = ttDat.drop(columns = ["label", "patId"])
    ttDatY = ttDat["label"]
    vDatX = vDat.drop(columns = ["label", "patId"])
    vDatY = vDat["label"]

    #set up xgboost model
    modXgb = xgb.XGBClassifier(
        objective = "binary:logistic",
        random_state  = rSeed,
        n_jobs=1
    )

    #parameter grid
    #https://www.xgboost.org/code/xgboost-hyperparameter-tuning-guide/?utm_source=chatgpt.com
    paramGrid = {
        "n_estimators": [300, 600, 900],
        "learning_rate": [0.15, 0.3, 0.45],
        "gamma": [0.7,  1, 3],
        "max_depth": [6, 9, 12]
    }

    #metrics to look at
    scoring = {
        "iou": iouP.iouScorer,
        "roc_auc": "roc_auc",
        "accuracy": "accuracy"
    }

    #set up cross validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle = True,
        random_state=rSeed
    )

    #set up grid search
    gridSearch = GridSearchCV(
        estimator=modXgb,
        param_grid=paramGrid,
        scoring=scoring,
        cv=cv,
        refit="iou", #fit the best model based on iou
        n_jobs=16, #cores
        verbose=2
    )

    #perform grid search and time
    startTime = time.time()
    gridRes_xgb = gridSearch.fit(
        X=ttDatX,
        y=ttDatY
    )
    endTime = time.time()
    cvTime = endTime - startTime

    #extract best model
    fitXgb = gridRes_xgb.best_estimator_

    #extract the average IoU for the cross validation of best model (essentially test IoU)
    testIou = gridRes_xgb.best_score_
    #get in-sample iou for best model
    insamplePred = fitXgb.predict(ttDatX)
    insampleIou = iouP.iouPoint(observed = ttDatY, predicted = insamplePred)

    #extract best model params
    paramsXgb = gridRes_xgb.best_params_

    #predict on validation set
    valPred = fitXgb.predict(vDatX)
    valPredProb = fitXgb.predict_proba(vDatX)[:,1]

    #validation metrics
    valRocAuc = roc_auc_score(y_true = vDatY, y_score = valPredProb)
    valIou = iouP.iouPoint(observed = vDatY, predicted = valPred)
    valAcc = accuracy_score(y_true = vDatY, y_pred = valPred)
    valDict = {
        "iou": valIou, "rocAuc": valRocAuc, "acc": valAcc, "cvTime": cvTime,
        "testIou": testIou, "insampleIou": insampleIou
    } 

    #add metrics to output
    resDict[pati] = paramsXgb | valDict

    print("Finished xgb for " + pati)
print("Model fitting completed")

#results as a table
resultsDat = pd.DataFrame.from_dict(resDict, orient = "index")
resultsDat.index.name = "patId"
resultsDat = resultsDat.reset_index()

#performance summary
summaryDict = {
    "maxIou": np.max(resultsDat["iou"]),
    "meanIou": np.mean(resultsDat["iou"]),
    "sdIou": np.std(resultsDat["iou"]),
    "maxRocAuc": np.max(resultsDat["rocAuc"]),
    "meanRocAuc": np.mean(resultsDat["rocAuc"]),
    "sdRocAuc": np.std(resultsDat["rocAuc"]),
    "maxAcc": np.max(resultsDat["acc"]),
    "meanAcc": np.mean(resultsDat["acc"]),
    "sdAcc": np.std(resultsDat["acc"]),
    "maxTime": np.max(resultsDat["cvTime"]),
    "meanTime": np.mean(resultsDat["cvTime"]),
    "sdTime": np.std(resultsDat["cvTime"])
}

#exports
#full results
resultsDat.to_csv(resultsOut, index = False)
#summary
filePath = open(summaryOut, "wb") 
pickle.dump(obj = summaryDict,
            file = filePath)
filePath.close()
#can be read in like: 
# with open(summaryOut, "rb") as f:
#     obj = pickle.load(f)

print("Finished")
