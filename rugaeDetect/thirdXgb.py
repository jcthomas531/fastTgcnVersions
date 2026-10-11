import xgboost as xgb
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score
import numpy as np
rSeed = 826

#############################
#helper functions

#make a dictionary where each entry has a data frame for a single hold out patient
#and a data frame for the rest of the patients
def patSplitter(x):
    # Unique patient numbers
    uniqPatNums = x["patId"].str.extract(r"([0-9]{3})")[0].unique()

    # Create output dictionary
    leftOutList = {}

    for pati in uniqPatNums:
        # Identify rows belonging to this patient number
        logicInd = x["patId"].str.contains(pati)

        # Patient and non-patient data frames
        patiDf = x[logicInd].copy()
        nonPatiDf = x[~logicInd].copy()

        # Store them
        leftOutList[f"pat{pati}"] = {
            "pat": patiDf,
            "nonPat": nonPatiDf
        }

    return leftOutList


def iouPoint(observed, predicted):
    pred1 = predicted == 1
    obs1 = observed == 1
    numer = np.sum(pred1 & obs1)
    denom = np.sum(pred1 | obs1)
    return numer/denom

iouScorer = make_scorer(iouPoint)

##############################


#pfh data only

#read in data
datPreFull = pd.read_csv("K:/iowaExpTest/rugaePipeOptim/combinedDescriptors/pre/remesh8500/norm0.012_desc1.3/combinedDesc.csv")
datPre = datPreFull.loc[:, datPreFull.columns.str.startswith("pfh") | datPreFull.columns.isin(["label", "patId", "nx", "ny", "z", "x", "nz", "y"])]

datPostFull = pd.read_csv("K:/iowaExpTest/rugaePipeOptim/combinedDescriptors/post/remesh8500/norm0.012_desc1.3/combinedDesc.csv")
datPost = datPostFull.loc[:, datPostFull.columns.str.startswith("pfh") | datPostFull.columns.isin(["label", "patId", "nx", "ny", "z", "x", "nz", "y"])]

dat = pd.concat([datPre, datPost], axis = 0, ignore_index=True)

#rotating patient hold out
splitDat = patSplitter(dat)

#testing with a single iteration, will move to looping after


#test/training and validation data
ttDat = splitDat["pat001"]["nonPat"]
vDat = splitDat["pat001"]["pat"]

#separate x and y
ttDatX = ttDat.drop(columns = ["label", "patId"])
ttDatY = ttDat["label"]
vDatX = vDat.drop(columns = ["label", "patId"])
vDatY = vDat["label"]


#set up xgboost model
modXgb = xgb.XGBClassifier(
    objective = "binary:logistic",
    random_state  = rSeed
    )

#parameter grid
#https://www.xgboost.org/code/xgboost-hyperparameter-tuning-guide/?utm_source=chatgpt.com
paramGrid = {
    "n_estimators": [50, 200], #[50, 100, 200],
    "learning_rate": [0.05, 0.2], #[0.05, 0.1, 0.2],
    "gamma": [0, 1], #[0, 0.1, 1],
    "max_depth": [3, 7] #[3, 5, 7]
}

#metrics to look at
from sklearn.metrics import accuracy_score, roc_auc_score
# scoring = {
#     "iou": iouScorer,
#     "rocAuc": "roc_auc",
#     "accuracy": "accuracy"
# }

# #set up cross validation
cv = StratifiedKFold(
    n_splits=3,
    shuffle = True,
    random_state=rSeed
)

#old, simple cv not tuning
# #perform cross validation
# cvScores = cross_val_score(
#     estimator=modXgb,
#     X=ttDatX,
#     y=ttDatY,
#     cv=cv,
#     scoring=iouScorer
# )
# #look at auc
# cvScores
# cvScores.mean()
# #fit model
# modXgb.fit(ttDatX, ttDatY)
# #predict
# valPred = modXgb.predict(vDatX)
# valPredProb = modXgb.predict_proba(vDatX)[:,1]



gridSearch = GridSearchCV(
    estimator=modXgb,
    param_grid=paramGrid,
    scoring=iouScorer,
    cv=cv,
    refit=True, 
    n_jobs=12, #cores
    verbose=2
)
gridSearch.fit(
    X=ttDatX,
    y=ttDatY
)
pd.DataFrame(gridSearch.cv_results_).sort_values("rank_test_score")
bestXgb = gridSearch.best_estimator_




#predict
valPred = bestXgb.predict(vDatX)
valPredProb = bestXgb.predict_proba(vDatX)[:,1]




valRocAuc = roc_auc_score(
    y_true=vDatY,
    y_score=valPredProb
)
valRocAuc
iouPoint(observed = vDatY, predicted = valPred)
