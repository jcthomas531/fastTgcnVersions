from xlrd import colname
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics
rSeed = 826


#data set 1
inDir = "K:/iowaExpTest/localDescriptors/rugAnnotForm_cSOriMastRemesh_localDescr/"
d1 = pd.read_csv(inDir + "postLabeledCsv/pat001Post_localDescrLabel.csv")
d1["patId"] = "pat001Post"
d2 = pd.read_csv(inDir + "postLabeledCsv/pat004Post_localDescrLabel.csv")
d2["patId"] = "pat004Post"
d3 = pd.read_csv(inDir + "postLabeledCsv/pat007Post_localDescrLabel.csv")
d3["patId"] = "pat007Post"
d4 = pd.read_csv(inDir + "postLabeledCsv/pat008Post_localDescrLabel.csv")
d4["patId"] = "pat008Post"
d5 = pd.read_csv(inDir + "postLabeledCsv/pat012Post_localDescrLabel.csv")
d5["patId"] = "pat0012Post"
d6 = pd.read_csv(inDir + "postLabeledCsv/pat013Post_localDescrLabel.csv")
d6["patId"] = "pat013Post"
d7 = pd.read_csv(inDir + "postLabeledCsv/pat014Post_localDescrLabel.csv")
d7["patId"] = "pat014Post"
d8 = pd.read_csv(inDir + "postLabeledCsv/pat015Post_localDescrLabel.csv")
d8["patId"] = "pat015Post"
dFirst = pd.concat([d1, d2, d3, d4, d5, d6, d7, d8])


#data set 2
dSecondFull = pd.read_csv("K:/iowaExpTest/rugaePipeOptim/combinedDescriptors/post/remesh8500/norm0.02_desc1.3/combinedDesc.csv")
dSecond = dSecondFull.loc[:, dSecondFull.columns.str.startswith("pfh") | dSecondFull.columns.isin(["label", "patId", "nx", "ny", "z", "x", "nz", "y"])]


#function for a simple train test split then a validation
def trainTestValid(dat):
    rSeed = 826
    X = dat[dat["patId"] != "pat007Post"].drop(columns = ["label", "patId"]).copy()
    y = dat[dat["patId"] != "pat007Post"]["label"].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rSeed, stratify=y)
    #check test train stratification
    trainRatio = sum(y_train)/len(y_train)
    testRatio = sum(y_test)/len(y_test)
    print(trainRatio)
    print(testRatio)
    #model
    clf_xgb = xgb.XGBClassifier(
        objective = "binary:logistic",
        seed = rSeed,
        early_stopping_rounds = 10,
        eval_metric = "aucpr"  
    )
    clf_xgb.fit(
        X_train, 
        y_train, 
        verbose = True,
        eval_set = [(X_test, y_test)]
    )
    #validation data
    XValid = dat[dat["patId"] == "pat007Post"].drop(columns = ["label", "patId"]).copy()
    yValid = dat[dat["patId"] == "pat007Post"][["label"]].copy()
    predNew = clf_xgb.predict(XValid)
    predProbNew = clf_xgb.predict_proba(XValid)
    predProb0New = predProbNew[:,0]
    predProb1New = predProbNew[:,1]
    newDf = pd.DataFrame(yValid)
    newDf["pred"] = predNew
    newDf["predProb0"] = predProb0New
    newDf["predProb1"] = predProb1New
    fpr, tpr, thresholds = metrics.roc_curve(newDf["label"], newDf["predProb1"])
    return metrics.auc(fpr, tpr)

trainTestValid(dat = dFirst)
trainTestValid(dat = dSecond)



#function for a leave one out scheme, leaving out patient and using it as test data
def leaveOneOut(dat):
    rSeed = 826
    trainDat = dat[dat["patId"] != "pat007Post"]
    testDat = dat[dat["patId"] == "pat007Post"]
    X_train = trainDat.drop(columns = ["label", "patId"]).copy()
    X_test = testDat.drop(columns = ["label", "patId"]).copy()
    y_train = trainDat["label"].copy()
    y_test = testDat["label"].copy()
    #model
    clf_xgb = xgb.XGBClassifier(
        objective = "binary:logistic",
        seed = rSeed,
        early_stopping_rounds = 10,
        eval_metric = "aucpr"  
    )
    clf_xgb.fit(
        X_train, 
        y_train, 
        verbose = True,
        eval_set = [(X_test, y_test)]
    )
    #test performance
    predNew = clf_xgb.predict(X_test)
    predProbNew = clf_xgb.predict_proba(X_test)
    predProb0New = predProbNew[:,0]
    predProb1New = predProbNew[:,1]
    newDf = pd.DataFrame(y_test)
    newDf["pred"] = predNew
    newDf["predProb0"] = predProb0New
    newDf["predProb1"] = predProb1New
    fpr, tpr, thresholds = metrics.roc_curve(newDf["label"], newDf["predProb1"])
    return metrics.auc(fpr, tpr)


leaveOneOut(dat = dFirst)
leaveOneOut(dat = dSecond)
