import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    confusion_matrix,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from xlrd import colname

rSeed = 826

#testing
inPath = "K:/iowaExpTest/remeshDescriptorTesting/labeledCsv/pat004Pre_remesh50000_labeld.csv"
newPath = "K:/iowaExpTest/remeshDescriptorTesting/testMeshes/pat007Pre_remesh8500_labeld.csv"

#bring in arguements from snakemake
#inPath = sys.argv[1]

#read in data
dat = pd.read_csv(inPath)
dat["pat"] = "004"

#following this guide
#https://xgboost.readthedocs.io/en/stable/python/python_intro.html





# https://www.youtube.com/watch?v=aLOQD66Sj0g



# https://www.youtube.com/watch?v=GrJP9FLV3FE

#set up data
X = dat.drop(columns=["label", "pat"]).copy()
y = dat["label"].copy()

#split test train
#unbalanced in labeled vs unlabeled so when we split a training and a test
#set we will stratify on y so that we can maintain that percentage
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rSeed, stratify=y)

#build vanilla xgboost model
clf_xgb = xgb.XGBClassifier(
    objective = "binary:logistic",
    seed = rSeed,
    early_stopping_rounds = 10, #validation metric must improve at least once every X trees to continue training
    eval_metric = "aucpr"  
    )
clf_xgb.fit(
    X_train, 
    y_train, 
    verbose = True,
    eval_set = [(X_test, y_test)]
    )




#test on new data
#read in new data
dNew = pd.read_csv(newPath)
dNew["pat"] = "007"
#set up new data
XNew = dNew.drop(columns=["label", "pat"]).copy()
yNew = dNew["label"].copy()
#run thru model
predNew = clf_xgb.predict(XNew)
cmNew = confusion_matrix(yNew, predNew, labels=clf_xgb.classes_)
cmDispNew = ConfusionMatrixDisplay(confusion_matrix=cmNew, display_labels=clf_xgb.classes_)
cmDispNew.plot()
#predict class probability
predProbNew = clf_xgb.predict_proba(XNew)
predProb0New = predProbNew[:,0]
predProb1New = predProbNew[:,1]
newDf = pd.DataFrame(yNew)
newDf["pred"] = predNew
newDf["predProb0"] = predProb0New
newDf["predProb1"] = predProb1New
#auc for new mouth
fpr, tpr, thresholds = metrics.roc_curve(newDf["label"], newDf["predProb1"])
metrics.auc(fpr, tpr)
