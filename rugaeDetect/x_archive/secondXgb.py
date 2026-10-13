from xlrd import colname
import xgboost as xgb
import pandas as pd


dAll = pd.read_csv("K:/iowaExpTest/rugaePipeOptim/combinedDescriptors/post/remesh8500/norm0.02_desc1.3/combinedDesc.csv")
#dAll = pd.read_csv("K:/iowaExpTest/rugaePipeOptim/combinedDescriptors/post/decim8500/norm0.02_desc1.3/combinedDesc.csv")
dAll = dAll.loc[:, dAll.columns.str.startswith("pfh") | dAll.columns.isin(["label", "patId", "nx", "ny", "z", "x", "nz", "y"])]

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

rSeed = 826

X = dAll.drop(columns=["label"]).copy()
y = dAll[["label", "patId"]].copy()

#data checking
X.dtypes.unique()
X.dtypes.value_counts()
y.dtypes

#setting up a single LOOCV rep
X_train = X[X["patId"] != "pat007Post"].drop(columns=["patId"]).copy()
X_test = X[X["patId"] == "pat007Post"].drop(columns=["patId"]).copy()
y_train = y[y["patId"] != "pat007Post"].drop(columns=["patId"]).copy()
y_test = y[y["patId"] == "pat007Post"].drop(columns=["patId"]).copy()



#build xgboost model
#note that this is pretty vanilla, not specifying anythin like depth or learning rate
#see the video for how to optimize these hyperparams
#note that i beleive that xgboost will export the last model, not the best model at the end
#parameter scale_pos_weight helps with unbalanced data, adds penalty for incorrectly classifying minority class
#tutorial also shows how to draw a tree, skipping for now
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

#param values: clf_xgb.get_booster().save_config()

#see performance on test set
testPred = clf_xgb.predict(X_test)
cm = confusion_matrix(y_test, testPred, labels = clf_xgb.classes_)
cmDisp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf_xgb.classes_)
cmDisp.plot()




#predict class probability
predNew = clf_xgb.predict(X_test)
predProbNew = clf_xgb.predict_proba(X_test)
predProb0New = predProbNew[:,0]
predProb1New = predProbNew[:,1]
newDf = pd.DataFrame(y_test)
newDf["pred"] = predNew
newDf["predProb0"] = predProb0New
newDf["predProb1"] = predProb1New

#auc for new mouth
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(newDf["label"], newDf["predProb1"])
metrics.auc(fpr, tpr)
