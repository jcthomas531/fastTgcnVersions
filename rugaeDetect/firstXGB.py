from xlrd import colname
import xgboost as xgb
import pandas as pd





#testing
inDir = "K:/iowaExpTest/localDescriptors/rugAnnotForm_cSOriMastRemesh_localDescr/"



d1 = pd.read_csv(inDir + "preLabeledCsv/pat001Pre_localDescrLabel.csv")
d1["pat"] = "001"
d2 = pd.read_csv(inDir + "preLabeledCsv/pat004Pre_localDescrLabel.csv")
d2["pat"] = "004"
d4 = pd.read_csv(inDir + "preLabeledCsv/pat008Pre_localDescrLabel.csv")
d4["pat"] = "008"
d5 = pd.read_csv(inDir + "preLabeledCsv/pat012Pre_localDescrLabel.csv")
d5["pat"] = "012"
d6 = pd.read_csv(inDir + "preLabeledCsv/pat013Pre_localDescrLabel.csv")
d6["pat"] = "013"
d7 = pd.read_csv(inDir + "preLabeledCsv/pat014Pre_localDescrLabel.csv")
d7["pat"] = "014"
d8 = pd.read_csv(inDir + "preLabeledCsv/pat015Pre_localDescrLabel.csv")
d8["pat"] = "015"

dAll = pd.concat([d1, d2, d4, d5, d6, d7, d8])


#testing other set
# dFull = pd.read_csv("K:/iowaExpTest/rugaePipeOptim/combinedDescriptors/post/decim8500/norm0.02_desc1.3/combinedDesc.csv")
# dFull = dFull.rename(columns={"patId": "pat"})
# dFull = dFull.loc[:, dFull.columns.str.startswith("pfh") | dFull.columns.isin(["label", "pat", "nx", "ny", "z", "x", "nz", "y"])]
# dAll = dFull[dFull["pat"] != "pat007Post"]





#following this guide
#https://xgboost.readthedocs.io/en/stable/python/python_intro.html





# https://www.youtube.com/watch?v=aLOQD66Sj0g



# https://www.youtube.com/watch?v=GrJP9FLV3FE

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

rSeed = 826

X = dAll.drop(columns=["label", "pat"]).copy()
y = dAll["label"].copy()

#data checking
X.dtypes.unique()
X.dtypes.value_counts()
y.dtypes
y.unique()

#split test train
#unbalanced in labeled vs unlabeled so when we split a training and a test
#set we will stratify on y so that we can maintain that percentage
sum(y)/len(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rSeed, stratify=y)
#check to make sure stratification worked
sum(y_train)/len(y_train)
sum(y_test)/len(y_test)

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

#see performance on a new mouth
#
#first data set
#
d3 = pd.read_csv(inDir + "preLabeledCsv/pat007Pre_localDescrLabel.csv")
d3["pat"] = "007"
#
#other data set
# d3 = dFull[dFull["pat"] == "pat007Post"]
#
XNew = d3.drop(columns=["label", "pat"]).copy()
yNew = d3["label"].copy()
#data checking
XNew.dtypes.unique()
XNew.dtypes.value_counts()
yNew.dtypes
yNew.unique()
#run through model
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
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(newDf["label"], newDf["predProb1"])
metrics.auc(fpr, tpr)

#plot and see what it looks like
d3["predLabel"] = predNew
d3["rugaeComparison"] = np.select(
    [
        (d3["label"] == 1) & (d3["predLabel"] == 1),
        (d3["label"] == 1) & (d3["predLabel"] == 0),
        (d3["label"] == 0) & (d3["predLabel"] == 1),
        (d3["label"] == 0) & (d3["predLabel"] == 0)

    ],
    [
        1,
        2,
        3,
        4
    ],
    default = 0
)
#plot just the predicted labels
import matplotlib.pyplot as plt
%matplotlib widget
colors = d3["predLabel"].map({
    0: "grey",
    1: "deeppink"
})
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    d3["x"],
    d3["y"],
    d3["z"],
    c=colors,
    s=5
)
plt.show()


#plot rugae comparison
%matplotlib widget
colors2 = d3["rugaeComparison"].map({
    1: "green",
    2: "blue",
    3: "orange",
    4: "grey"
})
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection="3d")
ax2.scatter(
    d3["x"],
    d3["y"],
    d3["z"],
    c=colors2,
    s=5
)
plt.show()