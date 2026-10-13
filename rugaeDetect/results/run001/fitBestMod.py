import pandas as pd
import re
import sys
sys.path.append("tools/rugaeDetect")
from patSplitter import patSplitter
import iouPoint as iouP
import xgboost as xgb
import numpy as np
import pyvista as pv
rSeed = 826

#fitting best model


#norm rad: 0.016
#desc rad: 1.3
#reduct: remesh
#resol: 8500

#read in CV settings of best result
cvSetPath = "K:/iowaExpTest/rugaePipeOptim/performance/run001/fold5/xgb/pfh/remesh8500/norm0.016_desc1.3/results.csv"
cvSet = pd.read_csv(cvSetPath)


#read in data
prePath = "K:/iowaExpTest/rugaePipeOptim/combinedDescriptors/pre/remesh8500/norm0.016_desc1.3/combinedDesc.csv"
postPath = "K:/iowaExpTest/rugaePipeOptim/combinedDescriptors/post/remesh8500/norm0.016_desc1.3/combinedDesc.csv"
colKeepLogical = lambda col: bool(re.match("^pfh_", col)) or col in ["label", "patId", "nx", "ny", "z", "x", "nz", "y"]
datPre = pd.read_csv(prePath, usecols=colKeepLogical)
datPost = pd.read_csv(postPath, usecols=colKeepLogical)
dat = pd.concat([datPre, datPost], axis = 0, ignore_index=True)

#applying a very conservative low variance filter
feats = dat.loc[:, dat.columns.str.match("^pfh_")]
featVar = feats.var()
dat = dat.drop(columns = featVar[featVar < 0.0001].index)

#unique patients ids
uniqPats = dat["patId"].str.extract(r"(pat[0-9]{3})")[0].unique()

#rotating hold out patient data frames
splitDat = patSplitter(dat)

#loop thru hold out groups
#all of the param values are the same for all of the validation sets so they are hard coded in
predDict = {}
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


    #set up fixed parameter values
    params = {
        "n_estimators": 400,
        "learning_rate": 0.2,
        "gamma": 0,
        "max_depth": 7
    }

    #set up xgboost model
    modXgb = xgb.XGBClassifier(
        objective="binary:logistic",
        random_state=rSeed,
        n_jobs=16,
        **params
    )

    #fit model
    fitXgb = modXgb.fit(
        X=ttDatX,
        y=ttDatY
    )

    #predict on validation set
    valPred = fitXgb.predict(vDatX)
    valPredProb = fitXgb.predict_proba(vDatX)[:,1]

    #output data set
    outDat = vDatX
    outDat["label"] = vDatY
    outDat["predLabel"] = valPred
    outDat["predProb"] = valPredProb
    outDat["labelComp"] = np.select(
        [
            (outDat["label"] == 1) & (outDat["predLabel"] == 1),
            (outDat["label"] == 1) & (outDat["predLabel"] == 0),
            (outDat["label"] == 0) & (outDat["predLabel"] == 1),
            (outDat["label"] == 0) & (outDat["predLabel"] == 0)

        ],
        [
            1,
            2,
            3,
            4
        ],
        default = 0
    )


    #object to output
    retObj = {
        "model": fitXgb,
        "pred": outDat
    }

    predDict[pati] = retObj


#visualizing predicted regions
dati = predDict["pat015"]["pred"]
pointsi = dati[["x", "y", "z"]].to_numpy()
labelsi = dati["labelComp"].to_numpy()
ploti = pv.Plotter()
ploti.add_points(
    points = pointsi,
    scalars = labelsi,
    render_points_as_spheres = True,
    point_size = 20,
    categories = True,
    cmap = ["green", "blue", "red", "grey"]
)
ploti.show()
