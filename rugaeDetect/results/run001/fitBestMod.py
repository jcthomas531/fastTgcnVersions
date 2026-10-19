from numpy import shape
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
    outDat = vDat
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
dati = predDict["pat007"]["pred"]

datPre = dati[dati["patId"].str.contains("Pre", na = False)]
pointsPre = datPre[["x", "y", "z"]].to_numpy()
labelsPre = datPre["labelComp"].to_numpy()
# plotPre = pv.Plotter()
# plotPre.add_points(
#     points = pointsPre,
#     scalars = labelsPre,
#     render_points_as_spheres = True,
#     point_size = 20,
#     categories = True,
#     cmap = ["green", "blue", "red", "grey"]
# )
# plotPre.show()


datPost = dati[dati["patId"].str.contains("Post", na = False)]
pointsPost = datPost[["x", "y", "z"]].to_numpy()
labelsPost = datPost["labelComp"].to_numpy()
# plotPost = pv.Plotter()
# plotPost.add_points(
#     points = pointsPost,
#     scalars = labelsPost,
#     render_points_as_spheres = True,
#     point_size = 20,
#     categories = True,
#     cmap = ["green", "blue", "red", "grey"]
# )
# plotPost.show()


#variable importance
#https://stackoverflow.com/questions/37627923/how-to-get-feature-importance-in-xgboost
#https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.get_score
modi = predDict["pat001"]["model"]
#number of times the feature is used to split data across all trees
weightImport = modi.get_booster().get_score(importance_type="weight")
#average gain when used in a split
gainImport = modi.get_booster().get_score(importance_type="gain")
#average cover when used in a split
coverImport = modi.get_booster().get_score(importance_type="cover")
#these are just sums instead of averages
#totGainImport = modi.get_booster().get_score(importance_type="total_gain")
#totCoverImport = modi.get_booster().get_score(importance_type="total_cover")

#ordering based on value
def orderIt(x):
    return dict(sorted(x.items(), key=lambda item: item[1], reverse=True))
def getTop(x, min):
    return {k: v for k, v in x.items() if v >= min}

weightImport = orderIt(weightImport)
weightImportTop = getTop(weightImport, min = 100)

gainImport = orderIt(gainImport)
gainImportTop = getTop(gainImport, min = 5)

coverImport = orderIt(coverImport)
coverImportTop = getTop(coverImport, min = 75)

import matplotlib.pyplot as plt


plt.figure(figsize=(12, 6))
plt.bar(weightImportTop.keys(), weightImportTop.values(), color = "dodgerblue")
plt.xticks(rotation=45)
plt.title("Variable importance: weight")
plt.ylabel("Weight")
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(gainImportTop.keys(), gainImportTop.values(), color = "firebrick")
plt.xticks(rotation=45)
plt.title("Variable importance: gain")
plt.ylabel("Gain")
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(coverImportTop.keys(), coverImportTop.values(), color = "purple")
plt.xticks(rotation=45)
plt.title("Variable importance: cover")
plt.ylabel("Cover")
plt.show()


#this would be good to show averaged across all validation sets



#functionizing variable importance
#arguement x is a fit model object like predDict["pat001"]["model"]
def varImport(x):
    #number of times the feature is used to split data across all trees
    weightImpDf = pd.DataFrame(
        x.get_booster().get_score(importance_type="weight").items(),
        columns=["var", "weigthImport"]
    )
    #average gain when used in a split
    gainImpDf = pd.DataFrame(
        x.get_booster().get_score(importance_type="gain").items(),
        columns = ["var", "gainImport"]
    )
    #average cover when used in a split
    coverImpDf = pd.DataFrame(
        x.get_booster().get_score(importance_type="cover").items(), 
        columns = ["var", "coverImport"]
    )
    #join data frames on var
    impDf = weightImpDf.merge(gainImpDf, on = "var", how = "outer")
    impDf = impDf.merge(coverImpDf, on = "var", how = "outer")
    return impDf


pats = list(predDict.keys())
patImpDfs = {}
for i in range(len(pats)):
    pati = pats[i]
    modi = predDict[pati]["model"]
    impDfi = varImport(modi)
    impDfi["patId"] = pati
    patImpDfs[pati] = impDfi

fullImpDf = pd.concat(patImpDfs.values(), ignore_index = True)
#some of these models seem not the have all of the variables... why is that? just a couple missing
varImpSum = (
    fullImpDf.groupby("var")
    .agg(
        n = ("patId", "size"),
        meanWeightImp = ("weigthImport", "mean"),
        meanGainImp = ("gainImport", "mean"),
        meanCoverImp = ("coverImport", "mean")
    )
    .reset_index()
)

weightImpSum = varImpSum.sort_values("meanWeightImp", ascending = False)
weightImpSumTop = weightImpSum.iloc[0:31]
gainImpSum = varImpSum.sort_values("meanGainImp", ascending = False)
gainImpSumTop = gainImpSum.iloc[0:31]
coverImpSum = varImpSum.sort_values("meanCoverImp", ascending = False)
coverImpSumTop = coverImpSum.iloc[0:31]


from plotnine import ggplot, aes, geom_col, theme_bw, theme, element_text, labs
def varImpPlotter(x, varString, color, title):
    plotx = (
        ggplot(
            x,
            aes(
                x = "reorder(var, -" + varString + ")",
                y = varString
            )
        ) +
        geom_col(
            fill = color
        ) +
        theme_bw() +
        theme(axis_text_x = element_text(angle = 45, hjust = 1)) +
        labs(title = title, x = "Variable")
    )
    return plotx
weightImpPlot = varImpPlotter(
    x = weightImpSumTop, 
    varString="meanWeightImp", 
    color = "dodgerblue", 
    title = "Average weight for each variable across all validation models, top 30")
gainImpPlot = varImpPlotter(
    x = gainImpSumTop,
    varString= "meanGainImp",
    color = "firebrick",
    title = "Average gain for each variable across all validation models, top 30"
    )
coverImpPlot = varImpPlotter(
    x = coverImpSumTop,
    varString= "meanCoverImp",
    color = "purple",
    title = "Average cover for each variable across all validation models, top 30"
    )

#save outputs
# run001WritingDir = "Y:/dissModels/intraoralSegmentation/writing/rugaePipeOptim/run001/"
# weightImpPlot.save(run001WritingDir + "weightImpPlot.png", dpi = 600, width=8, height = 5)
# gainImpPlot.save(run001WritingDir + "gainImpPlot.png", dpi = 600, width=8, height = 5)
# coverImpPlot.save(run001WritingDir + "coverImpPlot.png", dpi = 600, width=8, height = 5)