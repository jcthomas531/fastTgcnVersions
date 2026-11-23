import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from plotnine import (
    ggplot,
    aes,
    geom_point,
    geom_jitter,
    geom_boxplot,
    geom_hline,
    scale_x_continuous,
    theme,
    theme_bw,
    labs,
)

#-----
#plot median expansion across all patients for a single direction
#medDat: long data frame of per patient, per tooth medians in expansion format (medAll_expand)
#sumDat: per tooth summary table across patients (expandSummary), used to plot the means
#yVar: column of medDat to plot
#meanVar: column of sumDat holding the mean of yVar
#titleVar: direction name used in the plot title
def plotAllPat(medDat, sumDat, yVar, meanVar, titleVar):
    ploti = (
        ggplot(medDat, aes(x="toothNum", y=yVar, color="toothNum", shape="patNum"))
        + geom_hline(yintercept=0, color="gray")
        + geom_boxplot(aes(group="toothNum"))
        + geom_jitter(height=0, width=0.25)
        + scale_x_continuous(
            breaks=range(
                np.min(medDat["toothNum"]),
                np.max(medDat["toothNum"]) + 1,
                1,
            )
        )
        + theme_bw()
        + theme(legend_position="none")
        + labs(
            title="Median Expansion in " + titleVar + " Across All Patients",
            x="Tooth Number",
            y="Expansion",
            subtitle="Means plotted in red",
        )
        + geom_point(data=sumDat, mapping=aes(y=meanVar), color="red", shape="x", size=2)
    )
    return ploti
#-----

#load every patient's movement summary, stack the per tooth median tables, convert movement
#to expansion, summarize across patients and make the plots for the all patient report
#patient numbers are taken from the folder each pkl lives in (moveSummaryData/{seg1Pat}/mainResults.pkl)
#so nothing about which patients are included is hard coded here
#pklPaths: list of paths to each patient's mainResults.pkl, as produced by rule getRigidMovementSummary
def allPatMoveAnalysis(pklPaths):

    #copy input so the caller's list is not modified
    pklPathsC = list(pklPaths)

    #patient number is the name of the folder containing the pkl
    patNums = [Path(i).parent.name for i in pklPathsC]

    #duplicated patient numbers would silently overwrite each other in the dict below
    if len(set(patNums)) != len(patNums):
        raise ValueError("duplicate patient numbers in pklPaths: " + ", ".join(patNums))

    #load in all files for all patients
    allPatRes = {}
    for i in range(len(pklPathsC)):
        pati = patNums[i]
        pathi = pklPathsC[i]
        with open(pathi, "rb") as f:
            allPatRes[pati] = pickle.load(f)
    #object structure
    #allPatRes.keys()
    #allPatRes["pat004"].keys()
    #allPatRes["pat004"]["mainResults"]["3"].keys()

    #stack all patient median tables
    medAll = pd.concat(
        [allPatRes[i]["medianTable"].assign(patNum=i) for i in patNums], ignore_index=True
    )[["patNum", "toothNum", "medX", "medY", "medZ", "medL2"]]

    #convert movement to expansion
    #teeth 1-8 are on the other side of the midline so x is flipped to make outward movement positive
    medAll_expand = medAll.assign(
        #this is like ifelse in r
        medXExpand=lambda d: np.where(
            pd.to_numeric(d["toothNum"]) <= 8, -d["medX"], d["medX"]
        )
    )

    #tooth number to numeric for ease of plotting
    medAll_expand["toothNum"] = pd.to_numeric(medAll_expand["toothNum"])

    #summary statistics on expansion across all patients
    #equivalent of a group_by and summarise pipe in R
    expandSummary = medAll_expand.groupby("toothNum", as_index=False).agg(
        n=("patNum", "size"),
        meanX=("medXExpand", "mean"),
        medX=("medXExpand", "median"),
        sdX=("medXExpand", "std"),
        meanY=("medY", "mean"),
        medY=("medY", "median"),
        sdY=("medY", "std"),
        meanZ=("medZ", "mean"),
        medZ=("medZ", "median"),
        sdZ=("medZ", "std"),
        meanL2=("medL2", "mean"),
        medL2=("medL2", "median"),
        sdL2=("medL2", "std"),
    )

    #plots for each direction
    expandXPlot = plotAllPat(medDat=medAll_expand, sumDat=expandSummary, yVar="medXExpand", meanVar="meanX", titleVar="X")
    expandYPlot = plotAllPat(medDat=medAll_expand, sumDat=expandSummary, yVar="medY", meanVar="meanY", titleVar="Y")
    expandZPlot = plotAllPat(medDat=medAll_expand, sumDat=expandSummary, yVar="medZ", meanVar="meanZ", titleVar="Z")
    expandL2Plot = plotAllPat(medDat=medAll_expand, sumDat=expandSummary, yVar="medL2", meanVar="meanL2", titleVar="L2")

    return {
        "patNums": patNums,
        "allPatRes": allPatRes,
        "medAll_expand": medAll_expand,
        "expandSummary": expandSummary,
        "expandXPlot": expandXPlot,
        "expandYPlot": expandYPlot,
        "expandZPlot": expandZPlot,
        "expandL2Plot": expandL2Plot,
    }

#example
# pklPaths = [
#     "K:/iowaExpTest/rigidMovement/moveSummaryData/" + i + "/mainResults.pkl"
#     for i in ["pat001", "pat004", "pat007"]
# ]
# apmaRes = allPatMoveAnalysis(pklPaths = pklPaths)
# apmaRes["expandXPlot"]
