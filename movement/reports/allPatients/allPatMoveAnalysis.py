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
import sys
sys.path.append("../../tools/")
import allPatRes_process1 as p1
import readInAllPats as riap

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

    #read in data
    allPatRes, patNums = riap.readInAllPats(pklPaths)

    #process allPatRes
    medAll_expand = p1.allPatRes_process1(x = allPatRes, patNums = patNums)

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
