import pickle
import pandas as pd
import numpy as np
import sys
import math
from plotnine import *
# sys.path.append("movement/tools")
sys.path.append("../../tools/")
import desiredDirec as dd
import allPatRes_process1 as p1
import readInAllPats as riap

#now that we are talking about angles and radians, we can start thinking about directional statistics


#-----------------------------------------------------------------------------------------------------------------------------------
#plotting functions

#plot tooth movement directions in the x-y plane
def direcPlot(toothNum, unitVec):

    #-----------------
    #set up
    toothNum = str(toothNum)
    toothQuery = "toothNum == " + toothNum

    #create unit circle
    rads = np.linspace(0, 2 * np.pi, 200)
    unitCircle = pd.DataFrame({
        "x": np.cos(rads),
        "y": np.sin(rads)
    })

    #subset unit vector data to tooth of interest only
    toothi = unitVec.query(toothQuery)

    #desired direction of tooth of interest
    toothiDesired = dd.desiredDirec.query(toothQuery)

    #-----------------
    #create polygon for shaded region

    #build the region of acceptable directions as a wedge of the unit circle
    #this is pretty much just seq() and on the lwr radian to upr radian
    #.iloc[0] is arbitrary since there is exactly one row per tooth in desiredDirec
    accRads = np.linspace(
        toothiDesired["radian_lwrPLOT"].iloc[0],
        toothiDesired["radian_uprPLOT"].iloc[0],
        100
    )
    #convert the radians to locations on the unit circle 
    #origin added so that all points will connect to that
    #format as data frame
    accRegion = pd.DataFrame({
        "x": np.concatenate([[0], np.cos(accRads)]),
        "y": np.concatenate([[0], np.sin(accRads)])
    })

    #-----------------
    #create plot
    toothiPlot = (
        ggplot()
        #shaded region of acceptable directions, drawn first so it sits beneath all other layers
        + geom_polygon(data = accRegion, mapping = aes(x = "x", y = "y"), fill = "green", alpha = 0.15)
        #reference axes and unit circle
        + geom_hline(yintercept = 0, color = "gray")
        + geom_vline(xintercept = 0, color = "gray")
        + geom_path(data = unitCircle, mapping = aes(x = "x", y = "y"), color = "gray")
        #one arrow per patient from the origin to their x-y movement vector
        + geom_segment(
            data = toothi,
            mapping = aes(x = 0, y = 0, xend = "medX_u", yend = "medY_u", color = "patNum"),
            #arrowhead length in inches
            arrow = arrow(length = 0.15),
            size = 1
        )
        #desired direction as a dashed black arrow
        + geom_segment(
            data = toothiDesired,
            mapping = aes(x = 0, y = 0, xend = "x_u", yend = "y_u"),
            color = "black",
            linetype = "dashed",
            arrow = arrow(length = 0.15),
            size = 1
        )
        #equal scaling on both axes so the circle is round
        + coord_fixed(ratio = 1, xlim = (-1.1, 1.1), ylim = (-1.1, 1.1))
        + theme_bw()
        + labs(
            title = "Tooth " + toothNum + ", Direction of Movement",
            subtitle = "Using median movement vectors scaled to unit circle, X-Y plane",
            x = "X",
            y = "Y",
            color = "Patient"
        )
    )

    return toothiPlot




#direction and magnitude plot
def direcAndMagPlot(toothNum, unitVec):

    #-----------------
    #set up
    toothNum = str(toothNum)
    toothQuery = "toothNum == " + toothNum

    #create unit circle
    #for a reference magnitude, see desiredDirec
    rads = np.linspace(0, 2 * np.pi, 200)
    unitCircle = pd.DataFrame({
        "x": np.cos(rads) * dd.magVal,
        "y": np.sin(rads) * dd.magVal
    })

    #subset unit vector data to tooth of interest only
    toothi = unitVec.query(toothQuery)

    #desired direction of tooth of interest
    toothiDesired = dd.desiredDirec.query(toothQuery)

    #-----------------
    #create polygon for shaded region

    #build the region of acceptable directions as a wedge of the unit circle
    #this is pretty much just seq() and on the lwr radian to upr radian
    #.iloc[0] is arbitrary since there is exactly one row per tooth in desiredDirec
    accRads = np.linspace(
        toothiDesired["radian_lwrPLOT"].iloc[0],
        toothiDesired["radian_uprPLOT"].iloc[0],
        100
    )
    #convert the radians to locations on the unit circle 
    #origin added so that all points will connect to that
    #format as data frame
    accRegion = pd.DataFrame({
        "x": np.concatenate([[0], np.cos(accRads) * dd.magVal]),
        "y": np.concatenate([[0], np.sin(accRads) * dd.magVal])
    })

    #-----------------
    #create plot

    #limits so the plots are always square
    lim = 0.2 + max(
        toothi["medX"].abs().max(),
        toothi["medY"].abs().max(),
        dd.magVal
    )

    toothiPlot = (
        ggplot()
        #shaded region of acceptable directions, drawn first so it sits beneath all other layers
        + geom_polygon(data = accRegion, mapping = aes(x = "x", y = "y"), fill = "green", alpha = 0.15)
        #reference axes and unit circle
        + geom_hline(yintercept = 0, color = "gray")
        + geom_vline(xintercept = 0, color = "gray")
        + geom_path(data = unitCircle, mapping = aes(x = "x", y = "y"), color = "gray")
        #one arrow per patient from the origin to their x-y movement vector
        + geom_segment(
            data = toothi,
            mapping = aes(x = 0, y = 0, xend = "medX", yend = "medY", color = "patNum"),
            #arrowhead length in inches
            arrow = arrow(length = 0.15),
            size = 1
        )
        #desired direction as a dashed black arrow
        + geom_segment(
            data = toothiDesired,
            mapping = aes(x = 0, y = 0, xend = "x_mag", yend = "y_mag"),
            color = "black",
            linetype = "dashed",
            arrow = arrow(length = 0.15),
            size = 1
        )
        #equal scaling on both axes so the circle is round
        + coord_fixed(ratio = 1, xlim = (-lim, lim), ylim = (-lim, lim))
        + theme_bw()
        + labs(
            title = "Tooth " + toothNum + ", Direction and Magnitude of Movement",
            subtitle = "Using median movement vectors, X-Y plane",
            x = "X",
            y = "Y",
            color = "Patient"
        )
    )

    return toothiPlot




def angleAnalysis(pklPaths):
    #THIS IS A DIRECTIONAL PROBLEM AND NEEDS TO USE MOVEMENT NOT EXPANSION!!!!!
    #THE PROCESSING CREATES EXPANSION DATA SET BUT IT SHOULD NOT BE USED

    #----------------------
    #load in data and process into expansion format

    #read in data
    allPatRes, patNums = riap.readInAllPats(pklPaths)

    #process allPatRes
    medAll_expand = p1.allPatRes_process1(x = allPatRes, patNums = patNums)

    #----------------------
    #make into unit vectors
    unitVec = (
        medAll_expand[["patNum", "toothNum", "medX", "medY"]]
        .assign(
            l2_2d = lambda d: np.sqrt(d["medX"]**2 + d["medY"]**2), #the np version of sqrt operates in a vectorized manner unlike the math.sqrt version
            medX_u = lambda d: d["medX"]/d["l2_2d"],
            medY_u = lambda d: d["medY"]/d["l2_2d"],
            #to ensure these radians are always positive applying modulus function
            #thus they will always indicating rotation from postive x axis
            radian = lambda d: np.mod(np.arctan2(d["medY_u"], d["medX_u"]), 2*math.pi)
        )
        #order by tooth number and then patient number
        .sort_values(by = ["toothNum", "patNum"])
        #reset index after sorting
        .reset_index(drop = True)
    )


    #----------------------
    #produce plots
    uTeeth = unitVec["toothNum"].unique()
    direcHolder = {}
    direcMagHolder = {}
    for i in uTeeth:
        iStr = str(i)
        direcHolder["tooth" + iStr] = direcPlot(toothNum = i, unitVec = unitVec)
        direcMagHolder["tooth" + iStr] = direcAndMagPlot(toothNum = i, unitVec = unitVec)

    #----------------------
    #assess if observed vector is in region of acceptable directions and create score
    #score is a weighed sum of 1s and -1s depending on if the vector is within the acceptable direction
    #the weight is the l2 norm. values above 0 indicate overall expansion, below 0 means overall constriction
    #weights help bc if a tooth moves in the wrong direction but only a small amount, it doesnt screw over all expansion

    #table of desired radians prepped to join with the unitVec data
    ddJoin = (
        dd.desiredDirec[["toothNum", "radian"]]
        .copy()
        .rename(columns = {"radian": "desiredRadian"})
    )

    #join to observed data
    unitVec = unitVec.merge(
        ddJoin,
        how = "left",
        on = "toothNum",
        validate = "many_to_one"
    )

    #calculate values necessary to create score
    unitVec = (
        unitVec
        .assign(
            #abolute value of the difference between the observed and desired radian
            absRadDiff = lambda d: np.abs(d["radian"] - d["desiredRadian"]),
            #assess whether the difference is within the acceptable range
            withinRange = lambda d: np.where(d["absRadDiff"] <= dd.accRange, 1, 0),
            #calculate score contribution, l2 norm time 1 if within range and times -1 if not 
            scoreContrib = lambda d: np.where(d["withinRange"] == 1, d["l2_2d"], -d["l2_2d"])
        )
    )

    #calculate score for each patient and each tooth
    scorePat = (
        unitVec
        .groupby("patNum", as_index = False) #as_index = False keeps patNum as a column
        .agg(
            nTeeth = ("toothNum", "size"),
            score = ("scoreContrib", "sum"),
            meanContrib = ("scoreContrib", "mean")
        )
    )
    scoreTooth = (
        unitVec
        .groupby("toothNum", as_index = False) #as_index = False keeps patNum as a column
        .agg(
            nPats = ("patNum", "size"),
            score = ("scoreContrib", "sum"),
            meanContrib = ("scoreContrib", "mean")
        )
    )

    #--------------------
    return {
        "direcPlots": direcHolder,
        "direcMagPlots": direcMagHolder,
        "scorePat": scorePat,
        "scoreTooth": scoreTooth
    }





