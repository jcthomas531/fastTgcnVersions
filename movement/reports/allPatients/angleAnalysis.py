import pickle
import pandas as pd
import numpy as np
import sys
import math
#THIS MAY CAUSE A PROBLEM WHEN CALLED FROM A QMD FILE????? RELATIVE FILE PATHS???
sys.path.append("movement/tools")
import desiredDirec as dd


dd.desiredDirec



#THIS IS A DIRECTIONAL PROBLEM AND NEEDS TO USE MOVEMENT NOT EXPANSION!!!!!
#THE PROCESSING CREATES EXPANSION DATA SET BUT IT SHOULD NOT BE USED


#----------------------
#load in data and process into expansion format

# load in all files for all patients
patNums = ["pat" + i for i in ["001", "004", "007", "008", "012", "013", "014", "015"]]
pklPaths = [
    "K:/iowaExpTest/rigidMovement/moveSummaryData/" + i + "/mainResults.pkl"
    for i in patNums
]
allPatRes = {}
for i in range(len(pklPaths)):
    pati = patNums[i]
    pathi = pklPaths[i]
    with open(pathi, "rb") as f:
        allPatRes[pati] = pickle.load(f)
# object structure
# allPatRes.keys()
# allPatRes["pat004"].keys()
# allPatRes["pat004"]["mainResults"]["3"].keys()

# stack all patient median tables
medAll = pd.concat(
    [allPatRes[i]["medianTable"].assign(patNum=i) for i in patNums], ignore_index=True
)[["patNum", "toothNum", "medX", "medY", "medZ", "medL2"]]

# convert movement to expansion
medAll_expand = medAll.assign(
    # this is like ifelse in r
    # converting tooth num to numeric
    medXExpand=lambda d: np.where(
        pd.to_numeric(d["toothNum"]) <= 8, -d["medX"], d["medX"]
    )
)

# tooth number to numeric for ease of plotting
medAll_expand["toothNum"] = pd.to_numeric(medAll_expand["toothNum"])

#----------------------
#make into unit vectors
unitVec = (
    medAll_expand[["patNum", "toothNum", "medX", "medY"]]
    .assign(
        l2_2d = lambda d: np.sqrt(d["medX"]**2 + d["medY"]**2), #the np version of sqrt operates in a vectorized manner unlike the math.sqrt version
        medX_u = lambda d: d["medX"]/d["l2_2d"],
        medY_u = lambda d: d["medY"]/d["l2_2d"]
    )
)

#----------------------
#plot tooth movement directions in the x-y plane
from plotnine import *


def direcPlot(toothNum):

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

    #create plot
    toothiPlot = (
        ggplot()
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
            title = "Tooth " + toothNum + ", Median Movement Direction (X-Y Plane)",
            x = "X",
            y = "Y",
            color = "Patient"
        )
    )

    return toothiPlot


direcPlot(3)


#now that we are talking about angles and radians, we can start thinking about directional statistics




def direcAndMagPlot(toothNum):

    #set up
    toothNum = str(toothNum)
    toothQuery = "toothNum == " + toothNum

    #create unit circle
    rads = np.linspace(0, 2 * np.pi, 200)
    unitCircle = pd.DataFrame({
        "x": np.cos(rads) * 5,
        "y": np.sin(rads) * 5
    })

    #subset unit vector data to tooth of interest only
    toothi = unitVec.query(toothQuery)

    #desired direction of tooth of interest
    toothiDesired = dd.desiredDirec.query(toothQuery)

    #create plot
    toothiPlot = (
        ggplot()
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
        + coord_fixed(ratio = 1)
        + theme_bw()
        + labs(
            title = "Tooth " + toothNum + ", Median Movement Direction and Magnitude (X-Y Plane)",
            x = "X",
            y = "Y",
            color = "Patient"
        )
    )

    return toothiPlot


direcAndMagPlot(5)
