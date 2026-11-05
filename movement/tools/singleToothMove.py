import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_histogram, labs, theme_bw, geom_vline
import sys
sys.path.append("movement/tools")
from applySpatialTrans import applySpatialTrans
sys.path.append("tools")
import giveSurf





#movement for a single tooth
#all scan data in the usual vert and face formatting 
#preDati is the pre scan restricted to individual tooth i
#postDati is the post scan restricted to individual tooth i
#transi is the affine transformation
#patNum can be numeric or string, its only used for naming

#IMPORTANT NOTE
#the current placement of the axes on the scan can make the sign of the movement
#vectors challenging to interpret. to get around this, i am going to temporarily
#move the pre and the rigid scans far into quadrant 1 and then take the measurement
#this movement is arbitrary because the pre serves as the reference for the post and
#they are both being moved by the same transformation matrix
#this is done just for the ease of measurement and not saved
#i can see this being a point of contention when showing measurements side by side
#with images aligned to the axes differently, but i will cross that bridge later
#it wouldnt be too hard to make the images like that but i like it how it is currently


def singleToothMove(preDati, postDati, transi, patNum):
    #tooth number
    toothNum = preDati["face"]["toothNum"][0]
    #calculate position from rigid registration for tooth
    rigidi = applySpatialTrans(dat = preDati, trans=transi)


    #----------
    #for the sake of interpretation of movement, pre and rigid scans moved into quad 1
    #see note above for details, this is done only for the measurement and is not saved

    #transformation matrix for moving to q1
    quadMove = np.array([
        [1,0,0,500],
        [0,1,0,500],
        [0,0,1,500],
        [0,0,0,1]
    ])
    preDati_moved = applySpatialTrans(dat = preDati, trans = quadMove)
    rigidi_moved = applySpatialTrans(dat = rigidi, trans = quadMove)

    #calculating each points movement
    #remember that the median of the absolute value does not equal the absolute value of the mean
    diffi = pd.DataFrame({
        "x": rigidi_moved["vert"]["x"] - preDati_moved["vert"]["x"],
        "y": rigidi_moved["vert"]["y"] - preDati_moved["vert"]["y"],
        "z": rigidi_moved["vert"]["z"] - preDati_moved["vert"]["z"],
        "absX": np.abs(rigidi_moved["vert"]["x"] - preDati_moved["vert"]["x"]),
        "absY": np.abs(rigidi_moved["vert"]["y"] - preDati_moved["vert"]["y"]),
        "absZ": np.abs(rigidi_moved["vert"]["z"] - preDati_moved["vert"]["z"])
    })
    #calculate l2 norm
    diffi["l2Norm"] = np.linalg.norm(x = diffi[["x", "y", "z"]], axis = 1, ord = 2)
    
    #summary statistics on rigid movement
    #remember that the median of the absolute value does not equal the absolute value of the mean
    sumi = {
        #X
        "meanX": diffi["x"].mean(),
        "absMeanX": np.abs(diffi["x"].mean()),
        "medX": diffi["x"].median(),
        "absMedX": np.abs(diffi["x"].median()),
        "sdX": diffi["x"].std(),
        #y
        "meanY": diffi["y"].mean(),
        "absMeanY": np.abs(diffi["y"].mean()),
        "medY": diffi["y"].median(),
        "absMedY": np.abs(diffi["y"].median()),
        "sdY": diffi["y"].std(),
        #z
        "meanZ": diffi["z"].mean(),
        "absMeanZ": np.abs(diffi["z"].mean()),
        "medZ": diffi["z"].median(),
        "absMedZ": np.abs(diffi["z"].median()),
        "sdZ": diffi["z"].std(),
        #absoluate values of movement
        #X
        "meanAbsX": diffi["absX"].mean(),
        "medAbsX": diffi["absX"].median(),
        "sdAbsX": diffi["absX"].std(),
        #Y
        "meanAbsY": diffi["absY"].mean(),
        "medAbsY": diffi["absY"].median(),
        "sdAbsY": diffi["absY"].std(),
        #Z
        "meanAbsZ": diffi["absZ"].mean(),
        "medAbsZ": diffi["absZ"].median(),
        "sdAbsZ": diffi["absZ"].std(),
        #norm
        "meanL2": diffi["l2Norm"].mean(),
        "medL2": diffi["l2Norm"].median(),
        "sdL2": diffi["l2Norm"].std()
    }
    #----------

    #surfaces for visualizing tooth before and after rigid movement as well as the true post surface
    surf_prei = giveSurf.giveSurf(face = preDati["face"], vertex = preDati["vert"])
    surf_posti = giveSurf.giveSurf(face = postDati["face"], vertex = postDati["vert"])
    surf_rigidi = giveSurf.giveSurf(face = rigidi["face"], vertex = rigidi["vert"])

    #put results in nice container
    resulti = {
        "patNum": patNum,
        "toothNum": toothNum,
        "diffDat": diffi,
        "diffSummary": sumi,
        "surf_pre": surf_prei,
        "surf_post": surf_posti,
        "surf_rigid": surf_rigidi
    }
    return resulti







