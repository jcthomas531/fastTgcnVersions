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
def singleToothMove(preDati, postDati, transi, patNum):
    #tooth number
    toothNum = preDati["face"]["toothNum"][0]
    #calculate position from rigid registration for tooth
    rigidi = applySpatialTrans(dat = preDati, trans=transi)
    #calculating each points movement
    diffi = pd.DataFrame({
        "x": preDati["vert"]["x"] - rigidi["vert"]["x"],
        "y": preDati["vert"]["y"] - rigidi["vert"]["y"],
        "z": preDati["vert"]["z"] - rigidi["vert"]["z"],
        "absX": np.abs(preDati["vert"]["x"] - rigidi["vert"]["x"]),
        "absY": np.abs(preDati["vert"]["y"] - rigidi["vert"]["y"]),
        "absZ": np.abs(preDati["vert"]["z"] - rigidi["vert"]["z"])
    })
    #calculate l2 norm
    diffi["l2Norm"] = np.linalg.norm(x = diffi[["x", "y", "z"]], axis = 1, ord = 2)

    #summary statistics on rigid movement
    sumi = {
        #X
        "meanX": diffi["x"].mean(),
        "medX": diffi["x"].median(),
        "sdX": diffi["x"].std(),
        #y
        "meanY": diffi["y"].mean(),
        "medY": diffi["y"].median(),
        "sdY": diffi["y"].std(),
        #z
        "meanZ": diffi["z"].mean(),
        "medZ": diffi["z"].median(),
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







