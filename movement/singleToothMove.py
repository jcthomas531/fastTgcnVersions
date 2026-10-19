#---------------------------------------------------------------------------------------------------------
#functionizing

#first thing to functionize is the operations on each individual teeth
#then use that function within a new function that operates on all teeth in the mouth
#planning for this to work with the output of the rmtt function like below
#and the transformation for that tooth
preDati = rmtt.restrictMeshToTooth(preDf, toothNum = "3")
postDati = rmtt.restrictMeshToTooth(postDf, toothNum = "3")
#this helper function will be used within
def applySpatialTrans(dat, trans):
    #set up vertex matrix
    vMat = np.transpose(dat["vert"][["x", "y", "z"]].to_numpy())
    bottomRow = [1] * np.shape(vMat)[1]
    vMatReady = np.vstack([vMat, bottomRow])
    #multiply and format result
    newVertMat = np.dot(trans, vMatReady)
    newVertMatForm = np.transpose(np.delete(newVertMat, (3), axis = 0))
    #put new vertex information with face infromation
    #could add normals but probably dont need to right now
    rigid = {}
    rigid["vert"] = pd.DataFrame(newVertMatForm, columns=["x", "y", "z"])
    rigid["face"] = dat["face"]
    return rigid




def singleToothMove(preDati, postDati, transi):
    #calculate position from rigid registration for tooth
    rigidi = applySpatialTrans(dat = preDati, trans=transi)
    #calculating each points movement
    diffi = pd.data.frame({
        "x": preDati["x"] - rigidi["x"],
        "y": preDati["y"] - rigidi["y"],
        "z": preDati["z"] - rigidi["z"]
    })
    #calculate l2 norm
    diffi["l2Norm"] = np.linalg.norm(x = diffi[["x", "y", "z"]], axis = 1, ord = 2)

    #summary statistics on rigid movement
    sumi = {
        "meanX": diffi["x"].mean(),
        "medX": diffi["x"].median(),
        "sdX": diffi["x"].std(),
        "meanY": diffi["y"].mean(),
        "medY": diffi["y"].median(),
        "sdY": diffi["y"].std(),
        "meanZ": diffi["z"].mean(),
        "medZ": diffi["z"].median(),
        "sdZ": diffi["z"].std(),
        "meanL2": diffi["l2Norm"].mean(),
        "medL2": diffi["l2Norm"].median(),
        "sdL2": diffi["l2Norm"].std()
    }

    #visualize tooth before and after rigid movement
    surf_prei = giveSurf.giveSurf(face = preDati["face"], vertex = preDati["vert"])
    surf_rigidi = giveSurf.giveSurf(face = rigidi["face"], vertex = rigidi["vert"])

    #histograms for movement
