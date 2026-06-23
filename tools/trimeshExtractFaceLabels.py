import pandas as pd
#function that takes a trimesh object that was loaded in from a ply with face labels
#and returns a data frame with a row for each face corresponding to the color
#designed to work with trimeshToDf_labels()
def trimeshExtractFaceLabels(x):
    rgba = x.metadata["_ply_raw"]["face"]["data"]
    colorDf = pd.DataFrame({
        "red": rgba["red"].squeeze(),
        "green": rgba["green"].squeeze(),
        "blue": rgba["blue"].squeeze(),
        "alpha": rgba["alpha"].squeeze()
        })
    return colorDf
#example
# aaa = "K:/iowaExpansion/segResults/segResults_remeshT3dsEpoch270/pre/pat001Pre_modelReady_seg.ply"
# mesh = trimesh.load(aaa, process = False)
# trimeshExtractFaceLabels(mesh)