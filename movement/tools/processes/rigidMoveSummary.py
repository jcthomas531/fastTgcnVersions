import sys
import pickle
import pandas as pd
import pyvista as pv
sys.path.append("tools")
sys.path.append("movement/tools")
import readAndFormat as raf
from allToothMove import allToothMove


#testing
# prePath = "K:/iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMast_noScale_epoch260/rugAnnotForm_cSOriMastRemesh/pre/pat001Pre_formCSOriMastRemesh_seg.ply"
# postPath = "K:/iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMast_noScale_epoch260/rugAnnotForm_cSOriMastRemesh/post_superimp/pat001Post_formCSOriMastRemesh_rugAnnotSuperimp_seg.ply"
# transPath = "K:/iowaExpTest/spatialTrans/pat001SpatialTransMats.pkl"
# pklPath = "K:/iowaExpTest/testDir/moveTest.pkl"
# rigidImage = "K:/iowaExpTest/testDir/rigidImage.png"
# rigidHtml = "K:/iowaExpTest/testDir/rigidInteractive.html"


#take args from snakemake
prePath = sys.argv[1]
postPath = sys.argv[2]
transPath = sys.argv[3]
pklPath = sys.argv[4]
rigidImage = sys.argv[5]
rigidHtml = sys.argv[6]

#get patNum
import re
patNum = re.search(r"pat[0-9]{3}", prePath).group()

#read in data
with open(transPath, "rb") as f:
    transDict = pickle.load(f)
#load formatted meshes
preDat = raf.readAndFormat(file = prePath, arch = "U")
postDat = raf.readAndFormat(file = postPath, arch = "U")

#get movement
movement = allToothMove(preDat = preDat, postDat = postDat, transDict = transDict, patNum = patNum)

#unique teeth for this patient
uTeeth = list(movement.keys())

#create median table
summaryDict = {}
for i in movement.keys():
    summaryDict[i] = movement[i]["diffSummary"]
summaryDf = pd.DataFrame.from_dict(summaryDict, orient="index")
summaryDf = summaryDf.rename_axis("toothNum").reset_index()
#remember that the median of the absolute value does not equal the absolute value of the mean
medDf = summaryDf[["toothNum", "medX", "medY", "medZ", "absMedX", "absMedY", "absMedZ", "medL2"]]

#output main results
rigidResults = {
    "mainResults": movement,
    "movementTable": summaryDf,
    "medianTable": medDf
}
fp = open(pklPath, "wb")  
pickle.dump(obj = rigidResults,
            file = fp)
fp.close()
#can be read in like: 
# with open(pklPath, "rb") as f:
#     aaa = pickle.load(f)

#graphic for rigid transformation on all teeth
p = pv.Plotter(off_screen = True)
for i in range(len(uTeeth)):
    toothi = uTeeth[i]
    p.add_mesh(movement[toothi]["surf_pre"], color = "white")
    p.add_mesh(movement[toothi]["surf_rigid"], scalars = "rgba", rgb = True,  opacity = .6)
p.show_bounds(
    grid="front",
    location="outer",
    all_edges=True,
    xtitle="X",
    ytitle="Y",
    ztitle="Z"
)
p.view_xy()
p.screenshot(rigidImage)
p.export_html(rigidHtml)


