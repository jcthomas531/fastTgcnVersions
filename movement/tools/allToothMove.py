import sys
sys.path.append("movement/tools")
from singleToothMove import singleToothMove
import restrictMeshToTooth as rmtt

#function for getting summary of movement provided by singleToothMove for
#all of the teeth in the arch

def allToothMove(preDat, postDat, transDict, patNum):
    uniqTeeth = [i for i in transDict.keys() if i != "gum"]

    allTeethMove = {}
    for i in range(len(uniqTeeth)):
        #tooth level data
        toothi = uniqTeeth[i]
        preDati = rmtt.restrictMeshToTooth(preDat, toothNum = toothi)
        postDati = rmtt.restrictMeshToTooth(postDat, toothNum = toothi)
        transi = transDict[toothi]
        #get movement summary
        allTeethMove[toothi] = singleToothMove(
            preDati = preDati, postDati = postDati, transi = transi, patNum = patNum
        )
    return allTeethMove