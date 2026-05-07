import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import decimationFuns as d
import formatAndExportFuns as fe
import os
# sys.argv


#pull variables from snakemake
inFile = sys.argv[1]
outFile = sys.argv[2]


# print(inFile)
# print(outFile)

#perform decimation 
mDec = d.decim(inFile=inFile, nFace=16000)
#tranform to data frames
mVert, mFace = fe.trimeshToDfNoLabels(mDec)
#export
fe.dfToPlyExport(vertDf = mVert,
                 faceDf = mFace,
                 outFile = outFile)
print("decimation for " + inFile + " finisheds")

    