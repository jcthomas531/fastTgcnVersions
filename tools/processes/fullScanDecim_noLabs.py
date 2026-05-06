import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import decimationFuns as d
import formatAndExportFuns as fe
import os
# sys.argv

#module called arg parse that can print messages

#SET SEED NEXT TIME

# #set up function for full workflow
# def decimFlow(inFile, outFile, nFace = 16000):
#     #perform decimation 
#     mDec = d.decim(inFile=inFile, nFace=nFace)
#     #tranform to data frames
#     mVert, mFace = fe.trimeshToDfNoLabels(mDec)
#     #export
#     fe.dfToPlyExport(vertDf = mVert,
#                      faceDf = mFace,
#                      outFile = outFile)

# #directories and naming for scans
# inDir = sys.argv[1]
# allNames = os.listdir(inDir)
# outDir = sys.argv[2]
# newNames = [i[:-4] + "_dec016.ply" if i.endswith(".ply") else i
#                for i in allNames]


# #loop thru all files
# for i in range(len(newNames)):
#     inPath = inDir + allNames[i]
#     outPath = outDir + newNames[i]
#     decimFlow(inFile = inPath, outFile = outPath, nFace = 16000)
#     print(allNames[i] + " done")


#pull snakemake variables
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


    