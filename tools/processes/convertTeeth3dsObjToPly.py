import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import objJsonToDataFrames as ojtdf
import dfToPlyExport as dtpe

#testing
# objFile = "K:/teeth3DS/scanData/upper/019TUUZD/019TUUZD_upper.obj"
# jsonFile = "K:/teeth3DS/scanData/upper/019TUUZD/019TUUZD_upper.json"
# outFile = "K:/teeth3DS/scanData/upperPly/test.ply"

#extract snakemake variables
objFile = sys.argv[1]
jsonFile = sys.argv[2]
outFile = sys.argv[3]

#convert
vertDf, faceDf = ojtdf.objJsonToDataFrames(objFile = objFile,
                                           jsonFile = jsonFile)

#export
dtpe.dfToPlyExport(vertDf = vertDf,
                   faceDf = faceDf,
                   outFile = outFile)