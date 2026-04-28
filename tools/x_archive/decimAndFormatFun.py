import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import plyFunctions as pf
import convertPlyFiles as cf
sys.path.append("Y:/dissModels/intraoralSegmentation/tools/decimation")
import meshDecimation as d

#this function outputs each ply along the way which is probably unnecessary 
#but this can be changed later on
#
#the argument should take the file name with no .ply extension that way i can 
#force the correct naming convention for each step easier
#fullDir must end with /
#decimDir must end with /
#this is a little bit hacky at the moment bc it is hard coded to name with dec016
def decimAndFormat(fileName,
                   fullDir,
                   decimDir):
    
    if fileName.endswith(".ply"):
        raise ValueError("supply to file name without the extension")
        
    #path to the full scan
    fullFile = fullDir + fileName + ".ply"
    #path and file name for formatted full scan
    formFullFile = fullDir + fileName + "_Form.ply"
    #path and file name for decimated scan
    decFile = decimDir + fileName + "_dec016.ply"
    formDecFile = decimDir + fileName + "_dec016Form.ply"
    
    #convert full scan to corrent formatting
    cf.convertPly(inFile = fullFile, outFile = formFullFile)
    print("convert 1 done")
    #decimate full scan to 16000 faces
    #this looses face lables which is ok bc the ones given by the conversion process
    #are arbitrary as this is not annotated data
    d.myDecimate(inFile = formFullFile, outFile = decFile, nFace = 16000)
    print("decimation done")
    #convert decimated scan to correct formatting
    cf.convertPly(inFile = decFile, outFile = formDecFile)
    print("convert 2 done")
    
    finishMessage = "-----finished: " + fileName + "-----"
    print(finishMessage)
    




