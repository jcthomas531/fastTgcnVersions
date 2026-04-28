import os
import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools/convertRmeStl")
#sys.path.append("/Users/jthomas48/dissModels/intraoralSegmentation/tools/convertRmeStl")
import convertRmeStlToPly as csp
import re
from datetime import datetime
import logging

##########
#creating log
##########
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
logFilePath = f"K:/iowaRme/preDelivAndFinalScans/outputFiles/{timestamp}_stlToRmeConversionLog.txt"
logging.basicConfig(
    filename = logFilePath,
    level = logging.INFO,
    format="%(message)s",
    filemode="a"
    )
logging.warning("Converting iowaRme upper scans from stl files to ply")
logging.warning("Conversion process in convertRmeStlToPly.py")
logging.warning("Current date and time: " + datetime.now().strftime("%Y_%m_%d_%H_%M"))
logging.warning("Begin log")



##########
#loop thru and convert files
##########

#paths for converted ply files
#these paths will be the same for all patients
preDOutDir = "K:/iowaRme/preDelivAndFinalScans/preDelivScanU/fullScans/"
finOutDir = "K:/iowaRme/preDelivAndFinalScans/finalScanU/fullScans/"


allPats = os.listdir("K:/iowaRme/preDelivAndFinalScans/originalStl")

for i in range(len(allPats)):
    
    ##########
    #set up
    ##########
    
    #get patient number
    pati = allPats[i]
    
    #output files
    #patient specific file names
    preDOutName = pati + "u_preD.ply"
    finOutName = pati + "u_fin.ply" 
    #full file paths
    preDOutFilePath = preDOutDir + preDOutName
    finOutFilePath = finOutDir + finOutName
    
    #input files
    #paths and file names for original stls
    #directory of original files for the patient
    patPreDDir = "K:/iowaRme/preDelivAndFinalScans/originalStl/" +  pati +"/"
    patFinDir = patPreDDir + "final/"
    
    
    ##########
    #perform the conversion
    ##########
    
    ###
    #for pre delivery conversion
    ###
    
    #if the directory doesnt exist, we want to move to the next iteration
    #if this directory doesnt exist, then the sub dir "final" will not exist so
    #this will not cause problems downstream
    if not os.path.isdir(patPreDDir):
        logging.warning(pati + ": directory does not exist")
        continue
    
    #find file name of upper scan for pre delivery scan
    preDInNameList = [i for i in os.listdir(patPreDDir) if re.search("u\.stl$", i)]
    
    #for pre delivery scan
    if len(preDInNameList) == 1:
        #extract file name
        preDInName = preDInNameList[0]
        #append to file path
        preDInFilePath = patPreDDir + preDInName
        #check to make sure ply file doesnt already exist here
        #this really should be governed by a function arguement for overwritting 
        #but we can add that at a later time if this needs to be done again
        if not os.path.isfile(preDOutFilePath):
            #perform conversion to ply
            csp.convertRmeStlToPly(inFile = preDInFilePath,
                                   outFile = preDOutFilePath)
            logging.warning(".")
        else:
            logging.warning(pati + ": pre-delivery upper scan conversion already performed")
        
    elif len(preDInNameList) == 0:
        logging.warning(pati + ": pre-delivery upper scan not found")
    elif len(preDInNameList) > 1:
        logging.warning(pati + ": multiple pre-delivery upper scans found")
    else:
        logging.warning(pati + ": pre-delivery upper scan, unknown file issue")
        
    
    
    
    
    ###
    #for final scan conversion
    ###
    #if the "final" directory doesnt exist, we want to move on to the next iteration
    if not os.path.isdir(patFinDir):
        logging.warning(pati + ": 'final' subdirectory does not exist")
        continue
    
    #find file name of upper scan for final scan
    finInNameList = [i for i in os.listdir(patFinDir) if re.search("u\.stl$", i)]
        
    #for final scan
    if len(finInNameList) == 1:
        #extract file name
        finInName = finInNameList[0]
        #append to file path
        finInFilePath = patFinDir + finInName
        #check to make sure ply file doesnt already exist here
        #this really should be governed by a function arguement for overwritting 
        #but we can add that at a later time if this needs to be done again
        if not os.path.isfile(finOutFilePath):
            #perform conversion to ply
            csp.convertRmeStlToPly(inFile = finInFilePath,
                                   outFile = finOutFilePath)
            logging.warning(".")
        else:
            logging.warning(pati + ": final upper scan conversion already performed")
    elif len(finInNameList) == 0:
        logging.warning(pati + ": final upper scan not found")
    elif len(finInNameList) > 1:
        logging.warning(pati + ": multiple final upper scans found")
    else:
        logging.warning(pati + ": final upper scan, unknown file issue")



logging.warning("Process finished. End of log.")
logging.warning("Current date and time: " + datetime.now().strftime("%Y_%m_%d_%H_%M"))





