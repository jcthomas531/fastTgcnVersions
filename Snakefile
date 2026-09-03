import os
import re

#for running on the hpc

container: "../../containers/lorwyn_eclipsed.sif"
defaultThreads = 4

#tutorial: https://www.youtube.com/watch?v=r9PWnEmz_tc&t=1247s

#some rules
#cannot have empty lines in a rule

###############################################################################
#helper functions
##########
#for getting the directory dictionaries used in the initial helper functions for raw data
def patNamesAndPathDict(dir_, pattern = r'^pat[0-9]{3}', captureGroup = 0, fileExt = "all"):
    
    #get files and make file paths
    files = os.listdir(dir_)
    
    # filter by file extension
    if fileExt != "all":
        files = [file_ for file_ in files if file_.endswith(fileExt)]
    
    #make file paths
    paths = [dir_ + file_ for file_ in files]
    
    #extract patient names
    patNames = [re.search(pattern, i).group(captureGroup) for i in files]
    
    #create path dictionary
    pathDict = dict(zip(patNames, paths))
    
    return patNames, pathDict

###############################################################################
#directories
##########

grantDir = "../../../../Shared/gb_lss/Thomas/"

#iowaRme
#original stl directory
origStlDir = grantDir + "iowaRme/preDelivAndFinalScans/originalStl/"

#iowaRme:
#preD files and directories
preDFullScanDir = grantDir + "iowaRme/preDelivAndFinalScans/preDelivScanU/fullScans/"
#fin files and directories
finFullScanDir = grantDir + "iowaRme/preDelivAndFinalScans/finalScanU/fullScans/"

#iowaExpansion
#full, rugae annotated scans
iowaExpFullAnnotPreDir = grantDir + "iowaExpansion/fullRugaeAnnotScans/pre/"
iowaExpFullAnnotPostDir = grantDir + "iowaExpansion/fullRugaeAnnotScans/post/"
#segmentation model ready scans
iowaExpSegReadyPreDir = grantDir + "iowaExpansion/segReadyScans/pre/"
iowaExpSegReadyPostDir = grantDir + "iowaExpansion/segReadyScans/post/"
#TEMP
iowaExpSegReadyPreDir2 = grantDir + "iowaExpansion/segReadyScans2/pre/"
iowaExpSegReadyPostDir2 = grantDir + "iowaExpansion/segReadyScans2/post/"
#iowaExpTest
#original scans from itero
iowaExpTestOrigPreDir = grantDir + "iowaExpTest/scanData/orig/pre/"
iowaExpTestOrigPostDir = grantDir + "iowaExpTest/scanData/orig/post/"
iowaExpTestOrigFormPreDir = grantDir + "iowaExpTest/scanData/origForm/pre/"
iowaExpTestOrigFormPostDir = grantDir + "iowaExpTest/scanData/origForm/post/"
iowaExpTestOrigFormCSPreDir = grantDir + "iowaExpTest/scanData/origForm_cS/pre/"
iowaExpTestOrigFormCSPostDir = grantDir + "iowaExpTest/scanData/origForm_cS/post/"
iowaExpTestOrigFormCSPreMastRotMatDir = grantDir + "iowaExpTest/rotationMatrices/origForm_cS_mastRotMat/pre/"
iowaExpTestOrigFormCSPostMastRotMatDir = grantDir + "iowaExpTest/rotationMatrices/origForm_cS_mastRotMat/post/"
iowaExpTestOrigFormCSOriMastPreDir = grantDir + "iowaExpTest/scanData/origForm_cSOriMast/pre/"
iowaExpTestOrigFormCSOriMastPostDir = grantDir + "iowaExpTest/scanData/origForm_cSOriMast/post/"
iowaExpTestOrigFormCSOriMastRemeshPreDir = grantDir + "iowaExpTest/scanData/origForm_cSOriMastRemesh/pre/"
iowaExpTestOrigFormCSOriMastRemeshPostDir = grantDir + "iowaExpTest/scanData/origForm_cSOriMastRemesh/post/"
#segmeted directories
iowaExpTestSegPreDir = grantDir + "iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/origForm_cSOriMastRemesh/pre/"
iowaExpTestSegPostDir = grantDir + "iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/origForm_cSOriMastRemesh/post/"
#centroid size directiories
iowaExpTestCentSizeDir = grantDir + "iowaExpTest/centroidSize/"

#iowaExpTest
#rugae annotate (RA) scans
iowaExpTestRAPreDir = grantDir + "iowaExpTest/scanData/rugAnnot/pre/"
iowaExpTestRAPostDir = grantDir + "iowaExpTest/scanData/rugAnnot/post/"
iowaExpTestRAFormPreDir = grantDir + "iowaExpTest/scanData/rugAnnotForm/pre/"
iowaExpTestRAFormPostDir = grantDir + "iowaExpTest/scanData/rugAnnotForm/post/"
iowaExpTestRAFormCSPreDir = grantDir + "iowaExpTest/scanData/rugAnnotForm_cS/pre/"
iowaExpTestRAFormCSPostDir = grantDir + "iowaExpTest/scanData/rugAnnotForm_cS/post/"
iowaExpTestRAFormCSPreMastRotMatDir = grantDir + "iowaExpTest/rotationMatrices/rugAnnotForm_cS_mastRotMat/pre/"
iowaExpTestRAFormCSPostMastRotMatDir = grantDir + "iowaExpTest/rotationMatrices/rugAnnotForm_cS_mastRotMat/post/"
iowaExpTestRAFormCSOriMastPreDir = grantDir + "iowaExpTest/scanData/rugAnnotForm_cSOriMast/pre/"
iowaExpTestRAFormCSOriMastPostDir = grantDir + "iowaExpTest/scanData/rugAnnotForm_cSOriMast/post/"
iowaExpTestRAFormCSOriMastRemeshPreDir = grantDir + "iowaExpTest/scanData/rugAnnotForm_cSOriMastRemesh/pre/"
iowaExpTestRAFormCSOriMastRemeshPostDir = grantDir + "iowaExpTest/scanData/rugAnnotForm_cSOriMastRemesh/post/"
iowaExpTestRAFormCSOriMastRemeshDir = grantDir + "iowaExpTest/scanData/rugAnnotForm_cSOriMastRemesh/"
#directories for superimposition transformations
iowaExpTestRATransDir = grantDir + "iowaExpTest/superimposition/transformations/rugAnnotForm_cSOriMast/"
#directory for post scans with superimposition transformation applied
iowaExpTestRASuperimpPostScanDir = grantDir + "iowaExpTest/superimposition/superimpPostScan/rugAnnotForm_cSOriMast/"
iowaExpTestRARemeshSuperimpPostScanDir = grantDir + "iowaExpTest/superimposition/superimpPostScan/rugAnnotForm_cSOriMastRemesh/"
#directory for html visuals of pre and post scans without superimposition
iowaExpTestRANoSuperimpVisDir = grantDir + "iowaExpTest/superimposition/visuals/noSuperimp/rugAnnotFrom_cSOriMast/"
#directory for html visuals of pre and post scans with annoted rugae superimposition
iowaExpTestRASuperimpVisDir = grantDir + "iowaExpTest/superimposition/visuals/superimp/rugAnnotFrom_cSOriMast/"

#teeth3ds
#full plys
teeth3dsFullDir = grantDir + "teeth3DS/scanData/upperPly/"
teeth3dsFullCSDir = grantDir + "teeth3DS/scanData/upperPly_cS/"
teeth3dsCSMastRotMatDir = grantDir + "teeth3DS/rotationMatrices/upperPly_cS_mastRotMat/"
teeth3dsFullCSOriMastDir = grantDir + "teeth3DS/scanData/upperPly_cSOriMast/"
teeth3dsCSOriMastRemeshDir = grantDir + "teeth3DS/scanData/upperPly_cSOriMastRemesh/"
#original files
teeth3dsOrigFilesDir = grantDir + "teeth3DS/scanData/upper/"

#iosseg
#plys
iossegCleanUDir = grantDir + "IOSSegData/scanData/cleanU/"
iossegCleanUCSDir = grantDir + "IOSSegData/scanData/cleanU_cS/"
iossegCleanUCSMastRotMatDir = grantDir + "IOSSegData/rotationMatrices/cleanU_cS_mastRotMat/"
iossegCleanUCSOriMastDir = grantDir + "IOSSegData/scanData/cleanU_cSOriMast/"

#master arches
masterArchesDir = grantDir + "masterArches/"

#train test sets
trainTestDir_t3dsIosseg_cSOriMast = grantDir + "trainTestSets/t3dsIosseg_cSOriMast/"

#segmentation directories
iowaExpTestSegDir = grantDir + "iowaExpTest/segResults/"
iowaExpTestSeg1Dir_pre = iowaExpTestSegDir + "segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/pre/"
iowaExpTestSeg1Dir_post = iowaExpTestSegDir + "segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/post/"

#spatialTrans
iowaExpTestSpatialTransDir = grantDir + "iowaExpTest/spatialTrans/"

###############################################################################
#iowRme
#some patients may not have all of the files we need
#we are soley concerned about upper files
#the logic below creates lists (patNamesPreD and patNamesFin) of the patients who
#respectively have original stls for upper scans. these should be used throughout
#the rest of the logic as they represent the base truth of the scans that exist
##########

#original patient list
allPats = os.listdir(origStlDir)

#for preD
allPatsPreDStlDir = [origStlDir + i + "/" for i in allPats]
#ensure the upper file exists in this directory
preDStlHasU = []
for i in allPatsPreDStlDir:
    filesi = os.listdir(i)
    isU = any(j.endswith("u.stl") for j in filesi)
    preDStlHasU.append(isU)
#list of patient preD scans with an upper file
patNamesPreD = [pats for pats, logic in zip(allPats, preDStlHasU) if logic]

#for final
allPatsFinStlDir = [i + "final/" for i in allPatsPreDStlDir]
#ensure this directory exists and there is an upper file in it
finStlProper = []
for i in allPatsFinStlDir:
    finDirExists = os.path.isdir(i)
    if finDirExists:
        filesi = os.listdir(i)
        isU = any(j.endswith("u.stl") for j in filesi)
        finStlProper.append(finDirExists and isU)
    else:
        finStlProper.append(finDirExists)
#list of patient fin scans with upper files
patNamesFin = [pats for pats, logic in zip(allPats, finStlProper) if logic]

#finding the patients that have both a preD scan and a fin scan
patNamesBoth = list(set(patNamesPreD) & set(patNamesFin))

###############################################################################
#iowaRme
#navigating the orignal stl directory
########
#preD data: original stl files
origStlPreDDir = [origStlDir + i + "/" for i in patNamesPreD]
origStlPreDFile = []
for i in origStlPreDDir:
    filesi = os.listdir(i)
    #taking first one bc there should only be one, this isnt perfect logic, but it will 
    #get us started
    filenamei = [j for j in filesi if re.search("u\.stl$", j)][0] 
    origStlPreDFile.append(filenamei)
origStlPreDFilepath = [dir_ + file_ for dir_, file_ in zip(origStlPreDDir, origStlPreDFile)]
#make this into a dictionary so snakemake can use it easily
origStlPreDFilePathDict = dict(zip(patNamesPreD, origStlPreDFilepath))
#create helper function
#this calls into the snakemake wildcards where we have been using preDPat to
#represent preD patient names 
def getOrigStlPreD(wildcards):
    return origStlPreDFilePathDict[wildcards.preDPat]

#fin data: original stl files
origStlFinDir = [origStlDir + i + "/final/" for i in patNamesFin]
origStlFinFile = []
for i in origStlFinDir:
    filesi = os.listdir(i)
    #taking first one bc there should only be one, this isnt perfect logic, but it will 
    #get us started
    filenamei = [j for j in filesi if re.search("u\.stl$", j)][0] 
    origStlFinFile.append(filenamei)
origStlFinFilepath = [dir_ + file_ for dir_, file_ in zip(origStlFinDir, origStlFinFile)]
#make this into a dictionary so snakemake can use it easily
origStlFinFilePathDict = dict(zip(patNamesFin, origStlFinFilepath))
#create helper function
#this calls into the snakemake wildcards where we have been using finPat to
#represent fin patient names 
def getOrigStlFin(wildcards):
    return origStlFinFilePathDict[wildcards.finPat]

###############################################################################
#iowaExpTest
#get patient names and create directory dictionary for orig files
iowaExpTestPatsPre, iowaExpTestOrigPathDictPre = patNamesAndPathDict(iowaExpTestOrigPreDir)
iowaExpTestPatsPost, iowaExpTestOrigPathDictPost = patNamesAndPathDict(iowaExpTestOrigPostDir)
#create helper functions for using the raw data
def getIowaExpTestOrigPre(wildcards):
    return iowaExpTestOrigPathDictPre[wildcards.iowaExpTestPrePat]
def getIowaExpTestOrigPost(wildcards):
    return iowaExpTestOrigPathDictPost[wildcards.iowaExpTestPostPat]


#iowaExpTestRA
#get patient names and create directory dictionary for rugae annotated (RA) files
iowaExpTestRAPatsPre, iowaExpTestRAPathDictPre = patNamesAndPathDict(iowaExpTestRAPreDir)
iowaExpTestRAPatsPost, iowaExpTestRAPathDictPost = patNamesAndPathDict(iowaExpTestRAPostDir)
#create helper functions for using the raw data
def getIowaExpTestRAPre(wildcards):
    return iowaExpTestRAPathDictPre[wildcards.iowaExpTestRAPrePat]
def getIowaExpTestRAPost(wildcards):
    return iowaExpTestRAPathDictPost[wildcards.iowaExpTestRAPostPat]

#iowaExpTestRA, patient pairs, for superimp
#patient names for just the patients with both a pre and a post
iowaExpTestRAPatsBoth = list(set(iowaExpTestRAPatsPre) & set(iowaExpTestRAPatsPost))

#helper function for iowaExpTestRA files after then are segmented
iowaExpTestRAPatsPre_seg1, iowaExpTestRAPathDictPre_seg1 = patNamesAndPathDict(iowaExpTestSeg1Dir_pre, fileExt = ".ply")
iowaExpTestRAPatsPost_seg1, iowaExpTestRAPathDictPost_seg1 = patNamesAndPathDict(iowaExpTestSeg1Dir_post, fileExt = ".ply")
#create helper functions
def getIowaExpTestSeg1Pre(wildcards):
    return iowaExpTestRAPathDictPre_seg1[wildcards.seg1Pat]
def getIowaExpTestSeg1Post(wildcards):
    return iowaExpTestRAPathDictPost_seg1[wildcards.seg1Pat]

###############################################################################
#teeth3ds
#original obj and json files
#original patient list
allPats3ds = os.listdir(teeth3dsOrigFilesDir)
#directory for all patients
allPats3dsDir = [teeth3dsOrigFilesDir + i + "/" for i in allPats3ds]
#ensure the an obj and stl file exists in each directory
hasBoth3ds = []
for i in allPats3dsDir:
    filesi = os.listdir(i)
    isObj = any(j.endswith(".obj") for j in filesi)
    isJson = any(j.endswith(".json") for j in filesi)
    hasBoth3ds.append(isObj and isJson)
#take only those meeting this criteria
patNames3ds = [pats for pats, logic in zip(allPats3ds, hasBoth3ds) if logic]
#file paths for original obj and json files
pat3dsDir = [teeth3dsOrigFilesDir + i + "/" for i in patNames3ds]
orig3dsObjFile = []
orig3dsJsonFile = []
for i in pat3dsDir:
    filesi = os.listdir(i)
    #taking first one bc there should only be one, this isnt perfect logic, but it will 
    #get us started
    objFilenamei = [j for j in filesi if re.search(".obj$", j)][0] 
    orig3dsObjFile.append(objFilenamei)
    jsonFilenamei = [j for j in filesi if re.search(".json$", j)][0] 
    orig3dsJsonFile.append(jsonFilenamei)
orig3dsObjPath = [dir_ + file_ for dir_, file_ in zip(pat3dsDir, orig3dsObjFile)]
orig3dsJsonPath = [dir_ + file_ for dir_, file_ in zip(pat3dsDir, orig3dsJsonFile)]
#make this into a dictionary so snakemake can use it easily
orig3dsObjPathDict = dict(zip(patNames3ds, orig3dsObjPath))
orig3dsJsonPathDict = dict(zip(patNames3ds, orig3dsJsonPath))

#create helper function
#this calls into the snakemake wildcards
def getOrig3dsObj(wildcards):
    return orig3dsObjPathDict[wildcards.teeth3dsName]
def getOrig3dsJson(wildcards):
    return orig3dsJsonPathDict[wildcards.teeth3dsName]

###############################################################################
#iosseg

#iosseg
#get patient names and create directory dictionary
allIossegCleanUPats, iossegCleanUPathDict = patNamesAndPathDict(iossegCleanUDir, pattern = r'^[0-9]{3}')
#create helper functions for using the raw data
def getIossegCleanU(wildcards):
    return iossegCleanUPathDict[wildcards.iossegCleanUPat]

###############################################################################
#dependency lists
stlConvertNoLabsDepends = ["tools/stlToPlyFuns.py"]
makeSegReadyDeps = ["tools/getRegistration.py", "tools/trimeshToDfNoLabels.py", "tools/dfToPlyExport.py"]
superimpIowaExpAnnotRugaeDeps = ["tools/getRegistration.py", "tools/preprocess_point_cloud.py", "tools/trimeshToDf_labels.py", "tools/trimeshExtractFaceLabels.py", "tools/dfToPlyExport.py"]
manipulateAndFormatPack = ["tools/trimeshExtractFaceLabels.py", "tools/trimeshToDf_labels.py", "tools/dfToPlyExport.py"]
manipulateAndFormatPack2 = ["tools/trimeshExtractFaceLabels.py", "tools/trimeshToDf_labels.py", "tools/trimeshToDfNoLabels.py", "tools/dfToPlyExport.py"]
convertTeeth3dsObjToPlyDeps = ["tools/colorNumFrame.py", "tools/trimeshToDf_labels.py", "tools/objJsonToDataFrames.py", "tools/dfToPlyExport.py"]
getRotToMastDeps = ["tools/getRotToMaster.py", "tools/preprocess_point_cloud.py"]
remeshDeps = ["tools/trimeshExtractFaceLabels.py", "tools/colorNumFrame.py", "tools/trimeshToDf_labels.py", "tools/trimeshToDfNoLabels.py", "tools/dfToPlyExport.py"]
formatNoLabPlyDeps = ["tools/trimeshToDfNoLabels.py", "tools/dfToPlyExport.py"]
calcCentSizeDeps = [
"tools/readAndFormat.py",
"tools/toothCentroids.py",
"tools/toothGroupCent.py",
"tools/teethToCenterDist.py",
"tools/centroidSize.py",
"tools/toothVars.py",
"tools/plyRead.py",
"tools/archLength.py",
"tools/centToCentDist.py"
]
segDeps = [
"prediction/fastTgcnEasyPredictFun.py",
"fastTgcnEasy/dataloader.py",
"fastTgcnEasy/Baseline.py",
"fastTgcnEasy/loss.py",
"fastTgcnEasy/utils.py",
]
spatialTransDeps = [
"tools/restrictMeshToTooth.py",
"tools/readAndFormat.py",
"tools/toothVars.py",
"tools/colorNumFrame.py",
"tools/plyRead.py",
"tools/getRegistration.py",
"tools/preprocess_point_cloud.py"
]
rafDeps = ["tools/readAndFormat.py"]


###############################################################################
##################################BEGIN RULES##################################
###############################################################################

###############################################################################
#execution rules
##########


#rule specifying what is required to exist
rule all:
    input:
        #require the following things to exist
        #the wildcard {name} and what it stands for (given by the second expant arg) is passed
        #to any rule associated with this file
        #
        #master arches
        #
        masterArchesDir + "masterArch1/mA1Full.ply",
        #
        #iowaRme:
        #
        #upper scans converted from the original stls
        #preD
        expand(preDFullScanDir + "{preDPat}u_preD.ply", preDPat = patNamesPreD),
        #fin
        expand(finFullScanDir + "{finPat}u_fin.ply", finPat = patNamesFin),
        #
        #process teeth3ds
        #
        expand(teeth3dsFullDir + "{teeth3dsName}_U.ply", teeth3dsName = patNames3ds),
        expand(teeth3dsFullCSDir + "{teeth3dsName}_U_cS.ply", teeth3dsName = patNames3ds),
        expand(teeth3dsCSMastRotMatDir + "{teeth3dsName}_U_cS_mastRotMat.pkl", teeth3dsName = patNames3ds),
        expand(teeth3dsFullCSOriMastDir + "{teeth3dsName}_U_cSOriMast.ply", teeth3dsName = patNames3ds),
        expand(teeth3dsCSOriMastRemeshDir + "{teeth3dsName}_U_cSOriMastRemesh.ply", teeth3dsName = patNames3ds),
        #
        #process Iosseg
        #
        expand(iossegCleanUCSDir + "{iossegCleanUPat}_U_cS.ply", iossegCleanUPat = allIossegCleanUPats),
        expand(iossegCleanUCSMastRotMatDir + "{iossegCleanUPat}_U_cS_mastRotMat.pkl", iossegCleanUPat = allIossegCleanUPats),
        expand(iossegCleanUCSOriMastDir + "{iossegCleanUPat}_U_cSOriMast.ply", iossegCleanUPat = allIossegCleanUPats),
        #
        #train test splits
        #
        trainTestDir_t3dsIosseg_cSOriMast + "t3dsIosseg_cSOriMast_trainTestSplit.complete",
        #
        #process iowaExptTest
        #
        expand(iowaExpTestOrigFormPreDir + "{iowaExpTestPrePat}Pre_form.ply", iowaExpTestPrePat = iowaExpTestPatsPre),
        expand(iowaExpTestOrigFormPostDir + "{iowaExpTestPostPat}Post_form.ply", iowaExpTestPostPat = iowaExpTestPatsPost),
        expand(iowaExpTestOrigFormCSPreDir + "{iowaExpTestPrePat}Pre_formCS.ply", iowaExpTestPrePat = iowaExpTestPatsPre),
        expand(iowaExpTestOrigFormCSPostDir + "{iowaExpTestPostPat}Post_formCS.ply", iowaExpTestPostPat = iowaExpTestPatsPost),
        expand(iowaExpTestOrigFormCSPreMastRotMatDir + "{iowaExpTestPrePat}Pre_formCS_mastRotMat.pkl", iowaExpTestPrePat = iowaExpTestPatsPre),
        expand(iowaExpTestOrigFormCSPostMastRotMatDir + "{iowaExpTestPostPat}Post_formCS_mastRotMat.pkl", iowaExpTestPostPat = iowaExpTestPatsPost),
        expand(iowaExpTestOrigFormCSOriMastPreDir + "{iowaExpTestPrePat}Pre_formCSOriMast.ply", iowaExpTestPrePat = iowaExpTestPatsPre),
        expand(iowaExpTestOrigFormCSOriMastPostDir + "{iowaExpTestPostPat}Post_formCSOriMast.ply", iowaExpTestPostPat = iowaExpTestPatsPost),
        expand(iowaExpTestOrigFormCSOriMastRemeshPreDir + "{iowaExpTestPrePat}Pre_formCSOriMastRemesh.ply", iowaExpTestPrePat = iowaExpTestPatsPre),
        expand(iowaExpTestOrigFormCSOriMastRemeshPostDir + "{iowaExpTestPostPat}Post_formCSOriMastRemesh.ply", iowaExpTestPostPat = iowaExpTestPatsPost),
        #
        #iowaExpTest centroid size
        #
        #pre centroid size
        iowaExpTestCentSizeDir + "centSize_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/centSizePre.csv",
        iowaExpTestCentSizeDir + "centSize_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/archLengthPre.csv",
        #post centroid size
        iowaExpTestCentSizeDir + "centSize_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/centSizePost.csv",
        iowaExpTestCentSizeDir + "centSize_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/archLengthPost.csv",
        #
        #process iowaExpTestRA
        #
        expand(iowaExpTestRAFormPreDir + "{iowaExpTestRAPrePat}Pre_form.ply", iowaExpTestRAPrePat = iowaExpTestRAPatsPre),
        expand(iowaExpTestRAFormPostDir + "{iowaExpTestRAPostPat}Post_form.ply", iowaExpTestRAPostPat = iowaExpTestRAPatsPost),
        expand(iowaExpTestRAFormCSPreDir + "{iowaExpTestRAPrePat}Pre_formCS.ply", iowaExpTestRAPrePat = iowaExpTestRAPatsPre),
        expand(iowaExpTestRAFormCSPostDir + "{iowaExpTestRAPostPat}Post_formCS.ply", iowaExpTestRAPostPat = iowaExpTestRAPatsPost),
        expand(iowaExpTestRAFormCSPreMastRotMatDir + "{iowaExpTestRAPrePat}Pre_formCS_mastRotMat.pkl", iowaExpTestRAPrePat = iowaExpTestRAPatsPre),
        expand(iowaExpTestRAFormCSPostMastRotMatDir + "{iowaExpTestRAPostPat}Post_formCS_mastRotMat.pkl", iowaExpTestRAPostPat = iowaExpTestRAPatsPost),
        expand(iowaExpTestRAFormCSOriMastPreDir + "{iowaExpTestRAPrePat}Pre_formCSOriMast.ply", iowaExpTestRAPrePat = iowaExpTestRAPatsPre),
        expand(iowaExpTestRAFormCSOriMastPostDir + "{iowaExpTestRAPostPat}Post_formCSOriMast.ply", iowaExpTestRAPostPat = iowaExpTestRAPatsPost),
        expand(iowaExpTestRAFormCSOriMastRemeshPreDir + "{iowaExpTestRAPrePat}Pre_formCSOriMastRemesh.ply",  iowaExpTestRAPrePat = iowaExpTestRAPatsPre),
        expand(iowaExpTestRAFormCSOriMastRemeshPostDir + "{iowaExpTestRAPostPat}Post_formCSOriMastRemesh.ply", iowaExpTestRAPostPat = iowaExpTestRAPatsPost),
        #
        #superimp
        #
        #iowaExpTestRA, trans to superimp post on pre
        expand(iowaExpTestRATransDir + "{iowaExpTestRABothPats}SuperimpPostOnPreTrans_rugAnnot.pkl", iowaExpTestRABothPats = iowaExpTestRAPatsBoth),
        #iowaExpTestRA, post scans with annotated rugae superimposition transformation applied
        expand(iowaExpTestRASuperimpPostScanDir + "{iowaExpTestRABothPats}Post_formCSOriMast_rugAnnotSuperimp.ply", iowaExpTestRABothPats = iowaExpTestRAPatsBoth),
        expand(iowaExpTestRARemeshSuperimpPostScanDir + "{iowaExpTestRABothPats}Post_formCSOriMastRemesh_rugAnnotSuperimp.ply", iowaExpTestRABothPats = iowaExpTestRAPatsBoth),
        #iowaExpansion pre and post scan visualization htmls with no superimposition
        expand(iowaExpTestRANoSuperimpVisDir + "{iowaExpTestRABothPats}NoSuperimpVis_formCSOriMast.html", iowaExpTestRABothPats = iowaExpTestRAPatsBoth),
        #iowaExpansion pre and post scan visualization htmls with annotated rugae superimposition
        expand(iowaExpTestRASuperimpVisDir + "{iowaExpTestRABothPats}SuperimpVis_formCSOriMast.html", iowaExpTestRABothPats = iowaExpTestRAPatsBoth),
        #
        #segmentation
        #
        iowaExpTestSegDir + "segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/monitoringFiles/pre.complete",
        iowaExpTestSegDir + "segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/monitoringFiles/post.complete",
        #
        #spatial transformation matrices
        #
        expand(iowaExpTestSpatialTransDir + "{seg1Pat}SpatialTransMats.pkl", seg1Pat = iowaExpTestRAPatsBoth)


rule masterArches:
    input:
        masterArchesDir + "masterArch1/mA1Full.ply"

rule processIowaRme:
    input:
        #iowaRme:
        #upper scans converted from the original stls
        #preD
        expand(preDFullScanDir + "{preDPat}u_preD.ply", preDPat = patNamesPreD),
        #fin
        expand(finFullScanDir + "{finPat}u_fin.ply", finPat = patNamesFin)

rule processTeeth3ds:
    input:
        expand(teeth3dsFullDir + "{teeth3dsName}_U.ply", teeth3dsName = patNames3ds),
        expand(teeth3dsFullCSDir + "{teeth3dsName}_U_cS.ply", teeth3dsName = patNames3ds),
        expand(teeth3dsCSMastRotMatDir + "{teeth3dsName}_U_cS_mastRotMat.pkl", teeth3dsName = patNames3ds),
        expand(teeth3dsFullCSOriMastDir + "{teeth3dsName}_U_cSOriMast.ply", teeth3dsName = patNames3ds),
        expand(teeth3dsCSOriMastRemeshDir + "{teeth3dsName}_U_cSOriMastRemesh.ply", teeth3dsName = patNames3ds)

rule processIosseg:
    input:
        expand(iossegCleanUCSDir + "{iossegCleanUPat}_U_cS.ply", iossegCleanUPat = allIossegCleanUPats),
        expand(iossegCleanUCSMastRotMatDir + "{iossegCleanUPat}_U_cS_mastRotMat.pkl", iossegCleanUPat = allIossegCleanUPats),
        expand(iossegCleanUCSOriMastDir + "{iossegCleanUPat}_U_cSOriMast.ply", iossegCleanUPat = allIossegCleanUPats)

rule trainTestSplits:
    input:
        trainTestDir_t3dsIosseg_cSOriMast + "t3dsIosseg_cSOriMast_trainTestSplit.complete"

rule processIowaExpTest:
    input:
        expand(iowaExpTestOrigFormPreDir + "{iowaExpTestPrePat}Pre_form.ply", iowaExpTestPrePat = iowaExpTestPatsPre),
        expand(iowaExpTestOrigFormPostDir + "{iowaExpTestPostPat}Post_form.ply", iowaExpTestPostPat = iowaExpTestPatsPost),
        expand(iowaExpTestOrigFormCSPreDir + "{iowaExpTestPrePat}Pre_formCS.ply", iowaExpTestPrePat = iowaExpTestPatsPre),
        expand(iowaExpTestOrigFormCSPostDir + "{iowaExpTestPostPat}Post_formCS.ply", iowaExpTestPostPat = iowaExpTestPatsPost),
        expand(iowaExpTestOrigFormCSPreMastRotMatDir + "{iowaExpTestPrePat}Pre_formCS_mastRotMat.pkl", iowaExpTestPrePat = iowaExpTestPatsPre),
        expand(iowaExpTestOrigFormCSPostMastRotMatDir + "{iowaExpTestPostPat}Post_formCS_mastRotMat.pkl", iowaExpTestPostPat = iowaExpTestPatsPost),
        expand(iowaExpTestOrigFormCSOriMastPreDir + "{iowaExpTestPrePat}Pre_formCSOriMast.ply", iowaExpTestPrePat = iowaExpTestPatsPre),
        expand(iowaExpTestOrigFormCSOriMastPostDir + "{iowaExpTestPostPat}Post_formCSOriMast.ply", iowaExpTestPostPat = iowaExpTestPatsPost),
        expand(iowaExpTestOrigFormCSOriMastRemeshPreDir + "{iowaExpTestPrePat}Pre_formCSOriMastRemesh.ply", iowaExpTestPrePat = iowaExpTestPatsPre),
        expand(iowaExpTestOrigFormCSOriMastRemeshPostDir + "{iowaExpTestPostPat}Post_formCSOriMastRemesh.ply", iowaExpTestPostPat = iowaExpTestPatsPost)

rule iowaExpTestCentroidSize:
    input:
        #pre centroid size
        iowaExpTestCentSizeDir + "centSize_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/centSizePre.csv",
        iowaExpTestCentSizeDir + "centSize_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/archLengthPre.csv",
        #post centroid size
        iowaExpTestCentSizeDir + "centSize_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/centSizePost.csv",
        iowaExpTestCentSizeDir + "centSize_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/archLengthPost.csv"

rule processIowaExpTestRA:
    input:
        expand(iowaExpTestRAFormPreDir + "{iowaExpTestRAPrePat}Pre_form.ply", iowaExpTestRAPrePat = iowaExpTestRAPatsPre),
        expand(iowaExpTestRAFormPostDir + "{iowaExpTestRAPostPat}Post_form.ply", iowaExpTestRAPostPat = iowaExpTestRAPatsPost),
        expand(iowaExpTestRAFormCSPreDir + "{iowaExpTestRAPrePat}Pre_formCS.ply", iowaExpTestRAPrePat = iowaExpTestRAPatsPre),
        expand(iowaExpTestRAFormCSPostDir + "{iowaExpTestRAPostPat}Post_formCS.ply", iowaExpTestRAPostPat = iowaExpTestRAPatsPost),
        expand(iowaExpTestRAFormCSPreMastRotMatDir + "{iowaExpTestRAPrePat}Pre_formCS_mastRotMat.pkl", iowaExpTestRAPrePat = iowaExpTestRAPatsPre),
        expand(iowaExpTestRAFormCSPostMastRotMatDir + "{iowaExpTestRAPostPat}Post_formCS_mastRotMat.pkl", iowaExpTestRAPostPat = iowaExpTestRAPatsPost),
        expand(iowaExpTestRAFormCSOriMastPreDir + "{iowaExpTestRAPrePat}Pre_formCSOriMast.ply", iowaExpTestRAPrePat = iowaExpTestRAPatsPre),
        expand(iowaExpTestRAFormCSOriMastPostDir + "{iowaExpTestRAPostPat}Post_formCSOriMast.ply", iowaExpTestRAPostPat = iowaExpTestRAPatsPost),
        expand(iowaExpTestRAFormCSOriMastRemeshPreDir + "{iowaExpTestRAPrePat}Pre_formCSOriMastRemesh.ply",  iowaExpTestRAPrePat = iowaExpTestRAPatsPre),
        expand(iowaExpTestRAFormCSOriMastRemeshPostDir + "{iowaExpTestRAPostPat}Post_formCSOriMastRemesh.ply", iowaExpTestRAPostPat = iowaExpTestRAPatsPost),
        #iowaExpTestRA, trans to superimp post on pre
        expand(iowaExpTestRATransDir + "{iowaExpTestRABothPats}SuperimpPostOnPreTrans_rugAnnot.pkl", iowaExpTestRABothPats = iowaExpTestRAPatsBoth),
        #iowaExpTestRA, post scans with annotated rugae superimposition transformation applied
        expand(iowaExpTestRASuperimpPostScanDir + "{iowaExpTestRABothPats}Post_formCSOriMast_rugAnnotSuperimp.ply", iowaExpTestRABothPats = iowaExpTestRAPatsBoth),
        expand(iowaExpTestRARemeshSuperimpPostScanDir + "{iowaExpTestRABothPats}Post_formCSOriMastRemesh_rugAnnotSuperimp.ply", iowaExpTestRABothPats = iowaExpTestRAPatsBoth)

#rule for just superimposition work
rule superimp:
    input:
        #iowaExpTestRA, trans to superimp post on pre
        expand(iowaExpTestRATransDir + "{iowaExpTestRABothPats}SuperimpPostOnPreTrans_rugAnnot.pkl", iowaExpTestRABothPats = iowaExpTestRAPatsBoth),
        #iowaExpTestRA, post scans with annotated rugae superimposition transformation applied
        expand(iowaExpTestRASuperimpPostScanDir + "{iowaExpTestRABothPats}Post_formCSOriMast_rugAnnotSuperimp.ply", iowaExpTestRABothPats = iowaExpTestRAPatsBoth),
        expand(iowaExpTestRARemeshSuperimpPostScanDir + "{iowaExpTestRABothPats}Post_formCSOriMastRemesh_rugAnnotSuperimp.ply", iowaExpTestRABothPats = iowaExpTestRAPatsBoth),
        #iowaExpansion pre and post scan visualization htmls with no superimposition
        expand(iowaExpTestRANoSuperimpVisDir + "{iowaExpTestRABothPats}NoSuperimpVis_formCSOriMast.html", iowaExpTestRABothPats = iowaExpTestRAPatsBoth),
        #iowaExpansion pre and post scan visualization htmls with annotated rugae superimposition
        expand(iowaExpTestRASuperimpVisDir + "{iowaExpTestRABothPats}SuperimpVis_formCSOriMast.html", iowaExpTestRABothPats = iowaExpTestRAPatsBoth)

#test rule
#rule test:
#    threads: defaultThreads
#    resources:
#        queue="UI-GPU",
#        gpus=3
#    output:
#        "test_pwd.txt"
#    shell:
#        """
#        pwd > {output}
#        """


rule segmentation:
    input:
        expand(iowaExpTestRARemeshSuperimpPostScanDir + "{iowaExpTestRABothPats}Post_formCSOriMastRemesh_rugAnnotSuperimp.ply", iowaExpTestRABothPats = iowaExpTestRAPatsBoth),
        iowaExpTestSegDir + "segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/monitoringFiles/pre.complete",
        iowaExpTestSegDir + "segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/monitoringFiles/post.complete"

rule spatialTrans:
    input:
        expand(iowaExpTestSpatialTransDir + "{seg1Pat}SpatialTransMats.pkl", seg1Pat = iowaExpTestRAPatsBoth)

###############################################################################
#pipeline rules
##########

##########################################
#IOWARME

#cannot directly run "snakemake convertPreDStlToPly -c1" because the input uses a wildcard via the helper
#function that snakemake will not be able to understand without the context of the rule all
#there are ways around this but this is fine for now
rule convertPreDStlToPly:
    threads: defaultThreads
    input: 
        #using preD stl helper function which makes use of wildcards
        inFile = getOrigStlPreD,
        script = "tools/processes/stlToPly_noLabs.py",
        deps = stlConvertNoLabsDepends
    output:
        outFile = preDFullScanDir + "{preDPat}u_preD.ply"
    shell:
        """
        python {input.script} "{input.inFile}" "{output.outFile}"
        """

#iowaRme: convert original final scan stls to plys
rule convertFinStlToPly:
    threads: defaultThreads
    input:
        inFile = getOrigStlFin,
        script = "tools/processes/stlToPly_noLabs.py",
        deps = stlConvertNoLabsDepends
    output:
        outFile = finFullScanDir + "{finPat}u_fin.ply"
    shell:
        """
        python {input.script} "{input.inFile}" "{output.outFile}"
        """



##########################################
#TEETH3DS

#teeth3ds
#convert obj/json combos into plys
rule convertTeeth3dsObjToPly:
    threads: defaultThreads
    input:
        #using the helper function
        objFile = getOrig3dsObj,
        jsonFile = getOrig3dsJson,
        script = "tools/processes/convertTeeth3dsObjToPly.py",
        deps = convertTeeth3dsObjToPlyDeps
    output:
        outFile = teeth3dsFullDir + "{teeth3dsName}_U.ply"
    shell:
        """
        python {input.script} {input.objFile} {input.jsonFile} {output.outFile}
        """

#teeth3ds
#center and scaled
rule centerAndScaleTeeth3ds:
    threads: defaultThreads
    input:
        inPath = teeth3dsFullDir + "{teeth3dsName}_U.ply",
        script = "tools/processes/centerAndScale.py",
        deps = manipulateAndFormatPack2
    params:
        labs = True
    output:
        outPath = teeth3dsFullCSDir + "{teeth3dsName}_U_cS.ply"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath} {params.labs}
        """

#teeth3ds
#get rotation matrix to master arch
rule getRotToMastTeeth3ds:
    threads: defaultThreads
    input:
        inPath = teeth3dsFullCSDir + "{teeth3dsName}_U_cS.ply",
        script = "tools/processes/getRotMatToMasterArch.py",
        deps = getRotToMastDeps,
        masterArch = masterArchesDir + "masterArch1/mA1Full.ply"
    output:
        outPath = teeth3dsCSMastRotMatDir + "{teeth3dsName}_U_cS_mastRotMat.pkl"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath} {input.masterArch}
        """

#teeth3ds
#apply rotation matrix to center and scaled teeth3ds data
rule orientToMastTeeth3ds:
    threads: defaultThreads
    input:
        inPly = teeth3dsFullCSDir + "{teeth3dsName}_U_cS.ply",
        inMat = teeth3dsCSMastRotMatDir + "{teeth3dsName}_U_cS_mastRotMat.pkl",
        script = "tools/processes/rotate.py",
        deps = manipulateAndFormatPack2
    params:
        labs = True
    output:
        outPath = teeth3dsFullCSOriMastDir + "{teeth3dsName}_U_cSOriMast.ply"
    shell:
        """
        python {input.script} {input.inPly} {input.inMat} {output.outPath} {params.labs}
        """

#teeth3ds
#remesh scans that are rotated, centered, and scaled
rule remeshTeeth3ds:
    threads: defaultThreads
    input:
        inPath = teeth3dsFullCSOriMastDir + "{teeth3dsName}_U_cSOriMast.ply",
        script = "tools/processes/remesh.py",
        deps = remeshDeps
    params:
        labs = True
    output:
        outPath = teeth3dsCSOriMastRemeshDir + "{teeth3dsName}_U_cSOriMastRemesh.ply"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath} {params.labs}
        """


#################################################
#IOWAEXPTEST

#iowaExpTest
#format raw itero scans pre
rule formatRawIowaExpTestPre:
    threads: defaultThreads
    input:
        inPath = getIowaExpTestOrigPre,
        script = "tools/processes/formatRawIteroPly.py",
        deps = formatNoLabPlyDeps
    output:
        outPath = iowaExpTestOrigFormPreDir + "{iowaExpTestPrePat}Pre_form.ply"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath}
        """

#iowaExpTest
#format raw itero scans post
rule formatRawIowaExpTestPost:
    threads: defaultThreads
    input:
        inPath = getIowaExpTestOrigPost,
        script = "tools/processes/formatRawIteroPly.py",
        deps = formatNoLabPlyDeps
    output:
        outPath = iowaExpTestOrigFormPostDir + "{iowaExpTestPostPat}Post_form.ply"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath}
        """

#iowaExpTest
#cetner and scale pre
rule centerScaleIowaExpTestPre:
    threads: defaultThreads
    input:
        inPath = iowaExpTestOrigFormPreDir + "{iowaExpTestPrePat}Pre_form.ply",
        script = "tools/processes/centerAndScale.py",
        deps = manipulateAndFormatPack2
    params:
        labs = False
    output:
        outPath = iowaExpTestOrigFormCSPreDir + "{iowaExpTestPrePat}Pre_formCS.ply"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath} {params.labs}
        """

#iowaExpTest
#cetner and scale post
rule centerScaleIowaExpTestPost:
    threads: defaultThreads
    input:
        inPath = iowaExpTestOrigFormPostDir + "{iowaExpTestPostPat}Post_form.ply",
        script = "tools/processes/centerAndScale.py",
        deps = manipulateAndFormatPack2
    params:
        labs = False
    output:
        outPath = iowaExpTestOrigFormCSPostDir + "{iowaExpTestPostPat}Post_formCS.ply"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath} {params.labs}
        """

#iowaExpTest
#get rotation matrix to master arch, pre
rule getRotToMastIowaExpTestPre:
    threads: defaultThreads
    input:
        inPath = iowaExpTestOrigFormCSPreDir + "{iowaExpTestPrePat}Pre_formCS.ply",
        script = "tools/processes/getRotMatToMasterArch.py",
        deps = getRotToMastDeps,
        masterArch = masterArchesDir + "masterArch1/mA1Full.ply"
    output:
        outPath = iowaExpTestOrigFormCSPreMastRotMatDir + "{iowaExpTestPrePat}Pre_formCS_mastRotMat.pkl"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath} {input.masterArch}
        """

#iowaExpTest
#get rotation matrix to master arch, post
rule getRotToMastIowaExpTestPost:
    threads: defaultThreads
    input:
        inPath = iowaExpTestOrigFormCSPostDir + "{iowaExpTestPostPat}Post_formCS.ply",
        script = "tools/processes/getRotMatToMasterArch.py",
        deps = getRotToMastDeps,
        masterArch = masterArchesDir + "masterArch1/mA1Full.ply"
    output:
        outPath = iowaExpTestOrigFormCSPostMastRotMatDir + "{iowaExpTestPostPat}Post_formCS_mastRotMat.pkl"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath} {input.masterArch}
        """

#iowaExpTest
#apply rotation matrix to center and scaled iowaExpTest pre data
rule orientToMastIowaExpTestPre:
    threads: defaultThreads
    input:
        inPly = iowaExpTestOrigFormCSPreDir + "{iowaExpTestPrePat}Pre_formCS.ply",
        inMat = iowaExpTestOrigFormCSPreMastRotMatDir + "{iowaExpTestPrePat}Pre_formCS_mastRotMat.pkl",
        script = "tools/processes/rotate.py",
        deps = manipulateAndFormatPack2
    params:
        labs = False
    output:
        outPath = iowaExpTestOrigFormCSOriMastPreDir + "{iowaExpTestPrePat}Pre_formCSOriMast.ply"
    shell:
        """
        python {input.script} {input.inPly} {input.inMat} {output.outPath} {params.labs}
        """

#iowaExpTest
#apply rotation matrix to center and scaled iowaExpTest post data
rule orientToMastIowaExpTestPost:
    threads: defaultThreads
    input:
        inPly = iowaExpTestOrigFormCSPostDir + "{iowaExpTestPostPat}Post_formCS.ply",
        inMat = iowaExpTestOrigFormCSPostMastRotMatDir + "{iowaExpTestPostPat}Post_formCS_mastRotMat.pkl",
        script = "tools/processes/rotate.py",
        deps = manipulateAndFormatPack2
    params:
        labs = False
    output:
        outPath = iowaExpTestOrigFormCSOriMastPostDir + "{iowaExpTestPostPat}Post_formCSOriMast.ply"
    shell:
        """
        python {input.script} {input.inPly} {input.inMat} {output.outPath} {params.labs}
        """

#iowaExpTest
#remesh scans that are rotated, centered, and scaled for pre data
rule remeshIowaExpTestPre:
    threads: defaultThreads
    input:
        inPath = iowaExpTestOrigFormCSOriMastPreDir + "{iowaExpTestPrePat}Pre_formCSOriMast.ply",
        script = "tools/processes/remesh.py",
        deps = remeshDeps
    params:
        labs = False
    output:
        outPath = iowaExpTestOrigFormCSOriMastRemeshPreDir + "{iowaExpTestPrePat}Pre_formCSOriMastRemesh.ply"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath} {params.labs}
        """

#iowaExpTest
#remesh scans that are rotated, centered, and scaled for post data
rule remeshIowaExpTestPost:
    threads: defaultThreads
    input:
        inPath = iowaExpTestOrigFormCSOriMastPostDir + "{iowaExpTestPostPat}Post_formCSOriMast.ply",
        script = "tools/processes/remesh.py",
        deps = remeshDeps
    params:
        labs = False
    output:
        outPath = iowaExpTestOrigFormCSOriMastRemeshPostDir + "{iowaExpTestPostPat}Post_formCSOriMastRemesh.ply"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath} {params.labs}
        """




#################################################
#IOWAEXPTEST RUGAE ANNOT

#iowaExpTestRA
#format and vert labs to face labs, rugae annot iowa exp test scans pre
rule formatAndLabsIowaExpTestRAPre:
    threads: defaultThreads
    input:
        inPath = getIowaExpTestRAPre,
        script = "tools/processes/ccRugaeAnnotVertToFaceLab.py",
        deps = formatNoLabPlyDeps
    output:
        outPath = iowaExpTestRAFormPreDir + "{iowaExpTestRAPrePat}Pre_form.ply"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath}
        """

#iowaExpTestRA
#format and vert labs to face labs, rugae annot iowa exp test scans post
rule formatAndLabsIowaExpTestRAPost:
    threads: defaultThreads
    input:
        inPath = getIowaExpTestRAPost,
        script = "tools/processes/ccRugaeAnnotVertToFaceLab.py",
        deps = formatNoLabPlyDeps
    output:
        outPath = iowaExpTestRAFormPostDir + "{iowaExpTestRAPostPat}Post_form.ply"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath}
        """

#iowaExpTestRA
#cetner and scale pre
rule centerScaleIowaExpTestRAPre:
    threads: defaultThreads
    input:
        inPath = iowaExpTestRAFormPreDir + "{iowaExpTestRAPrePat}Pre_form.ply",
        script = "tools/processes/centerAndScale.py",
        deps = manipulateAndFormatPack2
    params:
        labs = True
    output:
        outPath = iowaExpTestRAFormCSPreDir + "{iowaExpTestRAPrePat}Pre_formCS.ply"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath} {params.labs}
        """

#iowaExpTestRA
#cetner and scale post
rule centerScaleIowaExpTestRAPost:
    threads: defaultThreads
    input:
        inPath = iowaExpTestRAFormPostDir + "{iowaExpTestRAPostPat}Post_form.ply",
        script = "tools/processes/centerAndScale.py",
        deps = manipulateAndFormatPack2
    params:
        labs = True
    output:
        outPath = iowaExpTestRAFormCSPostDir + "{iowaExpTestRAPostPat}Post_formCS.ply"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath} {params.labs}
        """

#iowaExpTestRA
#get rotation matrix to master arch, pre
rule getRotToMastIowaExpTestRAPre:
    threads: defaultThreads
    input:
        inPath = iowaExpTestRAFormCSPreDir + "{iowaExpTestRAPrePat}Pre_formCS.ply",
        script = "tools/processes/getRotMatToMasterArch.py",
        deps = getRotToMastDeps,
        masterArch = masterArchesDir + "masterArch1/mA1Full.ply"
    output:
        outPath = iowaExpTestRAFormCSPreMastRotMatDir + "{iowaExpTestRAPrePat}Pre_formCS_mastRotMat.pkl"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath} {input.masterArch}
        """

#iowaExpTestRA
#get rotation matrix to master arch, post
rule getRotToMastIowaExpTestRAPost:
    threads: defaultThreads
    input:
        inPath = iowaExpTestRAFormCSPostDir + "{iowaExpTestRAPostPat}Post_formCS.ply",
        script = "tools/processes/getRotMatToMasterArch.py",
        deps = getRotToMastDeps,
        masterArch = masterArchesDir + "masterArch1/mA1Full.ply"
    output:
        outPath = iowaExpTestRAFormCSPostMastRotMatDir + "{iowaExpTestRAPostPat}Post_formCS_mastRotMat.pkl"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath} {input.masterArch}
        """

#iowaExpTestRA
#apply rotation matrix to center and scaled iowaExpTest pre data
rule orientToMastIowaExpTestRAPre:
    threads: defaultThreads
    input:
        inPly = iowaExpTestRAFormCSPreDir + "{iowaExpTestRAPrePat}Pre_formCS.ply",
        inMat = iowaExpTestRAFormCSPreMastRotMatDir + "{iowaExpTestRAPrePat}Pre_formCS_mastRotMat.pkl",
        script = "tools/processes/rotate.py",
        deps = manipulateAndFormatPack2
    params:
        labs = True
    output:
        outPath = iowaExpTestRAFormCSOriMastPreDir + "{iowaExpTestRAPrePat}Pre_formCSOriMast.ply"
    shell:
        """
        python {input.script} {input.inPly} {input.inMat} {output.outPath} {params.labs}
        """

#iowaExpTestRA
#apply rotation matrix to center and scaled iowaExpTest post data
rule orientToMastIowaExpTestRAPost:
    threads: defaultThreads
    input:
        inPly = iowaExpTestRAFormCSPostDir + "{iowaExpTestRAPostPat}Post_formCS.ply",
        inMat = iowaExpTestRAFormCSPostMastRotMatDir + "{iowaExpTestRAPostPat}Post_formCS_mastRotMat.pkl",
        script = "tools/processes/rotate.py",
        deps = manipulateAndFormatPack2
    params:
        labs = True
    output:
        outPath = iowaExpTestRAFormCSOriMastPostDir + "{iowaExpTestRAPostPat}Post_formCSOriMast.ply"
    shell:
        """
        python {input.script} {input.inPly} {input.inMat} {output.outPath} {params.labs}
        """

#iowaExpTestRA
#remesh scans that are rotated, centered, and scaled for pre data
rule remeshIowaExpTestRAPre:
    threads: defaultThreads
    input:
        inPath = iowaExpTestRAFormCSOriMastPreDir + "{iowaExpTestRAPrePat}Pre_formCSOriMast.ply",
        script = "tools/processes/remesh.py",
        deps = remeshDeps
    params:
        labs = True
    output:
        outPath = iowaExpTestRAFormCSOriMastRemeshPreDir + "{iowaExpTestRAPrePat}Pre_formCSOriMastRemesh.ply"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath} {params.labs}
        """

#iowaExpTestRA
#remesh scans that are rotated, centered, and scaled for post data
rule remeshIowaExpTestRAPost:
    threads: defaultThreads
    input:
        inPath = iowaExpTestRAFormCSOriMastPostDir + "{iowaExpTestRAPostPat}Post_formCSOriMast.ply",
        script = "tools/processes/remesh.py",
        deps = remeshDeps
    params:
        labs = True
    output:
        outPath = iowaExpTestRAFormCSOriMastRemeshPostDir + "{iowaExpTestRAPostPat}Post_formCSOriMastRemesh.ply"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath} {params.labs}
        """





#################################################
#IOWAEXPTEST RUGAE ANNOT SUPERIMPOSITION


#iowaExpTest RA superimp
#superimposition on annotated rugae region
rule superimpIowaExpTestRA:
    threads: defaultThreads
    input:
        #using the helper function
        prePath = iowaExpTestRAFormCSOriMastPreDir + "{iowaExpTestRABothPats}Pre_formCSOriMast.ply",
        postPath = iowaExpTestRAFormCSOriMastPostDir + "{iowaExpTestRABothPats}Post_formCSOriMast.ply",
        script = "superimposition/rugaeAnnotRegistration.py",
        deps = superimpIowaExpAnnotRugaeDeps
    output:
        transPath = iowaExpTestRATransDir + "{iowaExpTestRABothPats}SuperimpPostOnPreTrans_rugAnnot.pkl",
        outPlyPath = iowaExpTestRASuperimpPostScanDir + "{iowaExpTestRABothPats}Post_formCSOriMast_rugAnnotSuperimp.ply"
    shell:
        """
        python {input.script} {input.prePath} {input.postPath} {output.transPath} {output.outPlyPath}
        """

#iowaExpTest RA superimp
#apply superimposition trans to remeshed data
rule superimpIowaExpTestRARemesh:
    threads: defaultThreads
    input:
        inPly = iowaExpTestRAFormCSOriMastRemeshPostDir + "{iowaExpTestRABothPats}Post_formCSOriMastRemesh.ply",
        inMat = iowaExpTestRATransDir + "{iowaExpTestRABothPats}SuperimpPostOnPreTrans_rugAnnot.pkl",
        script = "tools/processes/rotate.py",
        deps = manipulateAndFormatPack2
    params:
        labs = True
    output:
        outPath = iowaExpTestRARemeshSuperimpPostScanDir + "{iowaExpTestRABothPats}Post_formCSOriMastRemesh_rugAnnotSuperimp.ply"
    shell:
        """
        python {input.script} {input.inPly} {input.inMat} {output.outPath} {params.labs}
        """


#iowaExpTest RA superimp
#html visuals for pre and post scans with no superimposition
rule makeNoSuperimpVisIowaExpTestRA:
    threads: defaultThreads
    input:
        prePath = iowaExpTestRAFormCSOriMastPreDir + "{iowaExpTestRABothPats}Pre_formCSOriMast.ply",
        postPath = iowaExpTestRAFormCSOriMastPostDir + "{iowaExpTestRABothPats}Post_formCSOriMast.ply",
        script = "superimposition/createSuperimpHtmlVisuals.py"
    params:
        color_ = "red",
    output:
        visHtml = iowaExpTestRANoSuperimpVisDir + "{iowaExpTestRABothPats}NoSuperimpVis_formCSOriMast.html"
    shell:
        """
        python {input.script} {input.prePath} {input.postPath} {params.color_} {output.visHtml}
        """

#iowaExpTest RA superimp
#html visuals for pre and post scans with annotated rugae superimposition
rule makeSuperimpVisIowaExpTestRA:
    threads: defaultThreads
    input:
        prePath = iowaExpTestRAFormCSOriMastPreDir + "{iowaExpTestRABothPats}Pre_formCSOriMast.ply",
        postPath = iowaExpTestRASuperimpPostScanDir + "{iowaExpTestRABothPats}Post_formCSOriMast_rugAnnotSuperimp.ply",
        script = "superimposition/createSuperimpHtmlVisuals.py"
    params:
        color_ = "green",
    output:
        visHtml = iowaExpTestRASuperimpVisDir + "{iowaExpTestRABothPats}SuperimpVis_formCSOriMast.html"
    shell:
        """
        python {input.script} {input.prePath} {input.postPath} {params.color_} {output.visHtml}
        """




#############################
#IOSSEG

#iosseg
#center and scale
rule centerAndScaleIosseg:
    threads: defaultThreads
    input:
        inPath = getIossegCleanU,
        script = "tools/processes/centerAndScale.py",
        deps = manipulateAndFormatPack2
    params:
        labs = True
    output:
        outPath = iossegCleanUCSDir + "{iossegCleanUPat}_U_cS.ply"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath} {params.labs}
        """

#iosseg
#get rotation matrix to master arch
rule getRotToMastIosseg:
    threads: defaultThreads
    input:
        inPath = iossegCleanUCSDir + "{iossegCleanUPat}_U_cS.ply",
        script = "tools/processes/getRotMatToMasterArch.py",
        deps = getRotToMastDeps,
        masterArch = masterArchesDir + "masterArch1/mA1Full.ply"
    output:
        outPath = iossegCleanUCSMastRotMatDir + "{iossegCleanUPat}_U_cS_mastRotMat.pkl"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath} {input.masterArch}
        """

#iosseg
#apply rotation matrix to center and scaled teeth3ds data
rule orientToMastIosseg:
    threads: defaultThreads
    input:
        inPly = iossegCleanUCSDir + "{iossegCleanUPat}_U_cS.ply",
        inMat = iossegCleanUCSMastRotMatDir + "{iossegCleanUPat}_U_cS_mastRotMat.pkl",
        script = "tools/processes/rotate.py",
        deps = manipulateAndFormatPack2
    params:
        labs = True
    output:
        outPath = iossegCleanUCSOriMastDir + "{iossegCleanUPat}_U_cSOriMast.ply"
    shell:
        """
        python {input.script} {input.inPly} {input.inMat} {output.outPath} {params.labs}
        """



#################################
#TRAIN TEST SETS

#tain test set for teeth3dsIosseg_cSOriMast
rule trainTestSplit_teeth3dsIosseg_cSOriMast:
    threads: defaultThreads
    input:
        #require all CSOriMastRemesh teeth3ds files, but they are not input into the script
        t3ds_cSOriMastRemesh = expand(teeth3dsCSOriMastRemeshDir + "{teeth3dsName}_U_cSOriMastRemesh.ply", teeth3dsName = patNames3ds),
        #require all cSRot teeth3ds files, but they are not input into the script
        ios_cSOriMast = expand(iossegCleanUCSOriMastDir + "{iossegCleanUPat}_U_cSOriMast.ply", iossegCleanUPat = allIossegCleanUPats),
        script = "tools/processes/trainTestSets/split_teeth3dsIosseg_cSOriMast.py"
    params:
        newDir = trainTestDir_t3dsIosseg_cSOriMast,
        t3dsDir = teeth3dsCSOriMastRemeshDir,
        iosDir = iossegCleanUCSOriMastDir
    output:
        #monitoring done by sentinel file
        touch(trainTestDir_t3dsIosseg_cSOriMast + "t3dsIosseg_cSOriMast_trainTestSplit.complete")
    shell:
        """
        python {input.script} {params.newDir} {params.t3dsDir} {params.iosDir}
        """

################################
#MASTER ARCHES

#create master arches
rule createMasterArches:
    threads: defaultThreads
    input:
        script = "tools/processes/createMasterArches.py",
        deps = manipulateAndFormatPack
    output:
        m1OutPath = masterArchesDir + "masterArch1/mA1Full.ply"
    shell:
        """
        python {input.script} {output.m1OutPath}
        """

#################################
#SEGMENTATION

#iowaExpTest
rule segIowaExpTestRAPre:
    threads: 16
    resources:
        queue="UI-GPU",
        gpus=3
    input:
        inScans = expand(iowaExpTestRAFormCSOriMastRemeshPreDir + "{iowaExpTestRAPrePat}Pre_formCSOriMastRemesh.ply", iowaExpTestRAPrePat = iowaExpTestRAPatsPre),
        script = "prediction/ftePrediction.py",
        deps = segDeps
    params:
        inDir_ = iowaExpTestRAFormCSOriMastRemeshPreDir,
        outDir_ = iowaExpTestSegDir + "segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/pre/",
        predMatDir = iowaExpTestSegDir + "segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/pre/predMat/"
    output:
        #monitoring done by sentinel file
        touch(iowaExpTestSegDir + "segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/monitoringFiles/pre.complete")
    shell:
        """
        python {input.script} {params.inDir_} {params.outDir_} {params.predMatDir}
        """

#iowaExpTest
rule segIowaExpTestRAPost:
    threads: 16
    resources:
        queue="UI-GPU",
        gpus=3
    input:
        inScans = expand(iowaExpTestRARemeshSuperimpPostScanDir + "{iowaExpTestRABothPats}Post_formCSOriMastRemesh_rugAnnotSuperimp.ply", iowaExpTestRABothPats = iowaExpTestRAPatsBoth),
        script = "prediction/ftePrediction.py",
        deps = segDeps
    params:
        inDir_ = iowaExpTestRARemeshSuperimpPostScanDir,
        outDir_ = iowaExpTestSegDir + "segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/post/",
        predMatDir = iowaExpTestSegDir + "segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/post/predMat/"
    output:
        #monitoring done by sentinel file
        touch(iowaExpTestSegDir + "segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/monitoringFiles/post.complete")
    shell:
        """
        python {input.script} {params.inDir_} {params.outDir_} {params.predMatDir}
        """


##################################
#CENTROID SIZE AND ARCH LENGTH

#iowaExpTestRA
#get centroid size for segmented pre scans
rule getCentSizeIowaExpTestPre:
    threads: defaultThreads
    input:
        sentinelFile = iowaExpTestSegDir + "segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/monitoringFiles/pre.complete",
        script = "tools/processes/calculateCentroidSize.py",
        deps = calcCentSizeDeps
    params:
        dir_ = iowaExpTestSegDir + "segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/pre/"
    output:
        outCent = iowaExpTestCentSizeDir + "centSize_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/centSizePre.csv",
        outLength = iowaExpTestCentSizeDir + "centSize_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/archLengthPre.csv"
    shell:
        """
        python {input.script} {params.dir_} {output.outCent} {output.outLength}
        """

#iowaExpTestRA
#get centroid size for segmented post scans
rule getCentSizeIowaExpTestPost:
    threads: defaultThreads
    input:
        sentinelFile = iowaExpTestSegDir + "segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/monitoringFiles/post.complete",
        script = "tools/processes/calculateCentroidSize.py",
        deps = calcCentSizeDeps
    params:
        dir_ = iowaExpTestSegDir + "segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/post/"
    output:
        outCent = iowaExpTestCentSizeDir + "centSize_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/centSizePost.csv",
        outLength = iowaExpTestCentSizeDir + "centSize_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/archLengthPost.csv"
    shell:
        """
        python {input.script} {params.dir_} {output.outCent} {output.outLength}
        """






##################################
#get rigid registration spatial transformation matrix

rule getSpatialTransMats_iowaExpTestRA:
    threads: defaultThreads
    input:
        preSentinel = iowaExpTestSegDir + "segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/monitoringFiles/pre.complete",
        postSentinel = iowaExpTestSegDir + "segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/monitoringFiles/post.complete",
        prePath = getIowaExpTestSeg1Pre,
        postPath = getIowaExpTestSeg1Post,
        script = "movement/getSpatialTransMats.py",
        deps = spatialTransDeps
    output:
        outPath = iowaExpTestSpatialTransDir + "{seg1Pat}SpatialTransMats.pkl"
    shell:
        """
        python {input.script} {input.prePath} {input.postPath} {output.outPath}
        """






#####################################
#testing c++ stuff
#THIS IS A PERFECT EXAMPLE OF HOW TO HAVE A SINGLE RULE RUN FOR MULTIPLE WILDCARDS
#AND NOT HAVE TO MAKE SEPARATE RULES FOR EACH WILDCARD COMBO
localDescrDir1 = grantDir + "iowaExpTest/localDescriptors/rugAnnotForm_cSOriMastRemesh_localDescr/"

#phase and patient combinations
iowaExpTestRAPhasePatCombos = (
    [("pre", "Pre", pat) for pat in iowaExpTestRAPatsPre]
    + [("post", "Post", pat) for pat in iowaExpTestRAPatsPre]
    )
#raw csv output files
localDescr1OutputsCsv = [
    localDescrDir1 + f"{phase}/{pat}{CPhase}_localDescr.csv"
    for phase, CPhase, pat in iowaExpTestRAPhasePatCombos
]
#raw ply output files
localDescr1OutputsPly = [
    localDescrDir1 + f"{phase}/{pat}{CPhase}_localDescr.ply"
    for phase, CPhase, pat in iowaExpTestRAPhasePatCombos
]

#labeled csvs 
labeledDescr1Csv = [
    localDescrDir1 + f"{phase}LabeledCsv/{pat}{CPhase}_localDescrLabel.csv"
    for phase, CPhase, pat in iowaExpTestRAPhasePatCombos
]

rule localDescriptors:
    input:
        "tools/cpp/localDescriptors/build/localDescriptors",
        localDescr1OutputsCsv,
        localDescr1OutputsPly,
        labeledDescr1Csv


#this was in before and seemed necessary when switching from local machine to hpc
#rm tools/cpp/localDescriptors/build/CMakeCache.txt
#
#touch("tools/cpp/localDescriptors/localDescriptors_cmake.complete")
rule compileCmakeLocalDescriptors:
    threads: 1
    resources:
        queue="all.q"
    input:
        cmake = "tools/cpp/localDescriptors/CMakeLists.txt",
        cppFile = "tools/cpp/localDescriptors/localDescriptors.cpp"
    output:
        #this function is created in the process but not explicitly used in the bash code
        outFunction = "tools/cpp/localDescriptors/build/localDescriptors"
    shell:
        """
        cmake -S tools/cpp/localDescriptors -B tools/cpp/localDescriptors/build
        cmake --build tools/cpp/localDescriptors/build
        """

rule extractLocalDescriptors:
    threads: defaultThreads
    input:
        inFile = iowaExpTestRAFormCSOriMastRemeshDir + "{phase}/{pat}{CPhase}_formCSOriMastRemesh.ply",
        function = "tools/cpp/localDescriptors/build/localDescriptors"
    output:
        outPly = localDescrDir1 + "{phase}/{pat}{CPhase}_localDescr.ply",
        outCsv = localDescrDir1 + "{phase}/{pat}{CPhase}_localDescr.csv"
    shell:
        """
        {input.function} {input.inFile} {output.outPly} {output.outCsv}
        """

rule localDescrLabeledCsv:
    threads: defaultThreads
    input:
        meshPath = iowaExpTestRAFormCSOriMastRemeshDir + "{phase}/{pat}{CPhase}_formCSOriMastRemesh.ply",
        ldPath = localDescrDir1 + "{phase}/{pat}{CPhase}_localDescr.csv",
        script = "rugaeDetect/processes/produceLabeledDescriptorCsv.py",
        deps = rafDeps
    output:
        outPath = localDescrDir1 + "{phase}LabeledCsv/{pat}{CPhase}_localDescrLabel.csv"
    shell:
        """
        python {input.script} {input.meshPath} {input.ldPath} {output.outPath}
        """

