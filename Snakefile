import os
import re

#tutorial: https://www.youtube.com/watch?v=r9PWnEmz_tc&t=1247s

#some rules
#cannot have empty lines in a rule

###############################################################################
#helper functions
##########
#for getting the directory dictionaries used in the initial helper functions for raw data
def patNamesAndPathDict(dir_, pattern = r'^pat[0-9]{3}', captureGroup = 0):
    
    #get files and make file paths
    files = os.listdir(dir_)
    paths = [dir_ + file_ for file_ in files]
    
    #extract patient names
    patNames = [re.search(pattern, i).group(captureGroup) for i in files]
    
    #create path dictionary
    pathDict = dict(zip(patNames, paths))
    
    return patNames, pathDict

###############################################################################
#directories
##########

#iowaRme
#original stl directory
origStlDir = "K:/iowaRme/preDelivAndFinalScans/originalStl/"

#iowaRme:
#preD files and directories
preDFullScanDir = "K:/iowaRme/preDelivAndFinalScans/preDelivScanU/fullScans/"
#fin files and directories
finFullScanDir = "K:/iowaRme/preDelivAndFinalScans/finalScanU/fullScans/"

#iowaExpansion
#full, rugae annotated scans
iowaExpFullAnnotPreDir = "K:/iowaExpansion/fullRugaeAnnotScans/pre/"
iowaExpFullAnnotPostDir = "K:/iowaExpansion/fullRugaeAnnotScans/post/"
#segmentation model ready scans
iowaExpSegReadyPreDir = "K:/iowaExpansion/segReadyScans/pre/"
iowaExpSegReadyPostDir = "K:/iowaExpansion/segReadyScans/post/"
#TEMP
iowaExpSegReadyPreDir2 = "K:/iowaExpansion/segReadyScans2/pre/"
iowaExpSegReadyPostDir2 = "K:/iowaExpansion/segReadyScans2/post/"
#iowaExpTest
#original scans from itero
iowaExpTestOrigPreDir = "K:/iowaExpTest/scanData/orig/pre/"
iowaExpTestOrigPostDir = "K:/iowaExpTest/scanData/orig/post/"
iowaExpTestOrigFormPreDir = "K:/iowaExpTest/scanData/origForm/pre/"
iowaExpTestOrigFormPostDir = "K:/iowaExpTest/scanData/origForm/post/"
iowaExpTestOrigFormCSPreDir = "K:/iowaExpTest/scanData/origForm_cS/pre/"
iowaExpTestOrigFormCSPostDir = "K:/iowaExpTest/scanData/origForm_cS/post/"
iowaExpTestOrigFormCSPreMastRotMatDir = "K:/iowaExpTest/rotationMatrices/origForm_cS_mastRotMat/pre/"
iowaExpTestOrigFormCSPostMastRotMatDir = "K:/iowaExpTest/rotationMatrices/origForm_cS_mastRotMat/post/"
iowaExpTestOrigFormCSOriMastPreDir = "K:/iowaExpTest/scanData/origForm_cSOriMast/pre/"
iowaExpTestOrigFormCSOriMastPostDir = "K:/iowaExpTest/scanData/origForm_cSOriMast/post/"
iowaExpTestOrigFormCSOriMastRemeshPreDir = "K:/iowaExpTest/scanData/origForm_cSOriMastRemesh/pre/"
iowaExpTestOrigFormCSOriMastRemeshPostDir = "K:/iowaExpTest/scanData/origForm_cSOriMastRemesh/post/"
#segmeted directories
iowaExpTestSegPreDir = "K:/iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/origForm_cSOriMastRemesh/pre/"
iowaExpTestSegPostDir = "K:/iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/origForm_cSOriMastRemesh/post/"
#centroid size directiories
iowaExpTestCentSizeDir = "K:/iowaExpTest/centroidSize/"

#directories for superimposition transformations
iowaExpRugaeTransDir = "K:/iowaExpansion/superimposition/transformations/annotRugaeTrans/"
#directory for post scans with superimposition transformation applied
iowaExpRugaeSuperimpPostScanDir = "K:/iowaExpansion/superimposition/transPostScan/annotRugaeTransPostScan/"
#directory for html visuals of pre and post scans without superimposition
iowaExpNoSuperimpVisDir = "K:/iowaExpansion/superimposition/visuals/noSuperimp/"
#directory for html visuals of pre and post scans with annoted rugae superimposition
iowaExpAnnotRugaeSuperimpVisDir = "K:/iowaExpansion/superimposition/visuals/annotRugaeSuperimp/"

#teeth3ds
#full plys
teeth3dsFullDir = "K:/teeth3DS/scanData/upperPly/"
teeth3dsFullCSDir = "K:/teeth3DS/scanData/upperPly_cS/"
teeth3dsCSMastRotMatDir = "K:/teeth3DS/rotationMatrices/upperPly_cS_mastRotMat/"
teeth3dsFullCSOriMastDir = "K:/teeth3DS/scanData/upperPly_cSOriMast/"
teeth3dsCSOriMastRemeshDir = "K:/teeth3DS/scanData/upperPly_cSOriMastRemesh/"
#original files
teeth3dsOrigFilesDir = "K:/teeth3DS/scanData/upper/"

#iosseg
#plys
iossegCleanUDir = "K:/IOSSegData/scanData/cleanU/"
iossegCleanUCSDir = "K:/IOSSegData/scanData/cleanU_cS/"
iossegCleanUCSMastRotMatDir = "K:/IOSSegData/rotationMatrices/cleanU_cS_mastRotMat/"
iossegCleanUCSOriMastDir = "K:/IOSSegData/scanData/cleanU_cSOriMast/"

#master arches
masterArchesDir = "K:/masterArches/"

#train test sets
trainTestDir_t3dsIosseg_cSOriMast = "K:/trainTestSets/t3dsIosseg_cSOriMast/"


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
#iowaExpansion
#get patient names and create directory dictionary
#iowaExpPatsPre, iowaExpFullAnnotPathDictPre = patNamesAndPathDict(iowaExpFullAnnotPreDir)
#iowaExpPatsPost, iowaExpFullAnnotPathDictPost = patNamesAndPathDict(iowaExpFullAnnotPostDir)
#create helper functions for using the raw data
#def getIowaExpFullAnnotPre(wildcards):
#    return iowaExpFullAnnotPathDictPre[wildcards.iowaExpPrePat]
#def getIowaExpFullAnnotPost(wildcards):
#    return iowaExpFullAnnotPathDictPost[wildcards.iowaExpPostPat]

###############################################################################
#iowaExpTest
#get patient names and create directory dictionary
iowaExpTestPatsPre, iowaExpTestOrigPathDictPre = patNamesAndPathDict(iowaExpTestOrigPreDir)
iowaExpTestPatsPost, iowaExpTestOrigPathDictPost = patNamesAndPathDict(iowaExpTestOrigPostDir)
#create helper functions for using the raw data
def getIowaExpTestOrigPre(wildcards):
    return iowaExpTestOrigPathDictPre[wildcards.iowaExpTestPrePat]
def getIowaExpTestOrigPost(wildcards):
    return iowaExpTestOrigPathDictPost[wildcards.iowaExpTestPostPat]


###############################################################################
#iowaExpansion
#patient names for just the patients with both a pre and a post
#iowaExpPatsBoth = list(set(iowaExpPatsPre) & set(iowaExpPatsPost))
#create helper functions for using the raw data
#these are the same as above but using a different wildcard
#this is repetative and there is likely a better way to do this
#def getIowaExpFullAnnotPre_both(wildcards):
#    return iowaExpFullAnnotPathDictPre[wildcards.iowaExpPats]
#def getIowaExpFullAnnotPost_both(wildcards):
#    return iowaExpFullAnnotPathDictPost[wildcards.iowaExpPats]

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
superimpIowaExpAnnotRugaeDeps = ["tools/getRegistration.py", "tools/trimeshToDfNoLabels.py", "tools/dfToPlyExport.py"]
manipulateAndFormatPack = ["tools/trimeshExtractFaceLabels.py", "tools/trimeshToDf_labels.py", "tools/dfToPlyExport.py"]
manipulateAndFormatPack2 = ["tools/trimeshExtractFaceLabels.py", "tools/trimeshToDf_labels.py", "tools/trimeshToDfNoLabels.py", "tools/dfToPlyExport.py"]
convertTeeth3dsObjToPlyDeps = ["tools/colorNumFrame.py", "tools/trimeshToDf_labels.py", "tools/objJsonToDataFrames.py", "tools/dfToPlyExport.py"]
getRotToMastDeps = ["tools/getRotToMaster.py", "tools/preprocess_point_cloud.py"]
remeshDeps = ["tools/trimeshExtractFaceLabels.py", "tools/colorNumFrame.py", "tools/trimeshToDf_labels.py", "tools/trimeshToDfNoLabels.py", "tools/dfToPlyExport.py"]
formatRawIteroPlyDeps = ["tools/trimeshToDfNoLabels.py", "tools/dfToPlyExport.py"]
calcCentSizeDeps = [
"tools/readAndFormat.py",
"tools/toothCentroids.py",
"tools/teethToCenterDist.py",
"tools/centroidSize.py",
"tools/toothVars.py",
"tools/plyRead.py"
]

###############################################################################
##################################BEGIN RULES##################################
###############################################################################

#rule specifying what is required to exist
rule all:
    input:
        #require the following things to exist
        #the wildcard {name} and what it stands for (given by the second expant arg) is passed
        #to any rule associated with this file
        #iowaRme:
        #upper scans converted from the original stls
        #preD
        expand(preDFullScanDir + "{preDPat}u_preD.ply", preDPat = patNamesPreD),
        #fin
        expand(finFullScanDir + "{finPat}u_fin.ply", finPat = patNamesFin),
        #iowaExpansion, segmentation model ready data
        #pre
        #expand(iowaExpSegReadyPreDir + "{iowaExpPrePat}Pre_segReady.ply", iowaExpPrePat = iowaExpPatsPre),
        #post
        #expand(iowaExpSegReadyPostDir + "{iowaExpPostPat}Post_segReady.ply", iowaExpPostPat = iowaExpPatsPost),
        #iowaExpansion annotated rugae superimposition transformations
        #expand(iowaExpRugaeTransDir + "{iowaExpPats}AnnotRugaeTrans.pkl", iowaExpPats = iowaExpPatsBoth),
        #iowaExpansion post scans with annotated rugae superimposition transformation applied
        #expand(iowaExpRugaeSuperimpPostScanDir + "{iowaExpPats}Post_annotRugaeSuperimp.ply", iowaExpPats = iowaExpPatsBoth),
        #iowaExpansion pre and post scan visualization htmls with no superimposition
        #expand(iowaExpNoSuperimpVisDir + "{iowaExpPats}NoSuperimpVis.html", iowaExpPats = iowaExpPatsBoth),
        #iowaExpansion pre and post scan visualization htmls with annotated rugae superimposition
        #expand(iowaExpAnnotRugaeSuperimpVisDir + "{iowaExpPats}AnnotRugaeSuperimpVis.html", iowaExpPats = iowaExpPatsBoth),
        #train and test split for teeth3dsIosseg_cSRot
        "K:/trainTestSets/remeshT3dsIos_cSRot/remeshT3dsIos_cSRot_trainTestSplit.complete",
        #master arches
        masterArchesDir + "masterArch1/mA1Full.ply"




#rule for just superimposition work
#rule superimp:
#    input:
#        #iowaExpansion annotated rugae superimposition transformations
#        expand(iowaExpRugaeTransDir + "{iowaExpPats}AnnotRugaeTrans.pkl", iowaExpPats = iowaExpPatsBoth),
#        #iowaExpansion post scans with annotated rugae superimposition transformation applied
#        expand(iowaExpRugaeSuperimpPostScanDir + "{iowaExpPats}Post_annotRugaeSuperimp.ply", iowaExpPats = iowaExpPatsBoth),
#        #iowaExpansion pre and post scan visualization htmls with no superimposition
#        expand(iowaExpNoSuperimpVisDir + "{iowaExpPats}NoSuperimpVis.html", iowaExpPats = iowaExpPatsBoth),
#        #iowaExpansion pre and post scan visualization htmls with annotated rugae superimposition
#        expand(iowaExpAnnotRugaeSuperimpVisDir + "{iowaExpPats}AnnotRugaeSuperimpVis.html", iowaExpPats = iowaExpPatsBoth)

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
        iowaExpTestCentSizeDir + "centSize_t3dsIosseg_cSOriMastEpoch300/centSizePre.csv",
        #post centroid size
        iowaExpTestCentSizeDir + "centSize_t3dsIosseg_cSOriMastEpoch300/centSizePost.csv"

#cannot directly run "snakemake convertPreDStlToPly -c1" because the input uses a wildcard via the helper
#function that snakemake will not be able to understand without the context of the rule all
#there are ways around this but this is fine for now
rule convertPreDStlToPly:
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


#iowaExpansion
#make pre full annotated scans ready for the segmentation model
#rule makeIowaExpFullAnnotPreSegReady:
#    input:
#        #using helper function
#        inFile = getIowaExpFullAnnotPre,
#        script = "tools/processes/makeSegmentationReady.py",
#        deps = makeSegReadyDeps
#    output:
#        outFile = iowaExpSegReadyPreDir + "{iowaExpPrePat}Pre_segReady.ply"
#    shell:
#        """
#        python {input.script} {input.inFile} {output.outFile}
#        """

#iowaExpansion
#make post full annotated scans ready for the segmentation model
#rule makeIowaExpFullAnnotPostSegReady:
#    input:
#        #using helper function
#        inFile = getIowaExpFullAnnotPost,
#        script = "tools/processes/makeSegmentationReady.py",
#        deps = makeSegReadyDeps
#    output:
#        outFile = iowaExpSegReadyPostDir + "{iowaExpPostPat}Post_segReady.ply"
#    shell:
#        """
#        python {input.script} {input.inFile} {output.outFile}
#        """


##########################################
#BEGIN NEW TEETH3DS

#teeth3ds
#convert obj/json combos into plys
rule convertTeeth3dsObjToPly:
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
    input:
        inPath = teeth3dsFullCSDir + "{teeth3dsName}_U_cS.ply",
        script = "tools/processes/getRotMatToMasterArch.py",
        deps = getRotToMastDeps
    output:
        outPath = teeth3dsCSMastRotMatDir + "{teeth3dsName}_U_cS_mastRotMat.pkl"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath}
        """

#teeth3ds
#apply rotation matrix to center and scaled teeth3ds data
rule orientToMastTeeth3ds:
    input:
        inPly = teeth3dsFullCSDir + "{teeth3dsName}_U_cS.ply",
        inMat = teeth3dsCSMastRotMatDir + "{teeth3dsName}_U_cS_mastRotMat.pkl",
        script = "tools/processes/orientToMasterArch.py",
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

#END NEW TEETH3DS
################################################

#iowaExpansion
#superimposition on annotated rugae region
#rule superimpIowaExpAnnotRugae:
#    input:
#        #using the helper function
#        prePath = getIowaExpFullAnnotPre_both,
#        postPath = getIowaExpFullAnnotPost_both,
#        script = "superimposition/rugaeAnnotRegistration.py",
#        deps = superimpIowaExpAnnotRugaeDeps
#    output:
#        transPath = iowaExpRugaeTransDir + "{iowaExpPats}AnnotRugaeTrans.pkl",
#        outPlyPath = iowaExpRugaeSuperimpPostScanDir + "{iowaExpPats}Post_annotRugaeSuperimp.ply"
#    shell:
#        """
#        python {input.script} {input.prePath} {input.postPath} {output.transPath} {output.outPlyPath}
#        """


#iowaExpansion
#html visuals for pre and post scans with no superimposition
#rule makePrePostScanVisNoSuperimp:
#    input:
#        prePath = getIowaExpFullAnnotPre_both,
#        postPath = getIowaExpFullAnnotPost_both,
#        script = "superimposition/createSuperimpHtmlVisuals.py"
#    params:
#        color_ = "red",
#    output:
#        visHtml = iowaExpNoSuperimpVisDir + "{iowaExpPats}NoSuperimpVis.html"
#    shell:
#        """
#        python {input.script} {input.prePath} {input.postPath} {params.color_} {output.visHtml}
#        """

#iowaExpansion
#html visuals for pre and post scans with annotated rugae superimposition
#rule makePrePostScanVisAnnotRugaeSuperimp:
#    input:
#        prePath = getIowaExpFullAnnotPre_both,
#        postPath = iowaExpRugaeSuperimpPostScanDir + "{iowaExpPats}Post_annotRugaeSuperimp.ply",
#        script = "superimposition/createSuperimpHtmlVisuals.py"
#    params:
#        color_ = "green",
#    output:
#        visHtml = iowaExpAnnotRugaeSuperimpVisDir + "{iowaExpPats}AnnotRugaeSuperimpVis.html"
#    shell:
#        """
#        python {input.script} {input.prePath} {input.postPath} {params.color_} {output.visHtml}
#        """


#################################################
#BEGIN NEW IOWAEXPTEST

#iowaExpTest
#format raw itero scans pre
rule formatRawIowaExpTestPre:
    input:
        inPath = getIowaExpTestOrigPre,
        script = "tools/processes/formatRawIteroPly.py",
        deps = formatRawIteroPlyDeps
    output:
        outPath = iowaExpTestOrigFormPreDir + "{iowaExpTestPrePat}Pre_form.ply"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath}
        """

#iowaExpTest
#format raw itero scans post
rule formatRawIowaExpTestPost:
    input:
        inPath = getIowaExpTestOrigPost,
        script = "tools/processes/formatRawIteroPly.py",
        deps = formatRawIteroPlyDeps
    output:
        outPath = iowaExpTestOrigFormPostDir + "{iowaExpTestPostPat}Post_form.ply"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath}
        """

#iowaExpTest
#cetner and scale pre
rule centerScaleIowaExpTestPre:
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
    input:
        inPath = iowaExpTestOrigFormCSPreDir + "{iowaExpTestPrePat}Pre_formCS.ply",
        script = "tools/processes/getRotMatToMasterArch.py",
        deps = getRotToMastDeps
    output:
        outPath = iowaExpTestOrigFormCSPreMastRotMatDir + "{iowaExpTestPrePat}Pre_formCS_mastRotMat.pkl"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath}
        """

#iowaExpTest
#get rotation matrix to master arch, post
rule getRotToMastIowaExpTestPost:
    input:
        inPath = iowaExpTestOrigFormCSPostDir + "{iowaExpTestPostPat}Post_formCS.ply",
        script = "tools/processes/getRotMatToMasterArch.py",
        deps = getRotToMastDeps
    output:
        outPath = iowaExpTestOrigFormCSPostMastRotMatDir + "{iowaExpTestPostPat}Post_formCS_mastRotMat.pkl"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath}
        """

#iowaExpTest
#apply rotation matrix to center and scaled iowaExpTest pre data
rule orientToMastIowaExpTestPre:
    input:
        inPly = iowaExpTestOrigFormCSPreDir + "{iowaExpTestPrePat}Pre_formCS.ply",
        inMat = iowaExpTestOrigFormCSPreMastRotMatDir + "{iowaExpTestPrePat}Pre_formCS_mastRotMat.pkl",
        script = "tools/processes/orientToMasterArch.py",
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
    input:
        inPly = iowaExpTestOrigFormCSPostDir + "{iowaExpTestPostPat}Post_formCS.ply",
        inMat = iowaExpTestOrigFormCSPostMastRotMatDir + "{iowaExpTestPostPat}Post_formCS_mastRotMat.pkl",
        script = "tools/processes/orientToMasterArch.py",
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

#iowaExpTest
#get centroid size for segmented pre scans
rule getCentSizeIowaExpTestPre:
    input:
        dir_ = directory(iowaExpTestSegPreDir),
        script = "tools/processes/calculateCentroidSize.py",
        deps = calcCentSizeDeps
    output:
        outPath = iowaExpTestCentSizeDir + "centSize_t3dsIosseg_cSOriMastEpoch300/centSizePre.csv"
    shell:
        """
        python {input.script} {input.dir_} {output.outPath}
        """

#iowaExpTest
#get centroid size for segmented post scans
rule getCentSizeIowaExpTestPost:
    input:
        dir_ = directory(iowaExpTestSegPostDir),
        script = "tools/processes/calculateCentroidSize.py",
        deps = calcCentSizeDeps
    output:
        outPath = iowaExpTestCentSizeDir + "centSize_t3dsIosseg_cSOriMastEpoch300/centSizePost.csv"
    shell:
        """
        python {input.script} {input.dir_} {output.outPath}
        """

#END NEW IOWAEXPTEST
#################################################



#############################
#BEGIN NEW IOSSEG

#iosseg
#center and scale
rule centerAndScaleIosseg:
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
    input:
        inPath = iossegCleanUCSDir + "{iossegCleanUPat}_U_cS.ply",
        script = "tools/processes/getRotMatToMasterArch.py",
        deps = getRotToMastDeps
    output:
        outPath = iossegCleanUCSMastRotMatDir + "{iossegCleanUPat}_U_cS_mastRotMat.pkl"
    shell:
        """
        python {input.script} {input.inPath} {output.outPath}
        """

#iosseg
#apply rotation matrix to center and scaled teeth3ds data
rule orientToMastIosseg:
    input:
        inPly = iossegCleanUCSDir + "{iossegCleanUPat}_U_cS.ply",
        inMat = iossegCleanUCSMastRotMatDir + "{iossegCleanUPat}_U_cS_mastRotMat.pkl",
        script = "tools/processes/orientToMasterArch.py",
        deps = manipulateAndFormatPack2
    params:
        labs = True
    output:
        outPath = iossegCleanUCSOriMastDir + "{iossegCleanUPat}_U_cSOriMast.ply"
    shell:
        """
        python {input.script} {input.inPly} {input.inMat} {output.outPath} {params.labs}
        """

#END NEW IOSSEG
#############################


#tain test set for teeth3dsIosseg_cSOriMast
rule trainTestSplit_teeth3dsIosseg_cSOriMast:
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

#THIS IS TEMPORARY
#iowaExpansion
#pre full annotated scans CENTER SCALE AND NO ORIENTATION, SEGREADY2
#rule makeIowaExpFullAnnotPreSegReady2:
#    input:
#        #using helper function
#        inFile = getIowaExpFullAnnotPre,
#        script = "tools/processes/makeSegmentationReady2.py",
#        deps = makeSegReadyDeps
#    output:
#        outFile = iowaExpSegReadyPreDir2 + "{iowaExpPrePat}Pre_segReady2.ply"
#    shell:
#        """
#        python {input.script} {input.inFile} {output.outFile}
#        """

#iowaExpansion
#post full annotated scans CENTER SCALE AND NO ORIENTATION, SEGREADY2
#rule makeIowaExpFullAnnotPostSegReady2:
#    input:
#        #using helper function
#        inFile = getIowaExpFullAnnotPost,
#        script = "tools/processes/makeSegmentationReady2.py",
#        deps = makeSegReadyDeps
#    output:
#        outFile = iowaExpSegReadyPostDir2 + "{iowaExpPostPat}Post_segReady2.ply"
#    shell:
#        """
#        python {input.script} {input.inFile} {output.outFile}
#        """

#create master arches
rule createMasterArches:
    input:
        script = "tools/processes/createMasterArches.py",
        deps = manipulateAndFormatPack
    output:
        m1OutPath = masterArchesDir + "masterArch1/mA1Full.ply"
    shell:
        """
        python {input.script} {output.m1OutPath}
        """