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
preDDec016ScanDir = "K:/iowaRme/preDelivAndFinalScans/preDelivScanU/dec016Scans/"
preDDec016OriScanDir = "K:/iowaRme/preDelivAndFinalScans/preDelivScanU/dec016OriScans/"
PreDDec016OriSegDir = "K:/iowaRme/preDelivAndFinalScans/preDelivScanU/dec016OriSeg/"
preDSegReadyScansDir = "K:/iowaRme/preDelivAndFinalScans/preDelivScanU/segReadyScans/"
#fin files and directories
finFullScanDir = "K:/iowaRme/preDelivAndFinalScans/finalScanU/fullScans/"
finDec016ScanDir = "K:/iowaRme/preDelivAndFinalScans/finalScanU/dec016Scans/"
finDec016OriScanDir = "K:/iowaRme/preDelivAndFinalScans/finalScanU/dec016OriScans/"
finDec016OriSegDir = "K:/iowaRme/preDelivAndFinalScans/finalScanU/dec016OriSeg/"
finSegReadyScansDir = "K:/iowaRme/preDelivAndFinalScans/finalScanU/segReadyScans/"
#directories for transformations for registering final scan to preD scan
preDFinDec016TransDir = "K:/iowaRme/registTrans/preDFin_dec016/"
#centroid and distance directories
preDFunDec016DistDir = "K:/iowaRme/movement/preDFin_dec016/"

#iowaExpansion
#full, rugae annotated scans
iowaExpFullAnnotPreDir = "K:/iowaExpansion/fullRugaeAnnotScans/pre/"
iowaExpFullAnnotPostDir = "K:/iowaExpansion/fullRugaeAnnotScans/post/"
#segmentation model ready scans
iowaExpSegReadyPreDir = "K:/iowaExpansion/segReadyScans/pre/"
iowaExpSegReadyPostDir = "K:/iowaExpansion/segReadyScans/post/"
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
teeth3dsRemeshDir = "K:/teeth3DS/scanData/upperPlyRemesh/"
teeth3dsRandRotDir = "K:/teeth3DS/randomRotations/"
teeth3dsRemeshCSRotDir = "K:/teeth3DS/scanData/upperPlyRemeshCSRot/"
#original files
teeth3dsOrigFilesDir = "K:/teeth3DS/scanData/upper/"

#iosseg
#plys
iossegAllCleanUDir = "K:/IOSSegData/clean/allCleanU/"
iossegCleanCSRotUpperDir = "K:/IOSSegData/cleanCSRot/upper/"
#random rotations
iossegRandRotDir = "K:/IOSSegData/randomRotations/"


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
#iowaRme
#i do not currently have a set up to make snake make run the actual segmentation process
#because that involves using the HPC and i need to make an updated container 
#for this that contains snake make. In the mean time, I will take care of the 
#segementation manually 
#since that is the case, we need to use a bit of a work around since we cannot require
#the segmented plys to exist bc there is not rule that is able to make them
#i am going to use a similar process to what is used in the first step with
#the original stls

#we will no longer need this when i have snakemake working on hpc

#for preD scans
#dictionary for segmented file paths
preDDec016OriSegPath = [PreDDec016OriSegDir + i + "u_preD_dec016Ori_seg.ply" for i in patNamesPreD]
preDDec016OriSegDict = dict(zip(patNamesPreD, preDDec016OriSegPath))
#creating helper function to use with wildcards in rules section
def getPreDDec016OriSeg(wildcards):
    return preDDec016OriSegDict[wildcards.bothPat]


#for fin scans
#dictionary for segmented file paths
finDec016OriSegPath = [finDec016OriSegDir + i + "u_fin_dec016Ori_seg.ply" for i in patNamesFin]
finDec016OriSegDict = dict(zip(patNamesFin, finDec016OriSegPath))
#creating helper function to use with wildcards in rules section
def getFinDec016OriSeg(wildcards):
    return finDec016OriSegDict[wildcards.bothPat]

###############################################################################
#iowaExpansion
#get patient names and create directory dictionary
iowaExpPatsPre, iowaExpFullAnnotPathDictPre = patNamesAndPathDict(iowaExpFullAnnotPreDir)
iowaExpPatsPost, iowaExpFullAnnotPathDictPost = patNamesAndPathDict(iowaExpFullAnnotPostDir)
#create helper functions for using the raw data
def getIowaExpFullAnnotPre(wildcards):
    return iowaExpFullAnnotPathDictPre[wildcards.iowaExpPrePat]
def getIowaExpFullAnnotPost(wildcards):
    return iowaExpFullAnnotPathDictPost[wildcards.iowaExpPostPat]

###############################################################################
#iowaExpansion
#patient names for just the patients with both a pre and a post
iowaExpPatsBoth = list(set(iowaExpPatsPre) & set(iowaExpPatsPost))
#create helper functions for using the raw data
#these are the same as above but using a different wildcard
#this is repetative and there is likely a better way to do this
def getIowaExpFullAnnotPre_both(wildcards):
    return iowaExpFullAnnotPathDictPre[wildcards.iowaExpPats]
def getIowaExpFullAnnotPost_both(wildcards):
    return iowaExpFullAnnotPathDictPost[wildcards.iowaExpPats]

###############################################################################
#NOT CURRENTLY IN USE
#teeth3ds
#get patient names and create directory dictionary
teeth3dsFullPlyNames, teeth3dsFullPlyPathDict = patNamesAndPathDict(dir_ = teeth3dsFullDir,  pattern = r'^(.+)_', captureGroup = 1)
#create helper functions for using the raw data
def getTeeth3dsFullPly(wildcards):
    return teeth3dsFullPlyPathDict[wildcards.teeth3dsPlyName]

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

#iowaExpansion
#get patient names and create directory dictionary
allIossegCleanUPats, iossegCleanUPathDict = patNamesAndPathDict(iossegAllCleanUDir, pattern = r'^[0-9]{3}')
#create helper functions for using the raw data
def getIossegCleanU(wildcards):
    return iossegCleanUPathDict[wildcards.iossegCleanUPat]

###############################################################################
#dependency lists
stlConvertNoLabsDepends = ["tools/stlToPlyFuns.py"]
decimNoLabsDepends = ["tools/decimationFuns.py", "tools/formatAndExportFuns.py"]
orientTeeth3DSDepends = ["tools/registrationFuns.py"]
getRegistTransDepends = ["tools/plyToRegistTransformation.py", "tools/registrationFuns.py"]
centroidAndMeasureDepends = ["tools/trimeshToDf_labels.py", "tools/plyFunctions.py", "tools/centroidDistance.py", "tools/toothCentroids.py"]
makeSegReadyDeps = ["tools/getRegistration.py", "tools/trimeshToDfNoLabels.py", "tools/dfToPlyExport.py"]
remeshTeeth3dsFullPlysDeps = ["tools/trimeshToDf_labels.py", "tools/dfToPlyExport.py", "tools/colorNumFrame.py"]
superimpIowaExpAnnotRugaeDeps = ["tools/getRegistration.py", "tools/trimeshToDfNoLabels.py", "tools/dfToPlyExport.py"]
centerScaleRotatePlyDeps = ["tools/trimeshExtractFaceLabels.py", "tools/trimeshToDf_labels.py", "tools/dfToPlyExport.py"]

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
        #iowaRme preD upper scan plys decimated to 16000 faces
        expand(preDDec016ScanDir + "{preDPat}u_preD_dec016.ply", preDPat = patNamesPreD),
        #iowaRme fin upper scan plys decimated to 16000 faces
        expand(finDec016ScanDir + "{finPat}u_fin_dec016.ply", finPat = patNamesFin),
        #iowaRme preD upper decimated scans oriented to teeth3ds training data
        expand(preDDec016OriScanDir + "{preDPat}u_preD_dec016Ori.ply", preDPat = patNamesPreD),
        #iowaRme fin upper decimated scans oriented to teeth3ds training data
        expand(finDec016OriScanDir + "{finPat}u_fin_dec016Ori.ply", finPat = patNamesFin),
        #iowaRme preD segmentation ready
        expand(preDSegReadyScansDir + "{preDPat}u_preD_segReady.ply", preDPat = patNamesPreD),
        #iowaRme fin segmentationReady
        expand(finSegReadyScansDir + "{finPat}u_fin_segReady.ply", finPat = patNamesFin),
        #iowaRme transformations for registering fin scan to preD scan
        expand(preDFinDec016TransDir + "{bothPat}u_registTrans_dec016.pkl", bothPat = patNamesBoth),
        #iowaRme centroid and distance data
        expand(preDFunDec016DistDir + "{bothPat}u_dist_dec016.csv", bothPat = patNamesBoth),
        #visualizations
        # COMMENTING OUT FOR NOW UNTIL I FIGURE OUT THE RSCRIPT ISSUE
        #iowaRme centroid movement line plot
        #"movement/visualization/centroidMovement/centMovePatLines.html",
        #iowaRme cetroid movement bee swarm
        #"movement/visualization/centroidMovement/centMoveBeeSwarm.png"
        #iowaExpansion, segmentation model ready data
        #pre
        expand(iowaExpSegReadyPreDir + "{iowaExpPrePat}Pre_segReady.ply", iowaExpPrePat = iowaExpPatsPre),
        #post
        expand(iowaExpSegReadyPostDir + "{iowaExpPostPat}Post_segReady.ply", iowaExpPostPat = iowaExpPatsPost),
        #teeth3ds, full plys remeshed
        expand(teeth3dsRemeshDir + "{teeth3dsName}_U_remesh.ply", teeth3dsName = patNames3ds),
        #teeth3ds, creating random rotations which is monitored via a sentinel file
        teeth3dsRandRotDir + "allRotationsCreated.complete",
        #teeth3ds, remeshed files that are centered scaled and randomly rotated
        expand(teeth3dsRemeshCSRotDir + "{teeth3dsName}_U_remeshCSRot.ply", teeth3dsName = patNames3ds),
        #iowaExpansion annotated rugae superimposition transformations
        expand(iowaExpRugaeTransDir + "{iowaExpPats}AnnotRugaeTrans.pkl", iowaExpPats = iowaExpPatsBoth),
        #iowaExpansion post scans with annotated rugae superimposition transformation applied
        expand(iowaExpRugaeSuperimpPostScanDir + "{iowaExpPats}Post_annotRugaeSuperimp.ply", iowaExpPats = iowaExpPatsBoth),
        #iowaExpansion pre and post scan visualization htmls with no superimposition
        expand(iowaExpNoSuperimpVisDir + "{iowaExpPats}NoSuperimpVis.html", iowaExpPats = iowaExpPatsBoth),
        #iowaExpansion pre and post scan visualization htmls with annotated rugae superimposition
        expand(iowaExpAnnotRugaeSuperimpVisDir + "{iowaExpPats}AnnotRugaeSuperimpVis.html", iowaExpPats = iowaExpPatsBoth),
        #iosseg, creating random rotations which is monitored via a sentinel file
        iossegRandRotDir + "allRotationsCreated.complete",
        #iosseg, clean files that are cetnered scaled and randomly rotated
        expand(iossegCleanCSRotUpperDir + "{iossegCleanUPat}_U_cSRot.ply", iossegCleanUPat = allIossegCleanUPats),
        #train and test split for teeth3dsIosseg_cSRot
        "K:/trainTestSets/teeth3dsIosseg_cSRot/trainTestSplit.complete"




#rule for just superimposition work
rule superimp:
    input:
        #iowaExpansion annotated rugae superimposition transformations
        expand(iowaExpRugaeTransDir + "{iowaExpPats}AnnotRugaeTrans.pkl", iowaExpPats = iowaExpPatsBoth),
        #iowaExpansion post scans with annotated rugae superimposition transformation applied
        expand(iowaExpRugaeSuperimpPostScanDir + "{iowaExpPats}Post_annotRugaeSuperimp.ply", iowaExpPats = iowaExpPatsBoth),
        #iowaExpansion pre and post scan visualization htmls with no superimposition
        expand(iowaExpNoSuperimpVisDir + "{iowaExpPats}NoSuperimpVis.html", iowaExpPats = iowaExpPatsBoth),
        #iowaExpansion pre and post scan visualization htmls with annotated rugae superimposition
        expand(iowaExpAnnotRugaeSuperimpVisDir + "{iowaExpPats}AnnotRugaeSuperimpVis.html", iowaExpPats = iowaExpPatsBoth)

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

#iowaRme
#decimate preD scans
rule producePreDDec016Scans:
    input:
        inFile = preDFullScanDir + "{preDPat}u_preD.ply",
        script = "tools/processes/fullScanDecim_noLabs.py",
        deps = decimNoLabsDepends
    output:
        outFile = preDDec016ScanDir + "{preDPat}u_preD_dec016.ply"
    shell:
        """
        python {input.script} "{input.inFile}" "{output.outFile}"
        """

#iowaRme
#decimate fin scans
rule produceFinDec016Scans:
    input:
        inFile = finFullScanDir + "{finPat}u_fin.ply",
        script = "tools/processes/fullScanDecim_noLabs.py",
        deps = decimNoLabsDepends
    output:
        outFile = finDec016ScanDir + "{finPat}u_fin_dec016.ply"
    shell:
        """
        python {input.script} "{input.inFile}" "{output.outFile}"
        """

#iowaRme
#orient preD iowaRme scans in direction of teeth3ds training data
rule orientPreDDec016Scans:
    input:
        inFile = preDDec016ScanDir + "{preDPat}u_preD_dec016.ply",
        script = "tools/processes/orientToTeeth3DS.py",
        deps = orientTeeth3DSDepends
    output:
        outFile = preDDec016OriScanDir + "{preDPat}u_preD_dec016Ori.ply"
    shell:
        """
        python {input.script} "{input.inFile}" "{output.outFile}"
        """

#iowaRme
#orient fin iowaRme scans in direction of teeth3ds training data
rule orientFinDec016Scans:
    input:
        inFile = finDec016ScanDir + "{finPat}u_fin_dec016.ply",
        script = "tools/processes/orientToTeeth3DS.py",
        deps = orientTeeth3DSDepends
    output:
        outFile = finDec016OriScanDir + "{finPat}u_fin_dec016Ori.ply"
    shell:
        """
        python {input.script} "{input.inFile}" "{output.outFile}"
        """

#iowaRme
#get transformations that register fin scan to preD scan
rule getPreDFinRegistTrans:
    input:
        preDPath = getPreDDec016OriSeg,
        finPath = getFinDec016OriSeg,
        script = "tools/processes/getRegistTrans.py",
        deps = getRegistTransDepends
    output:
        outFile = preDFinDec016TransDir + "{bothPat}u_registTrans_dec016.pkl"
    shell:
        """
        python {input.script} {input.preDPath} {input.finPath} {output.outFile}
        """

#iowaRme
#get distance and centroid data for preD and fin scans
rule getPreDFinDist:
    input:
        preDPath = getPreDDec016OriSeg,
        finPath = getFinDec016OriSeg,
        transPath = preDFinDec016TransDir + "{bothPat}u_registTrans_dec016.pkl",
        script = "tools/processes/centroidAndMeasure.py",
        deps = centroidAndMeasureDepends
    output:
        outFile = preDFunDec016DistDir + "{bothPat}u_dist_dec016.csv"
    shell:
        """
        python {input.script} {input.preDPath} {input.finPath} {input.transPath} {output.outFile}
        """

#iowaRme
#make preD iowaRme scans ready for segmentation model via remeshing and orientation
rule makeIowaRmePreDSegmentationReady:
    input:
        inFile = preDFullScanDir + "{preDPat}u_preD.ply",
        script = "tools/processes/makeSegmentationReady.py",
        deps = makeSegReadyDeps
    output:
        outFile = preDSegReadyScansDir + "{preDPat}u_preD_segReady.ply"
    shell:
        """
        python {input.script} {input.inFile} {output.outFile}
        """

#iowaRme
#make fin iowaRme scans ready for segmentation model via remeshing and orientation
rule makeIowaRmeFinSegmentationReady:
    input:
        inFile = finFullScanDir + "{finPat}u_fin.ply",
        script = "tools/processes/makeSegmentationReady.py",
        deps = makeSegReadyDeps
    output:
        outFile = finSegReadyScansDir + "{finPat}u_fin_segReady.ply"
    shell:
        """
        python {input.script} {input.inFile} {output.outFile}
        """


#iowaRme
#basic visualizations for the centroid movement data
#this relies on an entire directory of files, using expand() functionality
# COMMENTING OUT FOR NOW UNTIL I FIGURE OUT THE RSCRIPT ISSUE
#rule makeCentroidVis:
#    input:
#        inFiles = expand(preDFunDec016DistDir + "{bothPat}u_dist_dec016.csv", bothPat = patNamesBoth),
#        script = "movement/visualization/centroidMovement/centroidMoveVis.R"
#    output:
#        patLines = "movement/visualization/centroidMovement/centMovePatLines.html",
#        beePlot = "movement/visualization/centroidMovement/centMoveBeeSwarm.png"
#    shell:
#        """
#        Rscript {input.script} {output.patLines} {output.beePlot}
#        """


#iowaExpansion
#make pre full annotated scans ready for the segmentation model
rule makeIowaExpFullAnnotPreSegReady:
    input:
        #using helper function
        inFile = getIowaExpFullAnnotPre,
        script = "tools/processes/makeSegmentationReady.py",
        deps = makeSegReadyDeps
    output:
        outFile = iowaExpSegReadyPreDir + "{iowaExpPrePat}Pre_segReady.ply"
    shell:
        """
        python {input.script} {input.inFile} {output.outFile}
        """

#iowaExpansion
#make post full annotated scans ready for the segmentation model
rule makeIowaExpFullAnnotPostSegReady:
    input:
        #using helper function
        inFile = getIowaExpFullAnnotPost,
        script = "tools/processes/makeSegmentationReady.py",
        deps = makeSegReadyDeps
    output:
        outFile = iowaExpSegReadyPostDir + "{iowaExpPostPat}Post_segReady.ply"
    shell:
        """
        python {input.script} {input.inFile} {output.outFile}
        """

#teeth3ds
#remesh full plys
rule remeshTeeth3dsFullPlys:
    input:
        #using the helper function
        objFile = getOrig3dsObj,
        jsonFile = getOrig3dsJson,
        script = "tools/processes/remeshFullPlyTeeth3Ds.py",
        deps = remeshTeeth3dsFullPlysDeps
    output:
        outFile = teeth3dsRemeshDir + "{teeth3dsName}_U_remesh.ply"
    shell:
        """
        python {input.script} {input.objFile} {input.jsonFile} {output.outFile}
        """


#iowaExpansion
#superimposition on annotated rugae region
rule superimpIowaExpAnnotRugae:
    input:
        #using the helper function
        prePath = getIowaExpFullAnnotPre_both,
        postPath = getIowaExpFullAnnotPost_both,
        script = "superimposition/rugaeAnnotRegistration.py",
        deps = superimpIowaExpAnnotRugaeDeps
    output:
        transPath = iowaExpRugaeTransDir + "{iowaExpPats}AnnotRugaeTrans.pkl",
        outPlyPath = iowaExpRugaeSuperimpPostScanDir + "{iowaExpPats}Post_annotRugaeSuperimp.ply"
    shell:
        """
        python {input.script} {input.prePath} {input.postPath} {output.transPath} {output.outPlyPath}
        """


#iowaExpansion
#html visuals for pre and post scans with no superimposition
rule makePrePostScanVisNoSuperimp:
    input:
        prePath = getIowaExpFullAnnotPre_both,
        postPath = getIowaExpFullAnnotPost_both,
        script = "superimposition/createSuperimpHtmlVisuals.py"
    params:
        color_ = "red",
    output:
        visHtml = iowaExpNoSuperimpVisDir + "{iowaExpPats}NoSuperimpVis.html"
    shell:
        """
        python {input.script} {input.prePath} {input.postPath} {params.color_} {output.visHtml}
        """

#iowaExpansion
#html visuals for pre and post scans with annotated rugae superimposition
rule makePrePostScanVisAnnotRugaeSuperimp:
    input:
        prePath = getIowaExpFullAnnotPre_both,
        postPath = iowaExpRugaeSuperimpPostScanDir + "{iowaExpPats}Post_annotRugaeSuperimp.ply",
        script = "superimposition/createSuperimpHtmlVisuals.py"
    params:
        color_ = "green",
    output:
        visHtml = iowaExpAnnotRugaeSuperimpVisDir + "{iowaExpPats}AnnotRugaeSuperimpVis.html"
    shell:
        """
        python {input.script} {input.prePath} {input.postPath} {params.color_} {output.visHtml}
        """

#teeth3ds, create random rotations
rule createRandomRotTeeth3ds:
    input:
        dirPath = teeth3dsFullDir,
        script = "tools/processes/getRandRotationsForDir.py"
    output:
        #monitoring done be sentinel file
        touch(teeth3dsRandRotDir + "allRotationsCreated.complete")
    shell:
        """
        python {input.script} {input.dirPath} "K:/teeth3DS/randomRotations/"
        """

#teeth3ds, center scale and randomly rotate the remeshed files
rule cSRotTeeth3dsRemeshed:
    input:
        inPath = teeth3dsRemeshDir + "{teeth3dsName}_U_remesh.ply",
        #rotation matrices monitored by sentinel file
        rotSentinel = teeth3dsRandRotDir + "allRotationsCreated.complete",
        script = "tools/processes/centerScaleRotatePly.py",
        deps = centerScaleRotatePlyDeps
    params:
        fileSuffix = "_remesh.ply"
    output:
        outPath = teeth3dsRemeshCSRotDir + "{teeth3dsName}_U_remeshCSRot.ply"
    shell:
        """
        python {input.script} {input.inPath} "K:/teeth3DS/randomRotations/" {params.fileSuffix} {output.outPath}
        """

#iosseg, create random rotations
rule createRandomRotIosseg:
    input:
        dirPath = iossegAllCleanUDir,
        script = "tools/processes/getRandRotationsForDir.py"
    output:
        #monitoring done be sentinel file
        touch(iossegRandRotDir + "allRotationsCreated.complete")
    shell:
        """
        python {input.script} {input.dirPath} "K:/IOSSegData/randomRotations/"
        """

#iosseg, center scale and randomly rotate the clean files
rule cSRotIossegCleanU:
    input:
        inPath = getIossegCleanU,
        #rotation matrices monitored by sentinel file
        rotSentinel = iossegRandRotDir + "allRotationsCreated.complete",
        script = "tools/processes/centerScaleRotatePly.py",
        deps = centerScaleRotatePlyDeps
    params:
        fileSuffix = ".ply"
    output:
        outPath = iossegCleanCSRotUpperDir + "{iossegCleanUPat}_U_cSRot.ply"
    shell:
        """
        python {input.script} {input.inPath} "K:/IOSSegData/randomRotations/" {params.fileSuffix} {output.outPath}
        """

#tain test set for teeth3dsIosseg_cSRot
rule trainTestSplit_Teeth3dsIosseg_cSRot:
    input:
        #require all remeshCSRot teeth3ds files, but they are not input into the script
        t3ds_remeshCSRot = expand(teeth3dsRemeshCSRotDir + "{teeth3dsName}_U_remeshCSRot.ply", teeth3dsName = patNames3ds),
        #require all cSRot teeth3ds files, but they are not input into the script
        ios_cSRot = expand(iossegCleanCSRotUpperDir + "{iossegCleanUPat}_U_cSRot.ply", iossegCleanUPat = allIossegCleanUPats),
        script = "tools/processes/trainTestSets/split_teeth3dsIosseg_cSRot.py"
    output:
        #monitoring done by sentinel file
        touch("K:/trainTestSets/teeth3dsIosseg_cSRot/trainTestSplit.complete")
    shell:
        """
        python {input.script}
        """