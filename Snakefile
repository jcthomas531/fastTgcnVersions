import os
import re

#tutorial: https://www.youtube.com/watch?v=r9PWnEmz_tc&t=1247s

#some rules
#cannot have empty lines in a rule

#iowaRme
#original stl directory and patient list
origStlDir = "K:/iowaRme/preDelivAndFinalScans/originalStl/"
allPats = os.listdir(origStlDir)

###############################################################################
#iowRme
#some patients may not have all of the files we need
#we are soley concerned about upper files
#the logic below creates lists (patNamesPreD and patNamesFin) of the patients who
#respectively have original stls for upper scans. these should be used throughout
#the rest of the logic as they represent the base truth of the scans that exist
##########

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
###############################################################################



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

#iowaRme:
#preD files and directories
preDFullScanDir = "K:/iowaRme/preDelivAndFinalScans/preDelivScanU/fullScans/"
preDDec016ScanDir = "K:/iowaRme/preDelivAndFinalScans/preDelivScanU/dec016Scans/"
preDDec016OriScanDir = "K:/iowaRme/preDelivAndFinalScans/preDelivScanU/dec016OriScans/"
#fin files and directories
finFullScanDir = "K:/iowaRme/preDelivAndFinalScans/finalScanU/fullScans/"
finDec016ScanDir = "K:/iowaRme/preDelivAndFinalScans/finalScanU/dec016Scans/"
finDec016OriScanDir = "K:/iowaRme/preDelivAndFinalScans/finalScanU/dec016OriScans/"

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
        expand(finDec016ScanDir + "{finPat}u_fin_dec016.ply", finPat = patNamesFin)
        



#cannot directly run "snakemake convertPreDStlToPly -c1" because the input uses a wildcard via the helper
#function that snakemake will not be able to understand without the context of the rule all
#there are ways around this but this is fine for now
rule convertPreDStlToPly:
    input: 
        #using preD stl helper function which makes use of wildcards
        inFile = getOrigStlPreD,
        script = "tools/processes/stlToPly_noLabs.py"
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
        script = "tools/processes/stlToPly_noLabs.py"
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
		script = "tools/processes/fullScanDecim_noLabs.py"
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
        script = "tools/processes/fullScanDecim_noLabs.py"
    output:
        outFile = finDec016ScanDir + "{finPat}u_fin_dec016.ply"
    shell:
        """
        python {input.script} "{input.inFile}" "{output.outFile}"
        """


