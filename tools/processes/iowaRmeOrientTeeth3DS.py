import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import registrationFuns as regi
import random
random.seed(826)

#pull variables from snakemake
inFile = sys.argv[1]
outFile = sys.argv[2]

#DOING FULL REGISTRATION IN ORDER TO ORIENT THE RME DATA IN THE SAME DIRECTION
#AS THE TEETH3DS TRAINING DATA IS PROBABLY OVERKILL BUT IT IS THE TOOLS I HAVE
#BUILT ALREADY

#details on registration
#source: what is being transformed
#target: the base for the registration

#arbitrary target scan from teeth3ds
teeth3dsTarget = "K:/trainTestSets/teeth3dsDecim016/train/00OMSZGW_UDecim016.ply"

#perform registration
regi.fullRegistFlow(targetFile=teeth3dsTarget, sourceFile=inFile, registerFile=outFile)