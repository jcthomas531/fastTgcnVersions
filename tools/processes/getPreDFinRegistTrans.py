import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import plyToRegistTransfromation as prt
import random
import pickle
random.seed(826)

#pull variables from snakemake
preDPath = sys.argv[1] #the preD scan, the target
finPath = sys.argv[2] #the fin scan, the source
outPath = sys.argv[3] #where to output the transformation

#this registration will register on the entire arch
regResult = prt.plyToRegistTransformation(targetFile = preDPath, sourceFile = finPath)

#cannot export the entire object easily, just exporting transformation now but
#can return here later to export more pieces of the object if they become necessary
filePath = open(outPath, "wb")
pickle.dump(obj = regResult.transformation,
            file = filePath)
filePath.close()

#can be read in like: 
# with open("K:/iowaRme/registTrans/preDFin_dec016/test2.pkl", "rb") as f:
#     obj = pickle.load(f)
