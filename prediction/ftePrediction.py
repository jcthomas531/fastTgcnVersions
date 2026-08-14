import sys
from pathlib import Path
projectRoot = Path(__file__).resolve().parent.parent
sys.path.append(str(projectRoot / "prediction/"))


import fastTgcnEasyPredictFun as ftep


#predNote = "iowaExpTest RugAnnotForm_cSOriMastRemesh pre scans, using model t3dsIosseg_cSOriMastEpoch300"
#print(predNote)

#pull variables from snakemake
inDir_ = sys.argv[1]
outDir_ = sys.argv[2]
preMatDir = sys.argv[3]

ftep.fastTgcnEasyPredict(inDir = inDir_,
                         outDir = outDir_,
                         predMatOutDir = preMatDir,
                         modelPath = str(projectRoot / "fastTgcnEasy/trainedModels/2026_07_09 t3dsIosseg_cSOriMastEpoch300.pth")
                         )



#print(predNote)