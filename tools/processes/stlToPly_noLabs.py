import sys
sys.path.append("tools")
import stlToPlyFuns as stlPly


#pull variables from snakemake
inFile = sys.argv[1]
outFile = sys.argv[2]


stlPly.convertRmeStlToPly(inFile = inFile, outFile = outFile)