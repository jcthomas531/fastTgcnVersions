#!/bin/bash


#designed to be used like
# ./decimator.sh /Shared/gb_lss/Thomas/iowaRme/fullScans/pat058/pat058l_01.ply /Shared/gb_lss/Thomas/iowaRme/dec016Scans/pat058/pat058l_01.ply
#dont forget to make it into a unix file and make it executable 
#CURRENTLY NOT WORKING ON HPC BC THEY DONT HAVE PYVISTA
#NEED TO SEE IF I CAN RUN IT SOMEWHERE ELSE, LOCALLY?

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <file1> <file2>"
    exit 1
fi

FILE1="$1"
FILE2="$2"

# Call Python and pass the arguments
python3 - <<END
import sys
sys.path.append("/Users/jthomas48/dissModels/fastTgcnVersions/tools")
import meshDecimation.py

file1 = "$FILE1"
file2 = "$FILE2"

myDecimate(inFile = file1, outFile = file2, nFace = 16000)
END