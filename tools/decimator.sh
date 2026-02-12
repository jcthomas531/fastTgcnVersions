#!/bin/bash

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