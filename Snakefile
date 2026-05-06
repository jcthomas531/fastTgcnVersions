
#testing for proof of concept, will need to be more specific later with names

import os

in_dir = "K:/iowaRme/testDir/decProcess1"
out_dir = "K:/iowaRme/testDir/decProcess2"

# get all relevant filenames
files = [i for i in os.listdir(in_dir) if i.endswith(".ply")]
names = [i.replace(".ply", "") for i in files]


rule all:
    input:
		#require the following things to exist
		#the wildcard /{name} and what it stands for (given by names) is passed to any rule associated with this file
        expand(out_dir + "/{name}_dec016.ply", name=names)

rule dec016PreDScans:
	output:
		outFile = out_dir + "/{name}_dec016.ply"
	input:
		inFile = in_dir + "/{name}.ply"
	shell:
		"""
		python tools/processes/fullScanDecim_noLabs.py "{input.inFile}" "{output.outFile}"
		"""