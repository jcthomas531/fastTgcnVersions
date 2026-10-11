#!/bin/bash
#$ -pe smp 64
#$ -o machShop2Output.o
#$ -e machShop2Error.e



apptainer exec ../../../containers/lorwyn.sif Rscript machShop2.R
