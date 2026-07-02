#!/bin/bash
#$ -l ngpus=3
#$ -pe smp 32
#$ -o outputFiles/$JOB_ID_fastTgcnEasyOut.o
#$ -e outputFiles/$JOB_ID_fastTgcnEasyError.e



apptainer exec ../../../containers/lorwyn.sif python trainRunner2.py
