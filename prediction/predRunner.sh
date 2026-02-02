#!/bin/bash
#$ -l ngpus=3
#$ -pe smp 16
#$ -o outputFiles/$JOB_ID_predOut.o
#$ -e outputFiles/$JOB_ID_predError.e



apptainer exec ../../../containers/pytorch2.sif python newObsPred.py
