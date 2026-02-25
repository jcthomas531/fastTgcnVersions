#!/bin/bash
#$ -l ngpus=3
#$ -pe smp 16
#$ -o outputFiles/$JOB_ID_predOut3.o
#$ -e outputFiles/$JOB_ID_predError3.e



apptainer exec ../../../containers/pytorch2.sif python predPipe3.py
