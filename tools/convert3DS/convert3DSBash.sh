#!/bin/bash
#$ -pe smp 32
#$ -o outputFiles/$JOB_ID_convert3DSOut.o
#$ -e outputFiles/$JOB_ID_convert3DSError.e



apptainer exec ../../../../containers/lorwyn.sif python -u convert3DSRunner.py