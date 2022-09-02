#!/bin/bash

### IMPORTANT: What I need to change:
### 1. PBS section: processors, memory, walltime, email, job name
### 2. loading application
### 3. Program command section
### 4. Check notes at end regarding GPU's, Gerhard


### Allocate one node and one "processor/CPU"
#PBS -l select=1:ncpus=4
### Allocate 1GB Memory
###PBS -l mem=128gb
### How long will the script/program run. Defines the queue and when to stop
### scripts/programs running too long
#PBS -l walltime=04:00:00
### Notify via email at (a)bort, (b)eginning and (e)nd
#PBS -m abe
### email adr of user
#PBS -M 19062095@sun.ac.za
### Name of the job
#PBS -N test_python_hpc
### Write any errors to the following file
#PBS -e errors.txt
### Write any STDOUT to the following file
#PBS -o standout.txt

### Everything that follows this will be executed:

# make sure I'm the only one that can read my output
umask 0077
# create a temporary directory with the job ID as name on node or network
# TMP=/scratch-small-local/${PBS_JOBID}
TMP=/scratch2/${PBS_JOBID}
mkdir -p ${TMP}
echo "Temporary work dir: ${TMP}"

# copy the input files to ${TMP}
echo "Copying from ${PBS_O_WORKDIR}/ to ${TMP}/"
/usr/bin/rsync -vax "${PBS_O_WORKDIR}/" ${TMP}/

cd ${TMP}

# program command
## load modules

module load python/3.7.2


which python >> which.txt

python3.7 run_sim.py

# cleanup

echo "Copying from ${TMP}/ to ${PBS_O_WORKDIR}/"
/usr/bin/rsync -vax ${TMP}/ "${PBS_O_WORKDIR}/"

# if the copy back succeeded, delete my temporary files
[ $? -eq 0 ] && /bin/rm -rf ${TMP}
echo "Temporary files removed"

### IMPORTANT: GPU support only at end of September, when giving GROMACS instructions don't include GPU's
### Once GPU works, do run on CPU's and GPU's exclusively, compare and send to Gerhard
