#!/bin/bash

jobid=$1

python3 -m pip install correctionlib==2.3.3
pip install --upgrade numpy==1.21.5
pip install scipy==1.10.1

# make dir for output (not really needed cz python script will make it)
mkdir outfiles

# run code
# pip install --user onnxruntime
python SCRIPTNAME --year YEAR --processor PROCESSOR PFNANO INFERENCE SYSTEMATICS GETLPWEIGHTS LOOSELEP TRIGGER PRESELECTION --n NUMJOBS --starti ${jobid} --sample SAMPLE --config METADATAFILE --channels CHANNELS

# remove incomplete jobs
rm -rf outfiles/*mu
rm -rf outfiles/*ele

#move output to eos
xrdcp -r -f outfiles/ EOSOUTPKL
