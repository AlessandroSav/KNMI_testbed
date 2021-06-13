#!/bin/bash

date=20200901 
end_date=20200915
# some script to process HARMONIE data
#
#
# run python script to create DALES input data

python create_input_NForcing.py $date $end_date

# copy input folder to BULL

pc_path="/nobackup/users/theeuwes/DALES_runs/2020*"
bull_path="theeuwes@bxshnr02:/nfs/home/users/theeuwes/work/DALES_runs/"

rsync -vau $pc_path $bull_path
