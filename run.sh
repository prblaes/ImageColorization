#!/bin/sh

np=8
nn=15
q="batch"
if [ -n "$1" ]; then
    let nn="$1"
fi

if [ -n "$2" ]; then
    let np="$2"
fi

qsub -j oe -q $q -o /shared/users/asousa/ImageColorization/output/param_sweep_calhouses/log.txt -l nodes=$nn:ppn=$np -l walltime=100000:00:00 /shared/users/asousa/ImageColorization/imgc.pbs
