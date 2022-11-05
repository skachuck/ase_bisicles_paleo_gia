#!/bin/bash -l

export PYTHONPATH=$PYTHONPATH:/global/cscratch1/sd/skachuck/ismip6results/scripts/intersection

for experiment in 10 ; do #5 7 9 10 13 ; do
    for rhe in gia-ub gia-best2 nogia ; do # nogia ; do
        cd ismip6-$experiment/$rhe
        pwd
        python ../../scripts/analyze_gia_run.py ./ ../../ismip6-$experiment.$rhe.pandas_stats.csv
        cd -
    done
done
