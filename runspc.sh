#!/bin/bash

# step stimulus
T=500
J=100.0
m=0.5
r=0.0
std=0.0
dpath="./data/J${J}-m${m}-r${r}_spc"
fpath="./plot/J${J}-m${m}-r${r}_spc"

rm -r $dpath
rm -r $fpath

# number of thread the computer supports to run in parallel
COUNTER=0
THREAD=3

sim=1
dat=0
fig=0

# simulations
# 2.7, 13.8, 2.6, 14.6 Hz (EPSV spontaneous)
inpE=10.5
inpP=53.0
inpS=6.5
inpV=11.0

# python runspc.py --inpE $inpE --inpP $inpP --inpS $inpS --inpV $inpV --recurrence $r --inhibit $m --conJ $J --T $T

if [ $sim -gt 0 ]
then
    for modV in 0
    do
        for modE in 0
        do
            for modP in `seq -1 1 1`
            do
                for modS in `seq -0.5 0.5 0.5`
                do
                    echo $COUNTER
                    if [ $COUNTER == $THREAD ]
                    then
                        python runspc.py --inpE $inpE --inpP $inpP --inpS $inpS --inpV $inpV --modE $modE --modP $modP --modS $modS --modV $modV --recurrence $r --inhibit $m --conJ $J --T $T
                        wait
                        COUNTER=0
                    else
                        python runspc.py --inpE $inpE --inpP $inpP --inpS $inpS --inpV $inpV --modE $modE --modP $modP --modS $modS --modV $modV --recurrence $r --inhibit $m --conJ $J --T $T &
                        COUNTER=$(( COUNTER + 1 ))
                    fi
                done
            done
        done
    done
fi
wait

exppath="./experiments/J${J}-m${m}-r${r}/frs"
if [ $dat -gt 0 ]
then
    mkdir $exppath

    # extract firing rate
    python extsim.py --dpath $dpath --fpath $fpath --rate --dist

    # analyse data
    mv ./data/*.json $exppath
    cp "${dpath}/E0.0P0.0S0.0V0.0/set.json" $exppath
fi

if [ $fig -gt 0 ]
then
    python anasim.py --exppath $exppath --transfer --var --sig --cov
fi