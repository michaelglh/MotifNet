#!/bin/bash

# step stimulus
T=500
J=25.0
m=0.5
r=0.0
std=0.0
dpath="./data/J${J}-m${m}-r${r}"
fpath="./plot/J${J}-m${m}-r${r}"

# number of thread the computer supports to run in parallel
COUNTER=1
THREAD=8

sim=1
fig=1

inpE=16.0
inpP=90.0
inpS=9.3
inpV=12.0

# low firing rate PV-dominated and SST-dominated
mEs=(-0.1 -0.6)
mPs=(0 -30)
mSs=(0 1.8)
mVs=(0 0)
as=(0.3 -0.3)
bs=(-0.3 0.3)

# # high firing rate PV-dominated
# mEs=(3.2)
# mPs=(-12.0)
# mSs=(-2.4)
# mVs=(0)
# as=(0.3)
# bs=(-0.3)

# samplings
epoch=100
for i in ${!mEs[@]}
do
    modE=${mEs[i]}
    modP=${mPs[i]}
    modS=${mSs[i]}
    modV=${mVs[i]}
    a=${as[i]}
    b=${bs[i]}
    
    if [ $sim -gt 0 ]
    then
        # python runsim.py --inpE $inpE --inpP $inpP --inpS $inpS --inpV $inpV --modE $modE --modP $modP --modS $modS --modV $modV --recurrence $r --inhibit $m --conJ $J --T $T

        COUNTER=1
        for gamma in 0.5
        do
            vtype='covariance'
            for pair in "0.1 0.1" "0.4 0.4" "0.1 0.4" "0.4 0.1"
            do
                ab=( $pair )
                echo "${ab[1]}"; echo "${ab[0]}";
                if [ $COUNTER == $THREAD ]
                then
                    python runsamp.py --epoch $epoch --vtype $vtype --alpha ${ab[0]} --beta ${ab[1]} --a $a --b $b --gamma $gamma --inpE $inpE --inpP $inpP --inpS $inpS --inpV $inpV --modE $modE --modP $modP --modS $modS --modV $modV --recurrence $r --inhibit $m --conJ $J --T $T
                    COUNTER=1
                else
                    python runsamp.py --epoch $epoch --vtype $vtype --alpha ${ab[0]} --beta ${ab[1]} --a $a --b $b --gamma $gamma --inpE $inpE --inpP $inpP --inpS $inpS --inpV $inpV --modE $modE --modP $modP --modS $modS --modV $modV --recurrence $r --inhibit $m --conJ $J --T $T &
                    COUNTER=$(( COUNTER + 1 ))
                fi
            done

            vtype='variance'
            for pair in "0.3 0.3" "0.3 -0.3" "-0.3 0.3" "3.0 3.0"
            do
                ab=( $pair )
                echo "${ab[1]}"; echo "${ab[0]}";
                if [ $COUNTER == $THREAD ]
                then
                    python runsamp.py --epoch $epoch --vtype $vtype --alpha ${ab[0]} --beta ${ab[1]} --gamma $gamma --inpE $inpE --inpP $inpP --inpS $inpS --inpV $inpV --modE $modE --modP $modP --modS $modS --modV $modV --recurrence $r --inhibit $m --conJ $J --T $T
                    COUNTER=1
                else
                    python runsamp.py --epoch $epoch --vtype $vtype --alpha ${ab[0]} --beta ${ab[1]} --gamma $gamma --inpE $inpE --inpP $inpP --inpS $inpS --inpV $inpV --modE $modE --modP $modP --modS $modS --modV $modV --recurrence $r --inhibit $m --conJ $J --T $T &
                    COUNTER=$(( COUNTER + 1 ))
                fi
            done
        done
        wait
    fi

    if [ $fig -gt 0 ]
    then
        python anasamp.py --epoch $epoch --vtype 'variance' --modE $modE --modP $modP --modS $modS --modV $modV --inhibit $m --conJ $J --a $a --b $b
        python anasamp.py --epoch $epoch --vtype 'covariance' --modE $modE --modP $modP --modS $modS --modV $modV --inhibit $m --conJ $J --a $a --b $b
    fi
done
