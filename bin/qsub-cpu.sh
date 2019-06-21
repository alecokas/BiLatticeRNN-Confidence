#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 script.sh output.log"
fi
SCRIPT=$1
LOG=$2

qsub -cwd -S /bin/bash -j y -P black-svr -l qp="low" -l mem_grab=4G -l mem_free=4G -pe smp 8 -o $LOG $SCRIPT
