#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 script.sh output.log"
fi
SCRIPT=$1
LOG=$2

qsub -cwd -S /bin/bash -j y -P black-svr -l qp="cuda-low" -o $LOG $SCRIPT
