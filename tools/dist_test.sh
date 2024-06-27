#!/usr/bin/env bash

set -x 
CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29503}

startTime=`date +%Y%m%d-%H:%M:%S`
startTime_s=`date +%s`

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox  \
#     --cfg-options data.samples_per_gpu=8 data.workers_per_gpu=4


OUTDIR=work_dirs/fastbev/exp/evaluate
# OUTDIR=$(dirname $CHECKPOINT)/evaluate
mkdir -p $OUTDIR
echo $OUTDIR
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch \
    --out $OUTDIR/results.pkl \
    --format-only  --cfg-options data.samples_per_gpu=8 data.workers_per_gpu=4 \
    --eval-options jsonfile_prefix=$OUTDIR 2>&1 | tee $OUTDIR/log.test.$startTime.txt > /dev/null &

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/eval.py ${CONFIG} --launcher pytorch \
        --out $OUTDIR/results.pkl \
        --eval bbox 2>&1 | tee $OUTDIR/log.eval.$startTime.txt > /dev/null &


endTime=`date +%Y%m%d-%H:%M:%S`
endTime_s=`date +%s`
sumTime=$[ $endTime_s - $startTime_s ]
echo ""
echo "######################## " "$startTime ---> $endTime" "cost time: $sumTime seconds"
echo ""