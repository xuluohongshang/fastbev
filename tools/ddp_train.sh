#!/usr/bin/env bash
set -x

CONFIG=$1
CFG_OPTIONS=$2
# RESUME_MODEL=$3

if [ -z "${RANK}" ]; then
    echo "warning: ddp env not found, will use default value"
    gpus=2
    WORLD_SIZE=1
    RANK=0
    MASTER_ADDR=localhost
    MASTER_PORT=23456
    echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX:debug"
fi



_MASTER_ADDR=$MASTER_ADDR
_MASTER_PORT=$MASTER_PORT

unset MASTER_ADDR
unset MASTER_PORT
unset RANK


echo "world size: ${WORLD_SIZE}"
echo "gpus: ${gpus}"
echo "CONFIG=$CONFIG"
echo "CFG_OPTIONS=$CFG_OPTIONS"


if [ "$2" ];then
    echo "CONFIG: ${CONFIG}"
    echo "CFG_OPTIONS: ${CFG_OPTIONS}"
    python -m torch.distributed.run \
        --nproc_per_node=${gpus} \
        --nnodes=${WORLD_SIZE} \
        --rdzv_id="lucas-training" \
        --rdzv_backend="c10d" \
        --rdzv_endpoint="${_MASTER_ADDR}:${_MASTER_PORT}" \
        tools/train.py ${CONFIG} --launcher pytorch --seed 666 --cfg-options  ${CFG_OPTIONS}
        # --no-validate \
        # --seed 666 \
        # --work-dir ${WORK_DIR} \
        # --load-from ${RESUME_MODEL}
        # --work-dir ${WORK_DIR} 
else
    echo "CONFIG: ${CONFIG}"
    python -m torch.distributed.run \
        --nproc_per_node=${gpus} \
        --nnodes=${WORLD_SIZE} \
        --rdzv_id="lucas-training" \
        --rdzv_backend="c10d" \
        --rdzv_endpoint="${_MASTER_ADDR}:${_MASTER_PORT}" \
        tools/train.py ${CONFIG} --launcher pytorch --seed 666
        # --no-validate \
        # --seed 666 \
        # --work-dir ${WORK_DIR} \
        # --load-from ${RESUME_MODEL}
        # --work-dir ${WORK_DIR} 
fi

## v100
# python -m torch.distributed.launch \
#          --nproc_per_node=${gpus} \
#          --nnodes=${WORLD_SIZE} \
#          --node_rank=${RANK} \
#          --master_addr=${MASTER_ADDR} \
#          --master_port=${MASTER_PORT} \
#          tools/train.py ${CONFIG} --launcher pytorch --work-dir ${WORK_DIR} \
#          --no-validate
#         #  --load-from ${RESUME_MODEL}
#          #  --resume-from ${RESUME_MODEL}