#!/bin/bash

# some args
MODEL_NAME=$1
REGISTRY_URL=$2
TP_SIZE=$3

shift 3
REMAINING_ARGS=("$@")

host_name=$(hostname)
echo "[Rank $SLURM_PROCID] currently on $host_name"

# activate the vllm env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vpo  # we want to use a different clean env for vllm
echo "[Rank $SLURM_PROCID]" `which python`

# find the next available port after 8001
start_port=$((8001 + 100*$SLURM_LOCALID))  # in case they are on the same node
while true; do
    if ! ss -tuln | awk '{print $5}' | grep ":$start_port"; then
        echo "[Rank $SLURM_PROCID] Next available port: $start_port"
        break
    fi
    ((start_port++))
done

# make some exception for the 405B models
ADD_SERVING_ARGS=(
    --enforce-eager 
    --enable-prefix-caching
)
if [[ $MODEL_NAME == *"405B"* ]]; then
    ADD_SERVING_ARGS=()
fi

url_port=$(echo "$REGISTRY_URL" | cut -d':' -f2)

mkdir -p slogs/$url_port-$SLURM_JOB_ID
# start the vllm instance
vllm serve $MODEL_NAME \
    --port $start_port \
    --tensor-parallel-size $TP_SIZE "${ADD_SERVING_ARGS[@]}" "${REMAINING_ARGS[@]}" \
    1>slogs/$url_port-$SLURM_JOB_ID/$host_name\_$SLURM_LOCALID.output.log \
    2>slogs/$url_port-$SLURM_JOB_ID/$host_name\_$SLURM_LOCALID.error.log &

# send http request to register the worker
# we check if the service is actually running from the registry server side
while ! grep -q "Uvicorn running" slogs/$url_port-$SLURM_JOB_ID/$host_name\_$SLURM_LOCALID.error.log; do
    echo "[Rank $SLURM_PROCID] Waiting for model $MODEL_NAME to come up on $host_name:$start_port..."
    sleep 1m
done

curl \
    -s \
    -X POST \
    -H "Content-Type: application/json" \
    -d "{\"url\":\"http://$host_name:$start_port\", \"model_name\": \"$MODEL_NAME\"}" \
    http://$REGISTRY_URL/workers/add

# start collecting stats using vllm metrics and wandb
# NOTE: this will only start after the previous curl command (register successfully)
# export WANDB__SERVICE_WAIT=300

# python log_lm_wandb.py \
#     --port $start_port \
#     --model_name $MODEL_NAME \
#     --job_id $SLURM_JOB_ID \
#     --global_rank $SLURM_PROCID \
#     --hostname $host_name \
#     --query_every_s 5

wait