#!/bin/bash
# profile_rank.sh

if [[ "$SLURM_PROCID" -eq 0 ]]; then
  METRICS="--gpu-metrics-device=all"
else
  METRICS=""
fi

model_size=$1
REPORT="/work/hdd/beih/yuli9/a40_${model_size}/pipe_${model_size}"
echo "Profiling model_size=${model_size}, report prefix=${REPORT}"
nsys profile --output "${REPORT}_${SLURM_PROCID}" $METRICS --cuda-memory-usage=true --force-overwrite=true --capture-range=cudaProfilerApi --capture-range-end=stop --stats=true --stop-on-exit=true python3 $HOME/cs533_final_project/pipeline.py "$model_size" nvidia-smi 
# done
