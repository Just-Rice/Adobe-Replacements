#!/bin/bash
# This script is meant to be launched on on-prem dual Nvidia GEFORCE 3090 hosts.
#
# It's meant to be provided with a numerical worker index and automatically run on
# the correct GPU and port, eg.
#
#   ./launch_worker.sh 4
#
# Once on-prem uses Kubernetes, this won't be necessary anymore.

set -euxo pipefail

declare -r WORKER_INDEX="$1"

# A dual 3090 can run eight jobs across two GPUs.
case "${WORKER_INDEX}" in
  "1")
    PORT=9001
    CUDA_VISIBLE_DEVICES=0
    ;;
  "2")
    PORT=9002
    CUDA_VISIBLE_DEVICES=0
    ;;
  "3")
    PORT=9003
    CUDA_VISIBLE_DEVICES=0
    ;;
  "4")
    PORT=9004
    CUDA_VISIBLE_DEVICES=0
    ;;
  "5")
    PORT=9005
    CUDA_VISIBLE_DEVICES=1
    ;;
  "6")
    PORT=9006
    CUDA_VISIBLE_DEVICES=1
    ;;
  "7")
    PORT=9007
    CUDA_VISIBLE_DEVICES=1
    ;;
  "8")
    PORT=9008
    CUDA_VISIBLE_DEVICES=1
    ;;

  *)
    echo "Invalid worker index as argument: ${WORKER_INDEX}"
    exit
    ;;
esac


# NB: This is the conventional venv directory
source python/bin/activate

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ./vocodes_server.py --port "${PORT}"
