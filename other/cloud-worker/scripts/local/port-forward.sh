#!/bin/bash
# Persistent port forward: localhost:$LOCAL_PORT -> pod ComfyUI (8188).
# Auto-reconnects if the tunnel drops. Ctrl-C to stop.
# Defaults to the B200 pod; override for others, e.g. the RTX 4090:
#   POD_HOST=213.173.102.163 POD_PORT=20497 LOCAL_PORT=8189 port-forward.sh
POD_HOST=${POD_HOST:-216.243.220.136}
POD_PORT=${POD_PORT:-17561}
LOCAL_PORT=${LOCAL_PORT:-8188}
while true; do
  ssh -N -L "$LOCAL_PORT":127.0.0.1:8188 \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -o StrictHostKeyChecking=accept-new \
    -p "$POD_PORT" root@"$POD_HOST"
  echo "tunnel dropped, reconnecting in 3s..."
  sleep 3
done
