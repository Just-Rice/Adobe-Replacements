#!/bin/bash
# This is a helpful script for local development using Docker images.

# NB(bt,2023-11-27): At some point we stopped using venv within the Docker container
#  # cd models/tts
#  # source python/bin/activate
#  # ./vocodes_server.py

cd /models/tts && python3.6 vocodes_server.py

