#!/bin/bash

set -euxo pipefail

TEST_VENV_DIRECTORY="./python"

# NB: "python" is the name of our virtual environment python directory.
if [ ! -d "${TEST_VENV_DIRECTORY}" ]; then
    python3.6 -m venv "${TEST_VENV_DIRECTORY}"
fi

source python/bin/activate
pip install --upgrade pip
pip install -r requirements-tacotron-python36.txt

# Run tests in modules that support tests
pytest text -vv

