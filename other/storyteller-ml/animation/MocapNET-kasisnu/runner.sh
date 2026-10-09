#!/bin/bash

set -ex

source /model_code/MocapNET/src/python/mnet4/pythonVirtualEnvironment/bin/activate && python3 $@ && mv out.bvh /tmp/some-input.bvh && cd /model_code/MocapNET/ && ./GroundTruthDumper --from dependencies/RGBDAcquisition/opengl_acquisition_shared_library/opengl_depth_and_color_renderer/Motions/DAZFriendlyCGSPEED_ZXYAndHandsAxisBigHands.bvh --merge /tmp/some-input.bvh dependencies/RGBDAcquisition/opengl_acquisition_shared_library/opengl_depth_and_color_renderer/Motions//mergeDazFriendlyAndAddHead.profile --setPositionRotation 0 0 0 0 0 0 --bvh /model_code/MocapNET/src/python/mnet4/out.bvh
