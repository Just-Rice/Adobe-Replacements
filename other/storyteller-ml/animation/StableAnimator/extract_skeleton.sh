#!/bin/bash

INPUT_VIDEO="inference/zeihan_trimmed.mp4"

OUTPUT_FRAMES_DIR="inference/output_1/frames"
OUTPUT_POSE_FRAMES_DIR="inference/output_1/poses"

mkdir -p $OUTPUT_FRAMES_DIR
mkdir -p $OUTPUT_POSE_FRAMES_DIR

ffmpeg -i $INPUT_VIDEO -q:v 1 -start_number 0 "${OUTPUT_FRAMES_DIR}/frame_%d.png"

python DWPose/skeleton_extraction.py \
  --target_image_folder_path="${OUTPUT_FRAMES_DIR}" \
  --ref_image_path="${OUTPUT_FRAMES_DIR}/frame_0.png" \
  --poses_folder_path="${OUTPUT_POSE_FRAMES_DIR}"
