#!/bin/bash

    #--source_video_paths data/videos \

python process_data.py \
    --source_video_paths data/videos/dance_1.mp4 \
    --saved_pose_dir data/saved_pkl \
    --saved_pose data/saved_pose \
    --saved_frame_dir data/saved_frames \
    --model_directory checkpoints2

