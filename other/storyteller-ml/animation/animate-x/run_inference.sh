#!/bin/bash

#python inference.py --cfg configs/Animate_X_infer.yaml

python inference_cli.py \
    --cfg configs/Animate_X_infer.yaml \
    --image_file data_test_copy/images/jeff_goldblum.jpg \
    --pose_directory data_test_copy/saved_pose/dance_2 \
    --frame_directory data_test_copy/saved_frames/dance_2 \
    --pickle_data_file data_test_copy/saved_pkl/dance_2.pkl \
    --model_checkpoints_directory checkpoints2 \
    --height 768 \
    --width 512

#    --height 512
#    --height 768
