#!/bin/bash

OUTPUT_DIR="inference/output_2/final_output"

POSE_FRAME_DIR="inference/output_2/poses"
STARTING_IMAGE="inference/output_1/frames/frame_0.png"

#mkdir -p $OUTPUT_DIR

#CUDA_VISIBLE_DEVICES=0 
python inference_advanced.py \
 --start_image_path="${STARTING_IMAGE}" \
 --frame_output_dir="${OUTPUT_DIR}" \
 --pose_images_dir="/tmp/pose" \
 --pre_pose_video_path="inference/zeihan_trimmed.mp4" \
 --width=1024 \
 --height=576 \
 --pretrained_model_name_or_path="checkpoints/stable-video-diffusion-img2vid-xt" \
 --posenet_model_name_or_path="checkpoints/Animation/pose_net.pth" \
 --face_encoder_model_name_or_path="checkpoints/Animation/face_encoder.pth" \
 --unet_model_name_or_path="checkpoints/Animation/unet.pth" \
 --guidance_scale=3.0 \
 --num_inference_steps=25 \
 --tile_size=16 \
 --overlap=4 \
 --noise_aug_strength=0.02 \
 --frames_overlap=4 \
 --decode_chunk_size=4 \
 --gradient_checkpointing


#ffmpeg -framerate 30 \
#  -start_number 0 \
#  -i "${OUTPUT_DIR}/animated_images/frame_%d.png" \
#  -c:v libx264 -pix_fmt yuv420p "${OUTPUT_DIR}/out.mp4" -y
                                                              
