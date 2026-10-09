import argparse
import os
import shutil
import cv2
import numpy as np
from PIL import Image
from diffusers.models.attention_processor import XFormersAttnProcessor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler

from animation.modules.attention_processor import AnimationAttnProcessor
from animation.modules.attention_processor_normalized import AnimationIDAttnNormalizedProcessor
from animation.modules.face_model import FaceModel
from animation.modules.id_encoder import FusionFaceId
from animation.modules.pose_net import PoseNet
from animation.modules.unet import UNetSpatioTemporalConditionModel
from animation.pipelines.inference_pipeline_animation import InferenceAnimationPipeline
import random
import subprocess
from pathlib import Path

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def load_images_from_folder(folder, width, height):
    images = []
    files = os.listdir(folder)
    png_files = [f for f in files if f.endswith('.png')]
    png_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    for filename in png_files:
        img = Image.open(os.path.join(folder, filename)).convert('RGB')
        img = img.resize((width, height))
        images.append(img)

    return images

def save_frames_as_png(frames, output_path):
    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]
    num_frames = len(pil_frames)
    for i in range(num_frames):
        pil_frame = pil_frames[i]
        save_path = os.path.join(output_path, f'frame_{i}.png')
        pil_frame.save(save_path)

def save_frames_as_mp4(frames, output_mp4_path, fps):
    print("Starting saving the frames as mp4")
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'H264' for better quality
    out = cv2.VideoWriter(output_mp4_path, fourcc, fps, (width, height))
    for frame in frames:
        frame_bgr = frame if frame.shape[2] == 3 else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=125,
                       loop=0)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference script"
    )

    # KEEP
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True
    )

    ## KEEP
    #parser.add_argument(
    #    "--validation_image",
    #    type=str,
    #    default=None,
    #    help=(
    #        "A set of paths to the controlnext conditioning image be evaluated every `--validation_steps`"
    #        " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
    #        " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
    #        " `--validation_image` that will be used with all `--validation_prompt`s."
    #    ),
    #)
    parser.add_argument(
        "--start_image_path",
        type=str,
        default=None,
        help=(
            "A set of paths to the controlnext conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )

    # ===================
    # Pose Input
    #   Supply either `pose_images_dir`, `pose_video_path`, or `pre_pose_video_path`.
    #    - pose_images_dir has inputs already fully processed
    #    - pose_video_path just needs to be converted to frames
    #    - pre_pose_video_path has not been converted to pose data frames
    # ===================
    parser.add_argument(
        "--pose_images_dir",
        type=str,
        default=None,
        help=(
            "the validation control images"
        ),
    )
    parser.add_argument(
        "--pose_video_path",
        type=str,
        default=None,
        help=(
            "the validation control video (optional)"
        ),
    )
    parser.add_argument(
        "--pre_pose_video_path",
        type=str,
        default=None,
        help=(
            "video to convert to pose"
        ),
    )

    parser.add_argument(
        "--frame_output_dir",
        type=str,
        default=None,
        required=True
    )

    parser.add_argument(
        "--video_output_path",
        type=str,
        default=None,
        required=True
    )

    # TODO FIX
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        required=False
    )

    # TODO FIX
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        required=False
    )

    # KEEP
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.0,
        required=False
    )

    # KEEP
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        required=False
    )

    # KEEP
    parser.add_argument(
        "--posenet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained posenet model",
    )

    # KEEP
    parser.add_argument(
        "--face_encoder_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained face encoder model",
    )

    # KEEP
    parser.add_argument(
        "--unet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained unet model",
    )

    # KEEP
    parser.add_argument(
        "--tile_size",
        type=int,
        default=16,
        required=False
    )

    # KEEP
    parser.add_argument(
        "--overlap",
        type=int,
        default=4,
        required=False
    )

    # KEEP
    parser.add_argument(
        "--noise_aug_strength",
        type=float,
        default=0.0,  # or set to 0.02
        required=False
    )

    # KEEP
    parser.add_argument(
        "--frames_overlap",
        type=int,
        default=4,
        required=False
    )

    # KEEP
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    # KEEP
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )

    # KEEP
    parser.add_argument(
        "--decode_chunk_size",
        type=int,
        default=None,
        required=False
    )

    # Input and output video FPS (these may require separation later)
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        required=False
    )

    args = parser.parse_args()
    return args

def split_video_to_frames(video_path, output_path):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    filename_format = f"{output_path}/frame_%d.png"
    command = [
        'ffmpeg',
        '-i', video_path,
        '-q:v', '1',
        '-start_number', '0',
        filename_format,
    ]
    print(f"Command: {command}", flush=True)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    #video_info = json.loads(result.stdout)

def merge_frames_to_mp4(frames_dir, output_path, framerate=30):
    filename_format = f"{frames_dir}/frame_%d.png"
    command = [
        'ffmpeg',
        '-start_number', '0',
        '-framerate', str(framerate),
        '-i', filename_format,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_path,
        '-y',
    ]
    print(f"Command: {command}", flush=True)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    #video_info = json.loads(result.stdout)
                                                              

def prepare_pose_frames(args):
    pose_images_dir = args.pose_images_dir

    if not pose_images_dir:
        raise Exception('pose_images_dir must be set, even when using other arguments')
    
    Path(pose_images_dir).mkdir(parents=True, exist_ok=True)

    if args.pre_pose_video_path:
        print("Preparing pose frames from pre-pose video.")
        pre_pose_frame_dir = Path(f"{pose_images_dir}/frames")
        pre_pose_frame_dir.mkdir(parents=True, exist_ok=True)
        split_video_to_frames(args.pre_pose_video_path, pre_pose_frame_dir)
        reference_image_path = pre_pose_frame_dir / "frame_0.png"
        command = [
            "python", "DWPose/skeleton_extraction.py",
            "--target_image_folder_path", pre_pose_frame_dir,
            "--ref_image_path", reference_image_path,
            "--poses_folder_path", pose_images_dir,
        ]
        print(f"Command: {command}", flush=True)
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

    elif args.pose_video_path:
        print("Preparing pose frames from pose video.")
        split_video_to_frames(args.pose_video_path, pose_images_dir)
    else:
        print("Pose frames assumed to already exist.")

    # TODO: autodetect resolution
    pose_images = load_images_from_folder(pose_images_dir, width=args.width, height=args.height)
    return pose_images

def prepare_pose_frames_hack_static_frame(args):
    """
    This version is meant to copy the starting frame n-many times.
    """
    pose_images_dir = args.pose_images_dir

    if not pose_images_dir:
        raise Exception('pose_images_dir must be set, even when using other arguments')
    
    Path(pose_images_dir).mkdir(parents=True, exist_ok=True)

    args.start_image_path

    if args.pre_pose_video_path:
        print("Preparing pose frames from pre-pose video.")
        pre_pose_frame_dir = Path(f"{pose_images_dir}/frames")
        pre_pose_frame_dir.mkdir(parents=True, exist_ok=True)
        split_video_to_frames(args.pre_pose_video_path, pre_pose_frame_dir)

        #dir_files = sorted(os.listdir(pre_pose_frame_dir))
        filenames = []
        for entry in os.scandir(pre_pose_frame_dir):
            if entry.is_file():
                filenames.append(entry.path)

        filenames.sort()
        print(filenames)

        for i, filename in enumerate(filenames):
            print(i)
            if i > 1:
                break
            shutil.copyfile(args.start_image_path, filename)

        #reference_image_path = pre_pose_frame_dir / "frame_0.png"
        reference_image_path = args.start_image_path
        command = [
            "python", "DWPose/skeleton_extraction.py",
            "--target_image_folder_path", pre_pose_frame_dir,
            "--ref_image_path", reference_image_path,
            "--poses_folder_path", pose_images_dir,
        ]
        print(f"Command: {command}", flush=True)
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

    elif args.pose_video_path:
        print("Preparing pose frames from pose video.")
        split_video_to_frames(args.pose_video_path, pose_images_dir)
    else:
        print("Pose frames assumed to already exist.")

    # TODO: autodetect resolution
    pose_images = load_images_from_folder(pose_images_dir, width=args.width, height=args.height)
    return pose_images


if __name__ == "__main__":
    args = parse_args()

    # torch.set_default_dtype(torch.float16)
    seed = 23123134
    # seed = 42
    # seed = 123
    seed_everything(seed)
    generator = torch.Generator(device='cuda').manual_seed(seed)

    feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision)
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        low_cpu_mem_usage=True,
    )
    pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0])
    face_encoder = FusionFaceId(
        cross_attention_dim=1024,
        id_embeddings_dim=512,
        # clip_embeddings_dim=image_encoder.config.hidden_size,
        clip_embeddings_dim=1024,
        num_tokens=4, )
    face_model = FaceModel()

    lora_rank = 128
    attn_procs = {}
    unet_svd = unet.state_dict()

    for name in unet.attn_processors.keys():
        if "transformer_blocks" in name and "temporal_transformer_blocks" not in name:
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                # print(f"This is AnimationAttnProcessor: {name}")
                attn_procs[name] = AnimationAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
            else:
                # print(f"This is AnimationIDAttnProcessor: {name}")
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_svd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_svd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = AnimationIDAttnNormalizedProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
                attn_procs[name].load_state_dict(weights, strict=False)
        elif "temporal_transformer_blocks" in name:
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = XFormersAttnProcessor()
            else:
                attn_procs[name] = XFormersAttnProcessor()
    unet.set_attn_processor(attn_procs)

    # resume the previous checkpoint
    if args.posenet_model_name_or_path is not None and args.face_encoder_model_name_or_path is not None and args.unet_model_name_or_path is not None:
        print("Loading existing posenet weights, face_encoder weights and unet weights.")
        if args.posenet_model_name_or_path.endswith(".pth"):
            pose_net_state_dict = torch.load(args.posenet_model_name_or_path, map_location="cpu")
            pose_net.load_state_dict(pose_net_state_dict, strict=True)
        else:
            print("posenet weights loading fail")
            print(1/0)
        if args.face_encoder_model_name_or_path.endswith(".pth"):
            face_encoder_state_dict = torch.load(args.face_encoder_model_name_or_path, map_location="cpu")
            face_encoder.load_state_dict(face_encoder_state_dict, strict=True)
        else:
            print("face_encoder weights loading fail")
            print(1/0)
        if args.unet_model_name_or_path.endswith(".pth"):
            unet_state_dict = torch.load(args.unet_model_name_or_path, map_location="cpu")
            unet.load_state_dict(unet_state_dict, strict=True)
        else:
            print("unet weights loading fail")
            print(1/0)

    torch.cuda.empty_cache()
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    pose_net.requires_grad_(False)
    face_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    weight_dtype = torch.float16
    # weight_dtype = torch.float32
    # weight_dtype = torch.bfloat16

    pipeline = InferenceAnimationPipeline(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=noise_scheduler,
        feature_extractor=feature_extractor,
        pose_net=pose_net,
        face_encoder=face_encoder,
    ).to(device='cuda', dtype=weight_dtype)

    os.makedirs(args.frame_output_dir, exist_ok=True)

    #pose_images = prepare_pose_frames(args)
    pose_images = prepare_pose_frames_hack_static_frame(args)
    
    num_frames = len(pose_images)

    start_image_path = args.start_image_path
    start_image = Image.open(args.start_image_path).convert('RGB')

    face_model.face_helper.clean_all()
    start_image_face = cv2.imread(start_image_path)
    start_image_bgr = cv2.cvtColor(start_image_face, cv2.COLOR_RGB2BGR)
    start_image_face_info = face_model.app.get(start_image_bgr)
    if len(start_image_face_info) > 0:
        start_image_face_info = sorted(start_image_face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
        start_image_id_ante_embedding = start_image_face_info['embedding']
    else:
        start_image_id_ante_embedding = None

    if start_image_id_ante_embedding is None:
        face_model.face_helper.read_image(start_image_bgr)
        face_model.face_helper.get_face_landmarks_5(only_center_face=True)
        face_model.face_helper.align_warp_face()

        if len(face_model.face_helper.cropped_faces) == 0:
            start_image_id_ante_embedding = np.zeros((512,))
        else:
            start_image_align_face = face_model.face_helper.cropped_faces[0]
            print('fail to detect face using insightface, extract embedding on align face')
            start_image_id_ante_embedding = face_model.handler_ante.get_feat(start_image_align_face)

    # generator = torch.Generator(device=accelerator.device).manual_seed(23123134)

    decode_chunk_size = args.decode_chunk_size
    video_frames = pipeline(
        image=start_image,
        image_pose=pose_images,
        height=args.height,
        width=args.width,
        num_frames=num_frames,
        tile_size=args.tile_size,
        tile_overlap=args.frames_overlap,
        decode_chunk_size=decode_chunk_size,
        motion_bucket_id=127.,
        fps=7,
        min_guidance_scale=args.guidance_scale,
        max_guidance_scale=args.guidance_scale,
        noise_aug_strength=args.noise_aug_strength,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        output_type="pil",
        validation_image_id_ante_embedding=start_image_id_ante_embedding,
    ).frames[0]

    for i in range(num_frames):
        img = video_frames[i]
        video_frames[i] = np.array(img)

    os.makedirs(args.frame_output_dir, exist_ok=True)
    save_frames_as_png(video_frames, args.frame_output_dir)

    merge_frames_to_mp4(args.frame_output_dir, args.video_output_path, framerate=args.fps)


# bash command_basic_infer.sh
