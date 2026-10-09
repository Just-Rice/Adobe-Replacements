import requests
import argparse
import os
import json
import time
from typing import Dict, Any, Optional
from jsonpath_ng import parse
import shutil
from pathlib import Path
from enum import IntEnum, auto
import os
import subprocess
import signal
import websockets
import httpx
import asyncio
import urllib
import uuid
import re

client_id = str(uuid.uuid4())

SERVER_IP = os.environ.get("SERVER_IP", "127.0.0.1:8188")
SERVER_START_TIMEOUT = int(os.environ.get("SERVER_START_TIMEOUT", 60))

TIMEOUT_SECONDS = int(os.environ.get("TIMEOUT_SECONDS", 1000))
PROMPT_ENDPOINT = f"http://{SERVER_IP}/prompt"
V2V_WORKFLOWS_DIRECTORY = os.environ.get("V2V_WORKFLOWS_DIRECTORY",
                                         "/workflow_configs")
COMFY_ROOT_DIRECTORY = os.environ.get("COMFY_ROOT_DIRECTORY",
                                      "/app/ComfyUI")

PREVIEW_IPA_WORKFLOW = os.environ.get("PREVIEW_IPA_WORKFLOW", "06-08-2024/yae_vid2vid_FirstPassPreview_30_07_API.json")
MAIN_IPA_WORKFLOW = os.environ.get("MAIN_IPA_WORKFLOW", "06-08-2024/yae_vid2vid_main_29-07_API.json")
FACE_DETAILER_WORKFLOW = os.environ.get("FACE_DETAILER_WORKFLOW", "3-06-2024/yae_vid2vid_FaceDetailer_1-07_API.json")
UPSCALER_WORKFLOW = os.environ.get("UPSCALER_WORKFLOW", "06-08-2024/yae_vid2vid_Upscale_25-07_API.json")
CINEMATIC_WORKFLOW = os.environ.get("CINEMATIC_WORKFLOW", "3-06-2024/yae_vid2vid_better_main+upscaler_04-07_API.json")
COG_V1_WORKFLOW = os.environ.get("COG_V1_WORKFLOW", "14-11-2024/yae_15-10_COG_API.json")

class PipelineType(IntEnum):
    BASE = auto()
    IPA = auto()
    FACE_DETAILER = auto()
    UPSCALER = auto()
    CINEMATIC = auto()
    PREVIEW = auto()
    COG_V1 = auto()


style_to_filename = {
    "anime_2_5d": "1_2.5d_anime_model.json",
    "anime_2d_flat": "2_2d_flat_anime_model.json",
    "cartoon_3d": "3_3d_cartoon_style.json",
    "comic_book": "4_comic_book_model.json",
    "anime_ghibli": "5_ghibli_anime_model.json",
    "ink_punk": "6_ink_punk.json",
    "ink_splash": "7_ink_splash.json",
    "ink_bw_style": "8_ink_w_and_b_style.json",
    "jojo_style": "9_jojo_style.json",
    "paper_origami": "10_paper_origami.json",
    "pixel_art": "11_pixel_art.json",
    "pop_art": "12_pop_art.json",
    "realistic_1": "13_realistic_1.json",
    "realistic_2": "14_realistic_2.json",
    "anime_retro_neon": "15_retro_neon_anime_90.json",
    "anime_standard": "16_standard_anime_model.json",
    "hr_giger": "17_hr_giger.json",
    "simpsons": "18_simpsons.json",
    "carnage": "19_carnage.json",
    "pastel_cute_anime": "20_pastel_cute_anime.json",
    "bloom_lighting": "21_bloom_lighting.json",
    "25d_horror": "22_25D_Horror.json",
    "creepy": "23_creepy.json",
    "creepy_vhs": "24_creepy_vhs.json",
    "trail_cam_footage": "25_trail_cam_footage.json",
    "old_black_white_movie": "26_old_black_white_movie.json",
    "horror_noir_black_white": "27_horror_noir_black_white.json",
    "techno_noir_black_white": "28_techno_noir_black_white.json",
    "black_white_20s": "29_black_white_20s.json",
    "cyberpunk_anime": "30_cyberpunk_anime.json",
    "dragonball": "31_dragonball.json",
    "realistic_matrix": "32_realistic_matrix.json",
    "realistic_cyberpunk": "33_realistic_cyberpunk.json",
    "dreamer": "34_dreamer.json",
}


def ensure_pipeline_input_present():
    left = Path(COMFY_ROOT_DIRECTORY) / "input/input.mp4"
    right = Path(COMFY_ROOT_DIRECTORY) / "ComfyUI" / "input/input.mp4"
    shutil.copy(left, right)

def convert_pipline_output_to_input(left):
    right = Path(COMFY_ROOT_DIRECTORY) / "input/input.mp4"
    shutil.copy(left, right)

def copy_frames_to_directory(stage, frames_dir):
    frames = Path(COMFY_ROOT_DIRECTORY) / "output" / "vid2vid" / stage / "Frames"
    frame_idx_matcher = re.compile('.*Frame_(\d+)*')
    for file in frames.iterdir():
        # output/vid2vid/FirstPass/Frame_00002_.png
        # output/vid2vid/SecondPass/Frames/Frame_0006.jpg
        # remove trailing prefiix and Frame_ prefix
        file_name = file.name
        frame_idx = frame_idx_matcher.match(file_name).group(1)
        target_file_name = f"{frame_idx.zfill(5)}{file.suffix}"
        shutil.copy(file, frames_dir / target_file_name)
        print(f"Copying {file} to {frames_dir / target_file_name}")

def generated_frames_count(frames_dir):
    return len(list(frames_dir.glob("*.jpg")))

# ffmpeg -i input.mp4 -qscale:v 2 output_%03d.jpg
async def extract_frames(input_video, output_directory):
    if not output_directory.exists():
        output_directory.mkdir(parents=True)
    command = [
        'ffmpeg',
        '-i', str(input_video),
        '-qscale:v', '2',
        str(f'{output_directory}/Frame_%05d.jpg')
    ]
    await cmd_runner(" ".join(command))
    print("Frames extracted successfully.")

def pad_frames(frames_dir, expected_count):
    # Copy last frames over till we reach expected_count
    # find first frame that exists
    last_frame = None
    last_frame_idx = 0
    for i in range(expected_count, 0, -1):
        frame = frames_dir / f"{i:05d}.jpg"
        if frame.exists():
            last_frame = frame
            last_frame_idx = i
            break
    for idx in range(last_frame_idx + 1, expected_count + 1):
        print(f"Copying padding frame {last_frame} to Frame_{idx:05d}.jpg")
        shutil.copy(last_frame, frames_dir / f"{idx:05d}.jpg")

def restore_last_pipeline_output(left):
    print("Restoring last pipeline output to input from: " + str(left))
    right = Path(COMFY_ROOT_DIRECTORY) / "output" / "vid2vid/SparseUpscaleInterp_00001.mp4"
    shutil.copy(left, right)

def ensure_frames_dir_present(stage):
    # output/vid2vid/FirstPass/Frames
    frames_dir = Path(COMFY_ROOT_DIRECTORY) / "output" / "vid2vid" / stage / "Frames"
    if not frames_dir.exists():
        frames_dir.mkdir(parents=True)

def ensure_frames_dir_empty(stage):
    frames_dir = Path(COMFY_ROOT_DIRECTORY) / "output" / "vid2vid" / stage / "Frames"
    if frames_dir.exists():
        for file in frames_dir.iterdir():
            try:
                file.unlink()
            except Exception as e:
                print(f"Error while deleting {file}")
                print(e)
                exit(1)

def ensure_output_directory_empty():
    shutil.rmtree(Path(COMFY_ROOT_DIRECTORY) / "output")
    (Path(COMFY_ROOT_DIRECTORY) / "output").mkdir(parents=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Run Comfy')
    parser.add_argument('--prompt', type=str, help='location of the prompt json', required=False)
    parser.add_argument('--style', type=str, help='style name', required=False)
    parser.add_argument('--global_ipa_image_filename', type=str, help='Global IPA image (optional)', required=False)
    parser.add_argument('--global_ipa_strength', type=float, help='Global IPA strength (optional)', required=False, default=1.0)
    parser.add_argument('--positive_prompt_filename', type=str, help='positive prompt', required=False)
    parser.add_argument('--negative_prompt_filename', type=str, help='negative prompt', required=False)
    parser.add_argument('--travel_prompt_filename', type=str, help='travel prompt', required=False)
    parser.add_argument('--pause-between-steps', type=int, help='pause between steps', required=False, default=1)
    parser.add_argument('--face-detailer-enabled', help='face detailer enabled', required=False, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--upscaler-enabled', help='upscale enabled', required=False, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--lipsync-enabled', help='lipsync enabled', required=False, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--disable-lcm', help='disable lcm (only applies to main workflow)', required=False, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--strength', type=float, help='strength', required=False, default=1.0)
    parser.add_argument('--frame_skip', type=int, help='frame skipping', required=False, default=None)
    
    parser.add_argument('--depth_video_filename', type=str, help='path of mp4 depth from engine for preprocessing', required=False)
    parser.add_argument('--outline_video_filename', type=str, help='path of mp4 outline from engine for preprocessing', required=False)
    parser.add_argument('--normals_video_filename', type=str, help='path of mp4 normals from engine for preprocessing', required=False)
    parser.add_argument('--generate-previews', help='generate previews', required=False, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--preview-frames-directory', type=str, help='previews directory', required=False)
    parser.add_argument('--enable-cinematic', help='enable cinematic', required=False, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--enable-cogvideo', help='enable cog video', required=False, default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument('--skip-comfy-startup', action='store_true', help='controls if this script tries to start comfy', default=False)
    parser.add_argument('--server-restart-trigger-file', type=str, help='server restart trigger file', required=False)
    return parser.parse_args()

args = parse_args()

def get_queued_jobs_count(server_address):
    try:
        response = requests.get(f"http://{server_address}/queue", timeout=5)
    except requests.exceptions.RequestException:
        print("Error while checking queue")
        return None
    queue_json = response.json()
    return len(queue_json.get("queue_pending", []))

def ensure_server_is_healthy(server_address):
    # check the health of the server (5 second timeout)
    response = requests.get(f"http://{server_address}/queue", timeout=5)
    if response.status_code == 200:
        print("Server is healthy")
    else:
        print("Server is unhealthy")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")

    # make sure there are no jobs already
    queue_json = response.json()
    if len(queue_json.get("queue_pending", [])) > 0:
        print("There are jobs in the queue already. Exiting...")
        exit(1)
    else:
        print("Queue is empty")

def interrupt_comfy_queue(server_address):
    requests.post(f"http://{server_address}/interrupt")
    print("Queue interrupted successfully.")

def wait_for_comfy_startup(server_address, timeout=SERVER_START_TIMEOUT):
    prompt_endpoint = f"http://{server_address}/prompt"
    start_time = time.time()
    started = False
    request_timeout = 2
    while time.time() - start_time < timeout:
        try:
            response = requests.get(prompt_endpoint, timeout=request_timeout)
            if response.status_code == 200:
                print("Server is healthy")
                started = True
                break
        except requests.exceptions.ConnectionError:
            print(f"Server is not up yet. Time elapsed: {time.time() - start_time:.2f}")
        time.sleep(1)
    if not started:
        print(f"Server failed to start in {SERVER_START_TIMEOUT} seconds")
        exit(1)

def invoke_facefusion(input_audio, input_video, output_video):
    # invoke facefusion
    command = [
        '/app/facefusion-runner.sh',
        '--input_audio', input_audio,
        '--input_video', input_video,
        '--output', output_video
    ]
    subprocess.check_output(command)
    print("Facefusion completed successfully.")

async def main():
    if not args.skip_comfy_startup:
        print("Starting comfy server")
        try:
            comfyui_process = subprocess.Popen(["../venv/bin/python", "main.py", "--highvram", "--disable-metadata"], cwd=COMFY_ROOT_DIRECTORY, stderr=subprocess.STDOUT, stdin=subprocess.PIPE)
            # await asyncio.create_subprocess_exec(
            #     "../venv/bin/python", "main.py", "--highvram", "--disable-metadata", cwd=COMFY_ROOT_DIRECTORY,
            #     stdout=subprocess.STDOUT,
            #     stderr=subprocess.STDOUT,
            #     stdin=subprocess.PIPE
            # )
        except Exception as e:
            print("Error while starting comfy server")
            print(e)
            exit(1)
        await asyncio.sleep(15)
    else:
        print("Skipping comfy server startup")
    # interrupt_comfy_queue(SERVER_IP)
    wait_for_comfy_startup(SERVER_IP)

    # print our process id, parent group id, comfyui process id, comfyui process group id, comfyui process group leader id, comfyui process group leader parent id
    # print(f"Process ID: {os.getpid()}")
    # print(f"Parent Group ID: {os.getpgrp()}")
    # print(f"ComfyUI Process ID: {comfyui_process.pid}")
    # print(f"ComfyUI Process Group ID: {os.getpgid(comfyui_process.pid)}")
    # print(f"ComfyUI Process Group Leader ID: {os.getpgid(comfyui_process.pid)}")
    # print(f"ComfyUI Process Group Leader Parent ID: {os.getppid()}")
    pipeline_type = PipelineType.IPA

    output_index = 0
    file_path = args.prompt
    prompt = None
    if args.enable_cogvideo:
        print("Generating prompt json for COG V1")
        workflow_filename = COG_V1_WORKFLOW
        positive_prompt_filename = args.positive_prompt_filename
        step_count = 25
        strength = 1.4
        if args.strength != 1.0:
            # Could we even make better spaghetti?
            strength = args.strength
        positive_prompt = ''
        if positive_prompt_filename:
            positive_prompt = open(positive_prompt_filename).read()
        ensure_output_directory_empty()
        prompt = generate_prompt_for_cog_v1(
            workflow_filename,
            positive_prompt,
            step_count,
            strength
        )
        print("running pipeline" + str(pipeline_type))
        await enqueue_prompt_and_wait(prompt)
        left = Path(COMFY_ROOT_DIRECTORY) / "output" / f"vid2vid/CogVideoX-I2V_00001.mp4"
        if left.exists():
            print("Pipeline completed successfully to output file: " + str(left))
        else:
            print("Pipeline failed to produce output")
            exit(1)
        restore_last_pipeline_output(left)
        return


    if args.style is not None:
        print("Generating prompt json")
        style = args.style
        positive_prompt_filename = args.positive_prompt_filename
        negative_prompt_filename = args.negative_prompt_filename
        travel_prompt_filename = args.travel_prompt_filename
        positive_prompt = None
        negative_prompt = None
        travel_prompt = None
        workflow_filename = None
        enable_lipsync = args.lipsync_enabled
        disable_lcm = args.disable_lcm
        cinematic_workflow_enabled = args.enable_cinematic
        denoise_first_pass = args.strength
        # defaults to none
        depth_video_filename = args.depth_video_filename
        outline_video_filename = args.outline_video_filename
        normals_video_filename = args.normals_video_filename
        if positive_prompt_filename:
            positive_prompt = open(positive_prompt_filename).read()
        if negative_prompt_filename:
            negative_prompt = open(negative_prompt_filename).read()
        if travel_prompt_filename:
            travel_prompt = open(travel_prompt_filename).read()
        ensure_output_directory_empty()
        if args.generate_previews:
            previews_directory = args.preview_frames_directory
            if not previews_directory:
                print("Previews directory not set")
                exit(1)
            ensure_frames_dir_present("FirstPass")
            if not (Path(previews_directory) / "first_pass" ).exists():
                (Path(previews_directory) / "first_pass").mkdir(parents=True)
            ensure_frames_dir_empty("FirstPass")
            workflow_filename = PREVIEW_IPA_WORKFLOW
            pipeline_type = PipelineType.PREVIEW
            prompt = generate_prompt_for_style(
                style,
                positive_prompt,
                negative_prompt,
                travel_prompt,
                workflow_filename,
                PipelineType.PREVIEW,
                denoise_first_pass=denoise_first_pass,
                enable_lipsync=enable_lipsync,
                disable_lcm=disable_lcm,
                global_ipa_image_filename=args.global_ipa_image_filename,
                global_ipa_strength=args.global_ipa_strength,
                frame_skip=args.frame_skip,
                depth_video_filename=depth_video_filename,
                outline_video_filename=outline_video_filename,
                normals_video_filename=normals_video_filename
            )
            print("running pipeline" + str(pipeline_type))
            await enqueue_prompt_and_wait(prompt)
            copy_frames_to_directory("FirstPass", Path(previews_directory) / "first_pass")
            pipeline_type = PipelineType.IPA
            workflow_filename = MAIN_IPA_WORKFLOW

        ensure_output_directory_empty()
        if pipeline_type == PipelineType.IPA:
            workflow_filename = MAIN_IPA_WORKFLOW

        time_before = time.perf_counter()

        validate_style_name(style)

        prompt = generate_prompt_for_style(
            style, 
            positive_prompt, 
            negative_prompt, 
            travel_prompt, 
            workflow_filename, 
            denoise_first_pass=denoise_first_pass,
            enable_lipsync=enable_lipsync,
            enable_cinematic=cinematic_workflow_enabled,
            disable_lcm=disable_lcm,
            global_ipa_image_filename=args.global_ipa_image_filename,
            global_ipa_strength=args.global_ipa_strength,
            frame_skip=args.frame_skip,
            depth_video_filename=depth_video_filename,
            outline_video_filename=outline_video_filename,
            normals_video_filename=normals_video_filename
        )

        print("running pipeline" + str(pipeline_type))
        await enqueue_prompt_and_wait(prompt)
        output_index = output_index + 1
        left = Path(COMFY_ROOT_DIRECTORY) / "output" / f"vid2vid/SparseUpscaleInterp_0000{output_index}.mp4"
        if left.exists():
            print("Pipeline completed successfully to output file: " + str(left))
        else:
            print("Pipeline failed to produce output")
            exit(1)
        if args.upscaler_enabled or args.enable_cinematic:
            pipeline_type = PipelineType.UPSCALER
            print("running pipeline" + str(pipeline_type))
            convert_pipline_output_to_input(left)
            workflow_filename = UPSCALER_WORKFLOW
            prompt = generate_prompt_for_style(
                style,
                positive_prompt,
                negative_prompt,
                travel_prompt,
                workflow_filename,
                pipeline_type,
                denoise_first_pass=denoise_first_pass,
                enable_lipsync=enable_lipsync,
                enable_cinematic=cinematic_workflow_enabled,
                global_ipa_image_filename=args.global_ipa_image_filename,
                global_ipa_strength=args.global_ipa_strength,
                frame_skip=args.frame_skip
            )
            if args.generate_previews:
                previews_directory = args.preview_frames_directory
                if not previews_directory:
                    print("Previews directory not set")
                    exit(1)
                if not (Path(previews_directory) / "second_pass" ).exists():
                    (Path(previews_directory) / "second_pass").mkdir(parents=True)
                ensure_frames_dir_empty("SecondPass")
            await enqueue_prompt_and_wait(prompt)
            if args.generate_previews:
                copy_frames_to_directory("SecondPass", Path(previews_directory) / "second_pass")
            output_index = output_index + 1
            left = Path(COMFY_ROOT_DIRECTORY) / "output" / f"vid2vid/SparseUpscaleInterp_0000{output_index}.mp4"
            if left.exists():
                print("Pipeline completed successfully to output file: " + str(left))
            else:
                print("Pipeline failed to produce output")
                exit(1)
        if args.face_detailer_enabled:
            pipeline_type = PipelineType.FACE_DETAILER
            print("running pipeline" + str(pipeline_type))
            convert_pipline_output_to_input(left)
            workflow_filename = FACE_DETAILER_WORKFLOW
            prompt = generate_prompt_for_style(
                style, 
                positive_prompt, 
                negative_prompt, 
                travel_prompt, 
                workflow_filename, 
                pipeline_type,
                denoise_first_pass=denoise_first_pass,
                enable_lipsync=enable_lipsync,
                global_ipa_image_filename=args.global_ipa_image_filename,
                global_ipa_strength=args.global_ipa_strength,
                frame_skip=args.frame_skip
            )
            await enqueue_prompt_and_wait(prompt)
            output_index = output_index + 1
            left = Path(COMFY_ROOT_DIRECTORY) / "output" / f"vid2vid/SparseUpscaleInterp_0000{output_index}.mp4"
        if enable_lipsync:
            # Ugh
            input_audio = Path(COMFY_ROOT_DIRECTORY) / "input" / "trimmed.wav"
            input_video = Path(COMFY_ROOT_DIRECTORY) / "output" / f"vid2vid/SparseUpscaleInterp_0000{output_index}.mp4"
            output_index = output_index + 1
            output_video = Path(COMFY_ROOT_DIRECTORY) / "output" / f"vid2vid/SparseUpscaleInterp_0000{output_index}.mp4"
            try:
                invoke_facefusion(input_audio, input_video, output_video)
            except Exception as e:
                print("Facefusion failed!")
                print(e)
            # check if output_video file exists
            if not output_video.exists():
                print("Facefusion failed!")
            else:
                left = output_video
        restore_last_pipeline_output(left)
        await extract_frames(left, Path(COMFY_ROOT_DIRECTORY) / "output" / "vid2vid" / "FinalPass" / "Frames")
        if not (Path(previews_directory) / "final_pass" ).exists():
            (Path(previews_directory) / "final_pass").mkdir(parents=True)
        copy_frames_to_directory("FinalPass", Path(previews_directory) / "final_pass")
        first_pass_frame_count = generated_frames_count(Path(previews_directory) / "first_pass")
        second_pass_frame_count = generated_frames_count(Path(previews_directory) / "second_pass")
        final_pass_frame_count = generated_frames_count(Path(previews_directory) / "final_pass")

        if second_pass_frame_count < first_pass_frame_count:
            pad_frames(Path(previews_directory) / "second_pass", first_pass_frame_count)
        if final_pass_frame_count < first_pass_frame_count:
            pad_frames(Path(previews_directory) / "final_pass", first_pass_frame_count)

        time_after = time.perf_counter()
        print(f"Time taken: {time_after - time_before:.2f} seconds")
    else:
        print(f'Loading prompt json from path: {file_path}')
        prompt_workflow = json.load(open(file_path))
        prompt = {"prompt": prompt_workflow, "client_id": client_id}
        await enqueue_prompt_and_wait(prompt)

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(SERVER_IP, prompt_id)) as response:
        return json.loads(response.read())

def get_queue(server_ip):
    with urllib.request.urlopen("http://{}/queue".format(server_ip)) as response:
        return json.loads(response.read())

def confirm_prompt_running_or_pending(prompt_id):
    queued_response = get_queue(SERVER_IP)
    prompt_possibly_queued = False

    for possible_state in ['queue_running', 'queue_pending']:
        for j in queued_response.get(possible_state,[]):
            try:
                if j[1] == prompt_id:
                    prompt_possibly_queued = True
                    break
            except IndexError:
                pass
    return prompt_possibly_queued 


async def smart_wait(ws, prompt_id, max_wait_time=10, prompt_execution_timeout=TIMEOUT_SECONDS):
    execution_start_time = time.monotonic_ns()
    last_prompt_update_time = time.monotonic_ns()
    while True:
        try:
            out = await ws.recv()
            last_msg_time = time.monotonic_ns()
            if last_msg_time - execution_start_time > prompt_execution_timeout * 1e9:
                print(f"Prompt {prompt_id} - execution time exceeded {prompt_execution_timeout} seconds")
                break
            if isinstance(out, str):
                message = json.loads(out)
                if message.get('data', None) and message['data'].get('prompt_id', None) == prompt_id:
                    last_prompt_update_time = time.monotonic_ns()
                    # print(f"Prompt {prompt_id} updated at {last_prompt_update_time}")
                if (last_msg_time - last_prompt_update_time > max_wait_time * 1e9) and not confirm_prompt_running_or_pending(prompt_id):
                    print(f"Prompt {prompt_id} might be stuck")
                    break
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break
            else:
                print('received binary message')
                if last_msg_time - last_prompt_update_time > max_wait_time * 1e9:
                    print(f"Prompt {prompt_id} might be stuck")
                    break
                continue
        except websockets.exceptions.ConnectionClosedError:
            print('Connection closed')
            break
    print(f"Prompt {prompt_id} - finished finished")

async def enqueue_prompt_and_wait(prompt):
    data = json.dumps(prompt).encode('utf-8')

    with httpx.Client() as client:
        response = client.post(PROMPT_ENDPOINT, data=data)
        if response.status_code == 200:
            print("Job enqueued successfully")
        else:
            print("Job failed to queue")
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.text}")
            exit(1)
        prompt_id = response.json()["prompt_id"]
        # Wait for the job to finish
        print(f"Waiting for job to finish (timeout: {TIMEOUT_SECONDS} seconds)")
        execution_start_time = time.perf_counter()
        async with websockets.connect("ws://{}/ws?clientId={}".format(SERVER_IP, client_id), max_size=1048576 * 10) as ws:
            await smart_wait(ws, prompt_id)
            current_time = time.perf_counter()
            execution_time = current_time - execution_start_time
            print('request executed in {:.2f} seconds'.format(execution_time))



def generate_prompt_for_cog_v1(workflow_filename, positive_prompt, step_count, strength=1.4):
        workflow_directory = Path(V2V_WORKFLOWS_DIRECTORY) / "workflows"
        workflow_json = json.load(open(workflow_directory / workflow_filename))

        modifications = {
            "$.320.inputs.text": positive_prompt,
            "$.464.inputs.value": step_count,
            "$.465.inputs.value": strength,
        }
        json_prompt = apply_jsonpath_mods(modifications, workflow_json)

        p = {"prompt": json_prompt, "client_id": client_id}
        return p


def generate_prompt_for_style(style_name, 
                              positive_prompt, 
                              negative_prompt, 
                              travel_prompt, 
                              workflow_filename=None, 
                              pipeline_type = PipelineType.BASE,
                              denoise_first_pass=1.0,
                              enable_lipsync: bool = False,
                              enable_cinematic: bool = False,
                              disable_lcm: bool = False,
                              global_ipa_image_filename: Optional[str] = None,
                              global_ipa_strength: float = 1.0,
                              frame_skip: Optional[int] = None,
                              depth_video_filename: str = None,
                              outline_video_filename: str = None,
                              normals_video_filename: str = None,
                            ) -> Path:
    styles_directory = Path(V2V_WORKFLOWS_DIRECTORY) / "styles"
    workflow_directory = Path(V2V_WORKFLOWS_DIRECTORY) / "workflows"
    mappings_directory = Path(V2V_WORKFLOWS_DIRECTORY) / "mappings"

    style_filename = style_to_filename[style_name]
    style_json = json.load(open(styles_directory / style_filename))

    workflow_json_filename = style_json["workflow_api_name"]
    mapping_json_filename = "mapping_one.json"

    if workflow_filename:
        workflow_json_filename = workflow_filename

    workflow_json = json.load(open(workflow_directory / workflow_json_filename))
    mapping_json = json.load(open(mappings_directory / mapping_json_filename))

    json_mods = {key: value for key, value in style_json["modifications"].items() if not key.startswith("cn_")}

    jsonpath_mods = get_jsonpath_mods(
        json_mods,
        mapping_json,
        pos_in=positive_prompt,
        neg_in=negative_prompt,
        travel_in=travel_prompt,
        pipeline_type=pipeline_type,
        denoise_first_pass=denoise_first_pass,
        enable_lipsync=enable_lipsync,
        enable_cinematic=enable_cinematic,
        disable_lcm=disable_lcm,
        global_ipa_image_filename=global_ipa_image_filename,
        global_ipa_strength=global_ipa_strength,
        frame_skip=frame_skip,
        depth_video_filename=depth_video_filename,
        outline_video_filename=outline_video_filename,
        normals_video_filename=normals_video_filename
    )

    print(*jsonpath_mods.items(), sep="\n")

    json_prompt = apply_jsonpath_mods(jsonpath_mods, workflow_json)

    p = {"prompt": json_prompt, "client_id": client_id}
    return p

async def cmd_runner(cmd: str):
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stderr=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE
    )

    stdout, stderr = await proc.communicate()

    print(f'[{cmd!r} exited with {proc.returncode}]')
    if stdout:
        print(f'[stdout]\n{stdout.decode()}')
    if stderr:
        print(f'[stderr]\n{stderr.decode()}')



def get_jsonpath_mods(style_mods: Dict[str, Any],
                      mapping_json: Dict[str, Any], 
                      pos_in: Optional[str] = None,
                      neg_in: Optional[str] = None, 
                      travel_in: Optional[str] = None, 
                      pipeline_type = PipelineType.BASE,
                      denoise_first_pass=1.0,
                      enable_lipsync: bool = False,
                      enable_cinematic: bool = False,
                      disable_lcm: bool = False,
                      global_ipa_image_filename: Optional[str] = None,
                      global_ipa_strength: float = 1.0,
                      frame_skip: Optional[int] = None,
                      depth_video_filename: str = None,
                      outline_video_filename: str = None,
                      normals_video_filename: str = None
                      ) -> Dict[str, Any]:

    modifications = {}
    new_mod_json = style_mods.copy()
    # Process "loras" differently
    loras = new_mod_json.get("loras", [])
    if len(loras) > 8:
        raise ValueError("Too many loras, max is 8")

    for index, lora in enumerate(loras):
        if "name" in lora and "strength" in lora:
            lora_strength = lora["strength"]
            if pipeline_type == PipelineType.FACE_DETAILER:
                lora_strength = 0.5
            new_mod_json[f"lora_{index + 1}_strength"] = lora_strength
            new_mod_json[f"lora_{index + 1}_name"] = lora["name"]

    # Exclude "loras" from further processing
    new_mod_json.pop("loras", None)

    # Handling positive and negative prompts
    if pos_in:
        new_mod_json["positive_prompt"] = f"{pos_in}, {new_mod_json.get('positive_prompt', '')},"
    if neg_in:
        new_mod_json["negative_prompt"] = f"{neg_in}, {new_mod_json.get('negative_prompt', '')},"

    if pipeline_type in [PipelineType.IPA, PipelineType.BASE, PipelineType.PREVIEW]:
        new_mod_json["denoise_first_pass"] = denoise_first_pass

    if enable_lipsync:
        new_mod_json["cn_lips_strength"] = 1.2

    if frame_skip:
        new_mod_json["every_nth_frame"] = frame_skip


    for key, value in new_mod_json.items():
        mapping_key = f"$.{key}"
        jsonpath_expr = parse(mapping_key)

        # Finding matches using jsonpath_ng
        matches = [match.value for match in jsonpath_expr.find(mapping_json)]
        if matches:
            mapping_value = matches[0]
            modifications[mapping_value] = value
        else:
            print(f"No mapping found for key '{key}'")

    # TODO(bt,2024-05-29): Move this to the mappings so we don't have to maintain the jsonpath
    # This is only relevant for the main workflow (not face fixer or upscaler)
    if disable_lcm:
        modifications["$.536.inputs.boolean_number"] = 0

    # TODO(bt,2024-06-22): Move this to the mappings so we don't have to maintain the jsonpath
    if global_ipa_image_filename:
        modifications["$.3723.inputs.image_path"] = global_ipa_image_filename
        modifications["$.900.inputs.value"] = global_ipa_strength

    # TODO(bt,2024-07-06): Move this to the mappings so we don't have to maintain the json path
    if travel_in:
        modifications["$.509.inputs.text"] = travel_in

    if enable_cinematic:
        modifications["$.2047.inputs.value"] = 576
        modifications["$.2048.inputs.value"] = 1024

    # TODO The 3 Mappings for depth outlines and normals apparently uses an array.
    if pipeline_type in [PipelineType.IPA, PipelineType.BASE, PipelineType.PREVIEW]:
        if depth_video_filename != None and outline_video_filename != None and normals_video_filename !=None:
            modifications["$.3731.inputs.text"] = depth_video_filename
            modifications["$.4120.inputs.text"] = normals_video_filename
            modifications["$.4122.inputs.text"] = outline_video_filename
    return modifications


def apply_jsonpath_mods(jsonpath_mods: Dict[str, Any], workflow_json: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in jsonpath_mods.items():
        # jsonpath-ng expects numbers to be wrapped in quotes
        parts = key.split(".")
        new_parts = [f"'{part}'" if part.isdigit() else part for part in parts]
        key = ".".join(new_parts)

        jsonpath_expr = parse(key)
        matches = jsonpath_expr.find(workflow_json)

        if matches:
            parent = matches[0].context.value

            if isinstance(parent, dict):
                last_part = new_parts[-1].strip("'")
                parent[last_part] = value

            elif isinstance(parent, list):
                index = int(new_parts[-1].strip("'"))
                if 0 <= index < len(parent):
                    parent[index] = value
                else:
                    print(f"Index {index} is out of bounds for key '{key}'")
        else:
            print(f"No mapping found for key '{key}'")

    return workflow_json


def validate_style_name(style):
    if style not in style_to_filename:
        print(f"Invalid style: {args.style}")
        print(f"Valid styles: {', '.join(style_to_filename.keys())}")
        exit(1)

if __name__ == "__main__":
    os.setpgrp()
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {e}")
    finally:
        os.killpg(0, signal.SIGKILL) # kill all processes in my group
