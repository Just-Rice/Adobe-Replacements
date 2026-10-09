import signal
import subprocess
import time

import shutil
from pathlib import Path
import uuid
import os
import json
import random
from typing import Dict, Any, Optional
import mimetypes
from moviepy.editor import ImageClip
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, Response
import requests
from jsonpath_ng import jsonpath, parse
from fastapi.middleware.cors import CORSMiddleware
import urllib
import websockets

# CORS header for which websites can import this content in a frame.
# This *technically* shouldn't be required, but the XHR requests are
# being made from a controlled frame.
#
# https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Security-Policy/frame-ancestors
#
# NB(bt,2024-03-30): The *.netlify.app is dangerous.
CORS_WHITELIST_ORIGINS = [
    "https://storyteller.ai",
    "https://render.storyteller.ai",
    "https://studio.storyteller.ai",
    "https://studio-testing.studio.storyteller.ai",
    "https://studio-staging.studio.storyteller.ai",
    "https://storytellerstudio.netlify.app",
    "https://branch-name--storytellerstudio.netlify.app",
    "https://deploy-preview-123--storytellerstudio.netlify.app",
    "https://pipeline-gottagofast.netlify.app",
    "https://branch-whatever--pipeline-gottagofast.netlify.app",
    "https://fakeyou.com",
    "https://*.fakeyou.com",
    "https://*.storyteller.ai",
    "https://*.netlify.app",
    "http://*.fakeyou.com:7000",
    "http://*.fakeyou.com:7001",
    "http://*.fakeyou.com:7002",
    "http://*.storyteller.ai:7000",
    "http://*.storyteller.ai:7001",
    "http://*.storyteller.ai:7002",
]


# TODO(kasisnu): Remove ugly globals
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_WHITELIST_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SERVER_IP = os.environ.get("SERVER_IP", "127.0.0.1:8188")
TIMEOUT_SECONDS = os.environ.get("TIMEOUT_SECONDS", 1000)
PROMPT_ENDPOINT = f"http://{SERVER_IP}/prompt"
V2V_WORKFLOWS_DIRECTORY = os.environ.get("V2V_WORKFLOWS_DIRECTORY",
                                         "/home/kasisnu/go/src/github.com/storytold/V2V-StylesList/V2VWorkflows/")
COMFY_ROOT_DIRECTORY = os.environ.get("COMFY_ROOT_DIRECTORY",
                                      "/home/kasisnu/go/src/github.com/comfyanonymous/ComfyUI")
SERVER_START_TIMEOUT = int(os.environ.get("SERVER_START_TIMEOUT", 60))
STYLE_LOCK = os.environ.get("STYLE_LOCK", None)
PREVIEW_WORKFLOW_NAME = os.environ.get("PREVIEW_WORKFLOW_NAME", 'yae_vid2vid_InstantPreview_14-05_API.json')
MAPPING_NAME = os.environ.get("MAPPING_NAME", "mapping_two.json")
AUTH_URL = os.environ.get("AUTH_URL", "https://api.fakeyou.com/v1/session")

# websocket.enableTrace(True)
client_id = str(uuid.uuid4())


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

allowed_keys = ["style", "positive_prompt", "negative_prompt"]



def can_use_vst(cookies):
    #todo(kasisnu): this didn't seem to work right away, might need digging into
    return True # FOR NOW
    """Check if the user is authenticated and if the 'video_style_transfer' feature flag is enabled."""
    try:
        response = requests.get(AUTH_URL, cookies=cookies)
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return False

    if response.status_code == 200:
        try:
            response_json = response.json()
        except ValueError:
            print("Invalid JSON response")
            return False

        if response_json.get("success") and response_json.get("logged_in"):
            username = response_json["user"]["core_info"]["username"]
            print(f"User is logged in: {username}")
            feature_flags = response_json["user"].get("maybe_feature_flags", {})
            if "video_style_transfer" in feature_flags:
                print("VST feature flag is found")
                return True
            else:
                print("VST feature flag not found")
        else:
            print("User is not logged in or session is invalid")
    else:
        print(f"Failed to authenticate ({response.status_code}): {response.text}")

    return False


def validate_request(request):
    try:
        request = json.loads(request)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON for 'request'")
    # check for style key
    if "style" not in request:
        raise HTTPException(status_code=400, detail="Missing 'style' key in 'request'")
    style = request["style"]
    if style not in style_to_filename:
        raise HTTPException(status_code=400,
                            detail=f"Invalid 'style' key in 'request'. Valid styles: {', '.join(style_to_filename.keys())}")
    # check for other keys
    for key in request:
        if key not in allowed_keys:
            raise HTTPException(status_code=400,
                                detail=f"Invalid key '{key}' in 'request'. Valid keys: {', '.join(allowed_keys)}")
    return request


def terminate_process(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def signal_handler(signum, frame):
    print("Signal received, shutting down.")
    terminate_process(comfyui_process)
    print("ComfyUI server terminated")
    exit(0)

@app.get('/health')
def health():
    return "All good"


@app.post("/preview/")
async def preview_request(request_obj: Request, request: str = Form(...), input_file: UploadFile = File(...)):
    cookies = request_obj.cookies
    if not can_use_vst(cookies):
        raise HTTPException(status_code=401, detail="No Video Style Transfer feature flag found")

    request = validate_request(request)

    if STYLE_LOCK:
        if request["style"] != STYLE_LOCK:
            raise HTTPException(status_code=400, detail=f"This endpoint is for '{STYLE_LOCK}' style only")
    try:
        request_style = request["style"]
        style_filename = style_to_filename[request["style"]]
        positive_prompt = request.get("positive_prompt", None)
        negative_prompt = request.get("negative_prompt", None)
        mime_type = input_file.content_type
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(input_file.filename)
        input_filename_ext = ''
        if 'video' in mime_type:
            input_filename_ext = ".mp4"
        elif 'image' in mime_type:
            input_filename_ext = mimetypes.guess_extension(mime_type)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        temp_filename = f"temp_files/{uuid.uuid4()}{input_filename_ext}"
        Path(temp_filename).parent.mkdir(parents=True, exist_ok=True)
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(input_file.file, buffer)
        if 'image' in mime_type:
            clip = ImageClip(temp_filename)
            fps = 24
            clip_duration = 1 / fps
            clip = clip.set_duration(clip_duration)
            input_filename = f"temp_files/{uuid.uuid4()}.mp4"
            clip.write_videofile(input_filename, fps=fps)
        else:
            input_filename = temp_filename
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Process video (or image converted to video) and generate preview
    preview_image_path = await process_video(request_style, input_filename, style_filename, positive_prompt, negative_prompt)

    return Response(preview_image_path, media_type='image/jpeg')


async def process_video(request_style: str, video_path: str, style_filename, positive_prompt=None, negative_prompt=None) -> Path:
    # generate random seed (otherwise video will cache completely and not generate a new output)
    styles_directory = Path(V2V_WORKFLOWS_DIRECTORY) / "styles"
    workflow_directory = Path(V2V_WORKFLOWS_DIRECTORY) / "workflows"
    mappings_directory = Path(V2V_WORKFLOWS_DIRECTORY) / "mappings"

    style_json = json.load(open(styles_directory / style_filename))
    workflow_json = json.load(open(workflow_directory / PREVIEW_WORKFLOW_NAME))
    mapping_json = json.load(open(mappings_directory / MAPPING_NAME))

    json_mods = style_json["modifications"]
    json_mods['cn_soft_edge_strength'] = 1.0
    jsonpath_mods = get_jsonpath_mods(json_mods, mapping_json, pos_in=positive_prompt, neg_in=negative_prompt)
    print(*jsonpath_mods.items(), sep="\n")

    json_prompt = apply_jsonpath_mods(jsonpath_mods, workflow_json)

    # generate random seed (otherwise video will cache completely and not generate a new output)
    try:
        json_prompt["173"]["inputs"]["seed"] = random.randint(0, 1000000)
    except KeyError:
        pass

    input_video_path = Path(COMFY_ROOT_DIRECTORY) / "input/input.mp4"
    shutil.move(video_path, input_video_path)
    p = {"prompt": json_prompt, "client_id": client_id, "extra_data": {"style": request_style}}
    data = json.dumps(p).encode('utf-8')
    execution_start_time = time.perf_counter()
    response = requests.post(PROMPT_ENDPOINT, data=data)
    if response.status_code == 200:
        print("Job enqueued successfully")
    else:
        raise HTTPException(status_code=500, detail=f"Failed to queue job")
    prompt_id = response.json()["prompt_id"]
    async with websockets.connect("ws://{}/ws?clientId={}".format(SERVER_IP, client_id)) as ws:
        images = await get_images(ws, prompt_id, '1517')
        current_time = time.perf_counter()
        execution_time = current_time - execution_start_time
        print('request executed in {:.2f} seconds'.format(execution_time))
        return images['1517'][0]



def get_jsonpath_mods(style_mods: Dict[str, Any], mapping_json: Dict[str, Any], pos_in: Optional[str] = None,
                      neg_in: Optional[str] = None) -> Dict[str, Any]:
    modifications = {}
    new_mod_json = style_mods.copy()
    # Process "loras" differently
    loras = new_mod_json.get("loras", [])
    if len(loras) > 8:
        raise ValueError("Too many loras, max is 8")

    for index, lora in enumerate(loras):
        if "name" in lora and "strength" in lora:
            new_mod_json[f"lora_{index + 1}_strength"] = lora["strength"]
            new_mod_json[f"lora_{index + 1}_name"] = lora["name"]

    # Exclude "loras" from further processing
    new_mod_json.pop("loras", None)

    # Handling positive and negative prompts
    if pos_in:
        new_mod_json["positive_prompt"] = f"{pos_in}, {new_mod_json.get('positive_prompt', '')},"
    if neg_in:
        new_mod_json["negative_prompt"] = f"{neg_in}, {new_mod_json.get('negative_prompt', '')},"

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


def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(SERVER_IP, url_values)) as response:
        return response.read()



def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(SERVER_IP, prompt_id)) as response:
        return json.loads(response.read())

def get_queue(server_ip):
    with urllib.request.urlopen("http://{}/queue".format(server_ip)) as response:
        return json.loads(response.read())

async def get_images(ws, prompt_id, filter_node_id=None):
    output_images = {}
    while True:
        out = await ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break #Execution is done
        else:
            continue #previews are binary data

    history = get_history(prompt_id)[prompt_id]
    for o in history['outputs']:
        for node_id in history['outputs']:
            if filter_node_id is not None and node_id != filter_node_id:
                continue
            node_output = history['outputs'][node_id]
            images_output = []
            if 'images' in node_output:
                for image in node_output['images']:
                    image_data = get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
            output_images[node_id] = images_output

    return output_images



if __name__ == "__main__":
    import uvicorn
    # install ComfyUI
    # os.system("python3 install.py")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    comfyui_process = subprocess.Popen(["../venv/bin/python", "main.py", "--highvram", "--disable-metadata"], cwd=COMFY_ROOT_DIRECTORY)


    # block until comfyUI server is up
    start_time = time.time()
    started = False
    while time.time() - start_time < SERVER_START_TIMEOUT:
        try:
            response = requests.get(PROMPT_ENDPOINT, timeout=5)
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
    print("ComfyUI server running, starting FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # cleanup
    terminate_process(comfyui_process)
