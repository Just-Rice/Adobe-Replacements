import requests
import argparse
import os
import json
import time
from typing import Dict, Any, Optional
from jsonpath_ng import jsonpath, parse
import shutil
from pathlib import Path
from enum import IntEnum, auto
import os
import subprocess
import signal
import urllib
import websockets
import asyncio
import uuid
client_id = str(uuid.uuid4())

SERVER_IP = os.environ.get("SERVER_IP", "127.0.0.1:8188")
SERVER_START_TIMEOUT = int(os.environ.get("SERVER_START_TIMEOUT", 60))

TIMEOUT_SECONDS = int(os.environ.get("TIMEOUT_SECONDS", 180))
PROMPT_ENDPOINT = f"http://{SERVER_IP}/prompt"
V2V_WORKFLOWS_DIRECTORY = os.environ.get("V2V_WORKFLOWS_DIRECTORY",
                                         "/workflow_configs")


COMFY_ROOT_DIRECTORY = os.environ.get("COMFY_ROOT_DIRECTORY",
                                      "/app/ComfyUI")


BASE_LIVE_PORTRAIT_WORKFLOW = os.environ.get("BASE_LIVE_PORTRAIT_WORKFLOW", "25-07-2024/yae_LivingPortrait(Cuda)_25-07_API.json")

class PipelineType(IntEnum):
    BASE = auto()


def ensure_input_directory_empty():
    input_directory = Path(COMFY_ROOT_DIRECTORY) / "input"
    for file in input_directory.iterdir():
        if file.is_file():
            file.unlink()

def ensure_output_directory_empty():
    output_directory = Path(COMFY_ROOT_DIRECTORY) / "output" / "vid2vid"
    for file in output_directory.iterdir():
        if file.is_file():
            print('deleteing file: ' + str(file) )
            file.unlink()

def ensure_pipeline_input_present(source_path, input_name=None):
    left = Path(source_path)
    left_filename = left.name
    right_filename = left_filename
    if input_name is not None:
        right_filename = input_name
    right = Path(COMFY_ROOT_DIRECTORY) / "input" / right_filename
    shutil.copy(left, right)

def primary_output_directory():
    return Path(COMFY_ROOT_DIRECTORY) / "output" / "vid2vid"

def expected_primary_output_name(primary_output_name="LivePortrait_00001.mp4"):
    directory = primary_output_directory()
    output = directory / primary_output_name
    return output

def ensure_output_present_at(new_filename, primary_output_name="LivePortrait_00001.mp4"):
    # Comfy will create a copy of the video that includes audio, but only if the driver video had audio.
    output = expected_primary_output_name(primary_output_name)
    output_with_audio = primary_output_directory() / "LivePortrait_00001-audio.mp4"
    if os.path.exists(output_with_audio):
        output = output_with_audio
    shutil.copy(output, new_filename)

def parse_args():
    parser = argparse.ArgumentParser(description='Run Comfy')
    parser.add_argument('--portrait-media-filename', type=str, help='path of mp4 depth from engine for preprocessing', required=True)
    parser.add_argument('--driver-media-filename', type=str, help='path of mp4 outline from engine for preprocessing', required=True)
    parser.add_argument('--input-is-image', action='store_true', help='input is image', default=False)
    parser.add_argument('--skip-comfy-startup', action='store_true', help='controls if this script tries to start comfy', default=True)
    parser.add_argument('--tmpdir', type=str, help='path of tmpdir for preprocessing', required=False)
    parser.add_argument('--output-filename', type=str, help='path of output filename', required=True)


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
    queued_jobs_count = len(queue_json.get("queue_pending", []))
    running_jobs_count = len(queue_json.get("queue_running", []))
    return queued_jobs_count + running_jobs_count    

def queued_count_is_zero(server_address):
    count = get_queued_jobs_count(server_address)
    print(f"Number of queued jobs detected: {count}")
    if count is None:
        return False
    return count == 0

def trigger_comfy_restart(restart_trigger_file):
    print(f"Touching {restart_trigger_file}")
    with open(restart_trigger_file, "w") as f:
        f.write(time.ctime())
    print("Restart triggered")

# KS: we're making the retries configurable but we should not increase the retries for all workflows without confirming we'd still meet slas. Some jobs are better to just fail. Retries compound over inference-job attempt_counts, comfy startup times, and actual inference times.
def ensure_server_is_healthy(server_address, restart_trigger_file=None, retries=2):
    wait_for_comfy_startup(server_address)
    if queued_count_is_zero(server_address):
        print("Server is healthy")
        return
    if restart_trigger_file is None:
        print("No restart trigger file provided. Exiting.")
        exit(1)

    for _ in range(retries):
        trigger_comfy_restart(restart_trigger_file)
        time.sleep(10)
        wait_for_comfy_startup(server_address)
        number_of_queued_jobs = get_queued_jobs_count(server_address)
        if queued_count_is_zero(server_address):
            print("Server is healthy")
            return
        print(f"Server is not healthy. Number of queued jobs: {number_of_queued_jobs}")
        print("Server is not healthy after restart. Retrying restart")

    wait_for_comfy_startup(server_address, timeout=5, terminate_on_failure=True)


def interrupt_comfy_queue(server_address):
    requests.post(f"http://{server_address}/interrupt")
    print("Queue interrupted successfully.")

def wait_for_comfy_startup(server_address, timeout=SERVER_START_TIMEOUT, terminate_on_failure=False):
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
        if terminate_on_failure:
            print("Exiting.")
            exit(1)


async def main():
    outer_retries = 1
    restart_trigger_file = args.server_restart_trigger_file

    pipeline_type = PipelineType.BASE
    workflow_filename = BASE_LIVE_PORTRAIT_WORKFLOW
    input_media_filename = args.portrait_media_filename
    input_driver_filename = args.driver_media_filename
    input_is_image = args.input_is_image

    for attempt_idx in range(outer_retries):
        ensure_server_is_healthy(SERVER_IP, restart_trigger_file)

        ensure_input_directory_empty()
        ensure_output_directory_empty()
        input_filename = "input.mp4"

        if input_media_filename:
            if input_is_image:
                ensure_pipeline_input_present(input_media_filename)
                input_filename = Path(input_media_filename).name
            else:
                ensure_pipeline_input_present(input_media_filename, "input.mp4")

        if input_driver_filename:
            ensure_pipeline_input_present(input_driver_filename, "driver.mp4")
        print("Generating prompt json using workflow: " + workflow_filename)
        prompt = generate_prompt_for_style(
            workflow_filename=workflow_filename,
            input_media_filename="input/" + input_filename,
            input_driver_filename="input/driver.mp4",
            input_is_image=input_is_image,
        )
        time_before = time.perf_counter()
        print("running pipeline" + str(pipeline_type))
        await enqueue_prompt_and_wait(prompt)
        time_after = time.perf_counter()
        primary_output_name = "LivePortrait_00001.mp4"
        if expected_primary_output_name(primary_output_name).exists():
            ensure_output_present_at(args.output_filename)
            print(f"Comfy pipeline ran successfully in {time_after - time_before:.2f} seconds")
            # We've successfully run the pipeline, so we can exit the loop
            return
        else:
            if restart_trigger_file is None:
                print("No restart trigger file provided. Exiting.")
                exit(1)
            if attempt_idx < outer_retries - 1:
                trigger_comfy_restart(restart_trigger_file)
                time.sleep(10)
    # If we're here, we've failed to run the pipeline twice. We've tried a server restart, and we're out of retries. Maybe this was a bad input, or maybe the server is down. Either way, it's okay to give up now. We meant well.
    exit(1)

def classify_error_message(message_data):
    if message_data['node_type'] == 'LivePortraitCropper':
        print("Error in LivePortraitCropper")

async def smart_wait(ws, prompt_id):
    while True:
        out = await ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'execution_error':
                print(message)
                classify_error_message(message['data'])
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break #Execution is done
        else:
            continue #previews are binary data


async def enqueue_prompt_and_wait(prompt):
    data = json.dumps(prompt).encode('utf-8')
    # Send HTTP post request
    response = requests.post(PROMPT_ENDPOINT, data=data)
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

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(SERVER_IP, prompt_id)) as response:
        return json.loads(response.read())

def generate_prompt_for_style(
                              workflow_filename: str,
                              input_media_filename: Optional[str] = None,
                                input_driver_filename: Optional[str] = None,
                                input_is_image: Optional[bool] = False,
                            ):
    workflow_directory = Path(V2V_WORKFLOWS_DIRECTORY) / "workflows"


    if workflow_filename:
        workflow_json_filename = workflow_filename

    workflow_json = json.load(open(workflow_directory / workflow_json_filename))


    jsonpath_mods = {
        "$.25.inputs.video": input_driver_filename
    }

    if input_is_image:
        jsonpath_mods["$.43.inputs.image_path"] = input_media_filename
        jsonpath_mods["$.26.inputs.video"] = input_driver_filename
        jsonpath_mods["$.46.inputs.boolean"] = 0
    else:
        jsonpath_mods["$.26.inputs.video"] = input_media_filename
        jsonpath_mods["$.46.inputs.boolean"] = 1

    print(*jsonpath_mods.items(), sep="\n")

    json_prompt = apply_jsonpath_mods(jsonpath_mods, workflow_json)

    p = {"prompt": json_prompt, "client_id": client_id}
    return p

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


if __name__ == "__main__":
    os.setpgrp()
    try:
        if not args.skip_comfy_startup:
            comfyui_process = subprocess.Popen(["../venv/bin/python", "main.py", "--highvram", "--disable-metadata"], cwd=COMFY_ROOT_DIRECTORY)
        asyncio.run(main())
    finally:
        if not args.skip_comfy_startup:
            os.killpg(0, signal.SIGKILL) # kill all processes in my group
