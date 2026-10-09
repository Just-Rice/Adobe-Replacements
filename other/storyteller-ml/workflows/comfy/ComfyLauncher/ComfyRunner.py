import asyncio
import json
import os
import sys
import threading
import signal

import aiohttp

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run Comfy')
    parser.add_argument('--prompt', type=str, help='location of the prompt json', required=True)
    return parser.parse_args()

# current_script_dir = os.path.dirname(os.path.realpath(__file__))
# file_path = os.path.join(current_script_dir, 'prompt', 'prompt.json')

args = parse_args()

file_path = args.prompt

print(f'Loading prompt json from path: {file_path}')

prompt = json.load(open(file_path))

comfy_path = os.path.dirname(os.path.abspath(__file__))
comfy_path = os.path.join(comfy_path, '..', 'ComfyUI')

# change to ComfyUI directory
os.chdir(comfy_path)

# add ComfyUI to path
sys.path.append(comfy_path)


from server import PromptServer
from execution import PromptQueue
from main import init_custom_nodes, cuda_malloc_warning, prompt_worker
import folder_paths


async def run():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = PromptServer(loop)
    q = PromptQueue(server)

    init_custom_nodes()

    cuda_malloc_warning()

    server.add_routes()

    threading.Thread(target=prompt_worker, daemon=True, args=(q, server,)).start()

    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.get_output_directory(), "checkpoints"))
    folder_paths.add_model_folder_path("clip", os.path.join(folder_paths.get_output_directory(), "clip"))
    folder_paths.add_model_folder_path("vae", os.path.join(folder_paths.get_output_directory(), "vae"))

    await server.start('127.0.0.1', 8188)

    async def send_prompt_async():
        # Load prompt
        p = {"prompt": prompt}
        data = json.dumps(p).encode('utf-8')
        # Send HTTP post request
        endpoint = "http://127.0.0.1:8188/prompt"
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, data=data) as resp:
                print(
                    await resp.text())


    await send_prompt_async()

    while server.prompt_queue.currently_running:
        await asyncio.sleep(0.05)

if __name__ == "__main__":
    asyncio.run(run())
