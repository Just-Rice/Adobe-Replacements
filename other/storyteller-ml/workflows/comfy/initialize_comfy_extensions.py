import os
import sys
import asyncio
import threading
# Init all the nodes

current_script_dir = os.path.dirname(os.path.realpath(__file__))
comfy_path = os.path.dirname(os.path.abspath(__file__))
comfy_path = os.path.join(comfy_path, 'ComfyUI')

# change to ComfyUI directory
os.chdir(comfy_path)

# add ComfyUI to path
sys.path.append(comfy_path)

from server import PromptServer
from execution import PromptQueue
from main import init_custom_nodes, cuda_malloc_warning, prompt_worker, execute_prestartup_script
import folder_paths

execute_prestartup_script()

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


if __name__ == "__main__":
    asyncio.run(run())
