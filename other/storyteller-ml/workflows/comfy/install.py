
import subprocess
import json
import os
from pathlib import Path
from urllib.parse import urlparse
from multiprocessing import Pool
import argparse
from functools import partial
from itertools import repeat

parser = argparse.ArgumentParser(description='Install ComfyUI and its dependencies')
parser.add_argument('--source', type=str, help='The source directory of the dependencies', required=False)
parser.add_argument('--dependencies-json', type=str, help='The path to the json file containing the dependencies', required=False)
parser.add_argument('--repos-json', type=str, help='The path to the json file containing the repositories', required=False)
parser.add_argument('--repos-checkout-latest', action='store_true', help='Checkout the latest commit of the repositories', default=False)

parser.add_argument('--concurrency', type=int, help='The number of concurrent processes to use', default=8)

parser.add_argument('--comfy-ui-url', type=str, help='The url of the ComfyUI repository', required=False)
parser.add_argument('--comfy-ui-commit', type=str, help='The commit of the ComfyUI repository', required=False)


args = parser.parse_args()


# set cwd to script location
os.chdir(os.path.dirname(os.path.realpath(__file__)))

venv_python = os.path.join('ComfyUI', 'venv', 'Scripts' if os.name == 'nt' else 'bin', 'python')


def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Print stdout live
    for line in iter(process.stdout.readline, ''):
        print(line, end='')

    # Wait for the process to terminate and get the exit code
    process.wait()

    # Check if the process exited with an error
    if process.returncode != 0:
        # Print any errors from stderr
        for line in iter(process.stderr.readline, ''):
            print(line, end='')
        print(f"Error executing command: {command}")

def setup_repo(repo_info, checkout_latest=False):
    url, commit, folder = repo_info['url'], repo_info['commit'], repo_info['folder']
    # save the current working directory
    original_cwd = os.getcwd()
    try:
        run_command(f"git clone {url} {folder}")
        if not checkout_latest:
            os.chdir(folder)
            run_command(f"git checkout {commit}")
    finally:
        os.chdir(original_cwd)


def setup_repos(json_file_path, concurrency=4, checkout_latest=False):
    with open(json_file_path, 'r') as file:
        repos = json.load(file)
    with Pool(concurrency) as pool:
        pool.starmap(setup_repo, zip(repos, repeat(checkout_latest)))

def setup_comfy_ui(comfyui_info):
    setup_repo(comfyui_info)

def setup_venv_and_install_requirements():
    run_command("python3 -m venv ComfyUI/venv")
    print('=============# Installing ComfyUI Requirements #=============')
    run_command(f"{venv_python} -m pip install -r ComfyUI/requirements.txt")
    print('=============# Installing extension missing requirements #=============')
    run_command(f"{venv_python} -m pip install -r required_requirements.txt")


def initialize_comfyui():
    run_command(f"{venv_python} initialize_comfy_extensions.py")

def install_dependency(dep, source_dir):
    location = dep['location']
    url = dep['url']

    # separate the pathname from the url host
    source_pathname = urlparse(url).path.lstrip('/')

    # Convert the location to a Path object and create parent directories if they don't exist
    path = Path(location)
    path.parent.mkdir(parents=True, exist_ok=True)

    source_path = Path() / source_dir /  source_pathname
    # Copy the dependency from source dir to the location
    print(f"Copying checkpoint from {source_path} to {location}")
    run_command(f"cp {source_path} {location}")

def install_dependencies(json_file_path, source_dir, concurrency=4):
    # Load the dependency JSON
    with open(json_file_path, 'r') as file:
        dependencies = json.load(file)

    with Pool(concurrency) as pool:
        pool.starmap(install_dependency, [(dep, source_dir) for dep in dependencies])

def main():
    if args.comfy_ui_url and args.comfy_ui_commit:
        comfyui_info = {
            "url": args.comfy_ui_url,
            "commit": args.comfy_ui_commit,
            "folder": "ComfyUI",
        }
        setup_comfy_ui(comfyui_info)

    if args.repos_json:
        setup_repos(args.repos_json, args.concurrency, args.repos_checkout_latest)
    else:
        print("Skipping repository setup as repos json is not provided")


    #initialize_comfyui()
    if args.source and args.dependencies_json:
        install_dependencies(args.dependencies_json, args.source, args.concurrency)
    else:
        print("Skipping dependency installation as source and dependencies json are not provided")


if __name__ == "__main__":
    main()
    open('ALREADY_SETUP.flag', 'w').close()
