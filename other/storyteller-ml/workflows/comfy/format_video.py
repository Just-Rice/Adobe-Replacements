import argparse
import os
import subprocess
import json
import sys
from pathlib import Path


def get_video_resolution(input_video):
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'json',
        input_video
    ]

    print(f"Command: {command}", flush=True)

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    video_info = json.loads(result.stdout)

    width = video_info['streams'][0]['width']
    height = video_info['streams'][0]['height']

    return width, height

def calculate_new_resolution(width, height, max_resolution):
    aspect_ratio = width / height
    max_width, max_height = map(int, max_resolution.split('x'))

    if width > max_width or height > max_height:
        if (max_width / aspect_ratio) <= max_height:
            new_width = max_width
            new_height = round(new_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = round(new_height * aspect_ratio)
    else:
        new_width, new_height = width, height

    new_width = 8 * round(new_width / 8)
    new_height = 8 * round(new_height / 8)

    return new_width, new_height

def format_timestamp(ms):
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    seconds, ms = divmod(ms, 1000)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int(ms):03d}"

def process_video(input_video, max_resolution, start_ms, end_ms, framerate, output_video = None):
    width, height = get_video_resolution(input_video)
    print(f"Original resolution: {width}x{height}")

    # new_width, new_height = calculate_new_resolution(width, height, max_resolution)
    # print(f"New resolution: {new_width}x{new_height}")

    start_time = format_timestamp(start_ms)
    duration_ms = end_ms - start_ms
    duration = format_timestamp(duration_ms)

    if not output_video:
        # NB: This is salt's old implicit output video naming
        input_vid_dir = Path(input_video).absolute().parent
        output_video = input_vid_dir.joinpath(f"input.mp4")

    command = [
        'ffmpeg',
        '-y',
        '-i', input_video,
        '-ss', start_time,
        '-t', duration,
        # '-vf', f'scale={new_width}:{new_height}',
        '-r', str(framerate),
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
        '-c:a', 'aac', '-b:a', '128k',
        output_video
    ]

    print(f"Command: {command}", flush=True)

    subprocess.run(command, check=True)
    print(f"Video processed successfully. Output saved as {output_video}")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        # print("Usage: python script.py <input_video> <max_resolution> <start_ms> <end_ms> <framerate>")
        print("Usage: python script.py <input_video> <start_ms> <end_ms> <framerate>")
        sys.exit(1)

    # Old Salt Args (replace this mess)
    input_video = sys.argv[1]
    output_video = None
    # max_resolution = sys.argv[2]
    max_resolution = None
    start_ms = int(sys.argv[2])
    end_ms = int(sys.argv[3])
    framerate = float(sys.argv[4])

    remaining_args = sys.argv[5:]
    print('remaining args', remaining_args)

    parser = argparse.ArgumentParser(description='Format Video')
    # TODO: Output and input are optional for now until we migrate away from salt's arguments. 
    parser.add_argument('--input', type=str, help='Input Video Filename', required=False)
    parser.add_argument('--output', type=str, help='Output Video Filename', required=False)

    args = parser.parse_args(remaining_args)

    if args.input:
        input_video = args.input
    
    if args.output:
        output_video = args.output

    process_video(input_video, max_resolution, start_ms, end_ms, framerate, output_video=output_video)
