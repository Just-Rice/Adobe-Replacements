from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import tempfile
import shutil
import datetime
import pickle
import magic
from PIL import Image

"""
Current worker filesystem architecture:

/tmp
/tmp/models
/tmp/models/wav2lip_gan.pth
/tmp/.tmppd1Kvq
/tmp/.tmppd1Kvq/input_audio_file
/tmp/templates_images
/tmp/templates_images/elon-musk.jpg
/tmp/templates_videos
/tmp/templates_videos/ben-shapiro-baby.mp4
/tmp/templates_videos/ben-shapiro-baby.mp4.faces
/tmp/templates_videos/dr-phil-bubble.mp4
/tmp/tmp7um0yxjo
/tmp/tmp7um0yxjo/padded_audio.wav
/tmp/tmp7um0yxjo/temp.wav
/tmp/assets
/tmp/assets/vocodes-short-end-bump.mp4

"""

if 'PRINT_ENV' in os.environ and os.environ['PRINT_ENV']:
  for k, v in os.environ.items():
    print('{}={}'.format(k, v))

print('========================================')
print('Python interpreter', sys.executable)
print('PyTorch version', torch.__version__)
print('CUDA Available?', torch.cuda.is_available())
print('CUDA Device count', torch.cuda.device_count())
print('========================================', flush=True)

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str,
                    help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--image_or_video_filename', type=str,
                    help='Filepath of video/image that contains faces to use', required=True)

parser.add_argument('--output_cached_faces_filename', type=str,
                    help='Output filename for the cached faces file', required=True)

parser.add_argument('--output_metadata_filename', type=str,
                    help='Output filename for the JSON containing width, height, etc.', required=True)

# NB: Not needed for processing the upload file for faces:
#
# parser.add_argument('--audio', type=str,
#                     help='Filepath of video/audio file to use as raw audio source', required=True)
# parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.',
#                                 default='results/result_voice.mp4')
#

parser.add_argument('--static', type=bool,
                    help='If True, then use only first video frame for inference', default=False)

# NB: I'm adjusting the FPS to match that of the end bump I made. The old default FPS was "25." (float 25)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 29.97)',
                    default=29.97, required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                    help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int,
                    help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int,
            help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                    help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
                    'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                    help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                    'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
                    help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                    'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
                    help='Prevent smoothing face detections over a short temporal window')

parser.add_argument('--audio_start_pad_millis', default=0, type=int,
                    help='Seconds to pad the start of the audio')

parser.add_argument('--audio_end_pad_millis', default=0, type=int,
                    help='Seconds to pad the end of the audio')

parser.add_argument('--end_bump_file', type=str,
                    help='Video file to concatenate at the end')

# This is a little more deliberate than "static" and causes us to pick an FPS, etc.
# This is useful when the filename doesn't have an extension.
parser.add_argument('--is_image', default=False, action='store_true',
                    help='Denote that the input is a single-frame image, not a video')

# Purely for debugging on the host machine:
parser.add_argument('--preserve_tempdir', default=False, action='store_true',
                    help='Keep the tempdir arround for debugging')

args = parser.parse_args()
args.img_size = 96

if os.path.isfile(args.image_or_video_filename) \
        and os.path.splitext(args.image_or_video_filename)[1] in ['.jpg', '.png', '.jpeg']:
    args.static = True
elif args.is_image:
    args.static = True


class FacelessFrameError(Exception):
    """Throw this when there are frames without faces."""
    pass

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                            flip_input=False, device=device)

    batch_size = args.face_det_batch_size

    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
            raise FacelessFrameError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results

def detect_faces_in_frames(frames, video_faces_pickle_file):
    t0 = datetime.datetime.now()
    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect([frames[0]])
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
    t1 = datetime.datetime.now()

    d = t1 - t0
    print('Total seconds to detect faces in {} frames: {}'.format(len(frames), d.total_seconds()))

    with open(video_faces_pickle_file, 'wb') as f:
        print('Saving pickle file: {}'.format(video_faces_pickle_file))
        pickle.dump(face_det_results, f)

    return face_det_results

#def datagen(frames, mels):
def datagen(frames, face_det_results, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    #face_det_results = detect_faces_in_frames(frames)

    for i, m in enumerate(mels):
        idx = 0 if args.static else i%len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

#def _load(checkpoint_path):
#    if device == 'cuda':
#        checkpoint = torch.load(checkpoint_path)
#    else:
#        checkpoint = torch.load(checkpoint_path,
#                                map_location=lambda storage, loc: storage)
#    return checkpoint

#def load_model(path):
#    model = Wav2Lip()
#    print("Load checkpoint from: {}".format(path), flush=True)
#    checkpoint = _load(path)
#    s = checkpoint["state_dict"]
#    new_s = {}
#    for k, v in s.items():
#        new_s[k.replace('module.', '')] = v
#    model.load_state_dict(new_s)
#
#    model = model.to(device)
#    return model.eval()


def main(tempdir):
    video_faces_pickle_file = args.output_cached_faces_filename
    print('Video faces pickle file: {}'.format(video_faces_pickle_file), flush=True)

    is_video = False
    frame_w = 0
    frame_h = 0
    frame_count = 0
    fps = 0.0
    duration_millis = 0.0

    if not os.path.isfile(args.image_or_video_filename):
        raise ValueError('--image_or_video_filename argument must be a valid path to video/image file')

    elif os.path.splitext(args.image_or_video_filename)[1] in ['.jpg', '.png', '.jpeg'] \
            or args.is_image:
        full_frames = [cv2.imread(args.image_or_video_filename)]
        fps = args.fps
        frame_h, frame_w = full_frames[0].shape[:-1]
        is_video = False
        frame_count = len(full_frames)

    else:
        video_stream = cv2.VideoCapture(args.image_or_video_filename)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('Reading video frames...')

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

            if args.rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

        frame_h, frame_w = full_frames[0].shape[:-1]

        if len(full_frames) < 2:
            is_video = False
            frame_count = len(full_frames)
        else:
            is_video = True
            frame_count = len(full_frames)
            duration_seconds = frame_count / fps
            duration_millis = int(duration_seconds * 1000)

    print("Number of frames available for inference: "+str(len(full_frames)), flush=True)
    print("Frame dimensions: {}x{}".format(frame_w, frame_h), flush=True)

    # NB: Instead of truncating, let's detect the entire file.
    # full_frames = full_frames[:len(mel_chunks)]

    print('Face detection time...', flush=True)
    print('Number of frames to detect faces in: {}'.format(len(full_frames)), flush=True)

    print('Detecting faces...', flush=True)
    try:
        _face_det_results = detect_faces_in_frames(full_frames, video_faces_pickle_file)
    except FacelessFrameError as e:
        print('One or more frames without faces were encountered.')
        print(e)
        sys.exit(5)

    print('Done detecting faces!', flush=True)

    mime_type = magic.from_file(args.image_or_video_filename, mime=True)
    file_size_bytes = os.path.getsize(args.image_or_video_filename)

    metadata = {
        'is_video': is_video,
        'width': frame_w,
        'height': frame_h,
        'num_frames': len(full_frames),
        'mimetype': mime_type,
        'file_size_bytes': file_size_bytes,
    }

    if is_video:
        metadata['fps'] = fps
        metadata['duration_millis'] = duration_millis

    with open(args.output_metadata_filename, 'w') as json_file:
        json.dump(metadata, json_file)

if __name__ == '__main__':
    tempdir = tempfile.mkdtemp()
    print('Using Python tempdir: {}'.format(tempdir), flush=True)
    try:
        main(tempdir)
    finally:
        if not args.preserve_tempdir:
            shutil.rmtree(tempdir)

