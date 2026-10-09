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
from PIL import Image

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

parser.add_argument('--face', type=str,
                    help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str,
                    help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.',
                                default='results/result_voice.mp4')

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

if os.path.isfile(args.face) and os.path.splitext(args.face)[1] in ['.jpg', '.png', '.jpeg']:
    args.static = True
elif args.is_image:
    args.static = True

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
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

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

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path), flush=True)
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

def maybe_pad_audio_file(tempdir, args):
    if not args.audio_start_pad_millis and not args.audio_end_pad_millis:
        return

    print('Padding audio with {} millis at start, {} millis at end.'.format(
        args.audio_start_pad_millis, args.audio_end_pad_millis), flush=True)

    padded_wav_filename = os.path.join(tempdir, 'padded_audio.wav')
    # ffmpeg padding: https://superuser.com/a/579110
    # Structure of filters: https://stackoverflow.com/a/55463101
    command = ' '.join([
        'ffmpeg',
        '-i {}'.format(args.audio),
        '-filter_complex "[0] adelay={}ms:all=true [a] ; [a] apad=pad_dur={}ms"'.format(args.audio_start_pad_millis, args.audio_end_pad_millis),
        '{}'.format(padded_wav_filename)
    ])
    print('command:', command, flush=True)
    subprocess.call(command, shell=True)
    args.audio = padded_wav_filename


def maybe_concatenate_end_bump(tempdir, args, frame_w, frame_h):
    if not args.end_bump_file:
        print('No end bump file.', flush=True)
        return

    print('Adding end bump file: {}'.format(args.end_bump_file), flush=True)

    bumpfile = args.end_bump_file

    # May need to resize our bump file...
    # After many attempts to get this into the same ffmpeg pipeline,
    # I'm giving up and just adding another step.
    #
    # NOTE: The FPS of the bump needs to match the video. Especially
    # important for videos generated from still photos, because that defaults
    # to the odd choice of "25 fps".
    if frame_w is not 1920 or frame_h is not 1080:
        print('Need to resize bump.', flush=True)
        bumpfile = os.path.join(tempdir, 'end_bump_resized.mp4')

        # Scale / aspect ratio / sar / resizing / padding:
        # https://stackoverflow.com/a/51946719
        command = ' '.join([
            'ffmpeg',
            '-i {}'.format(args.end_bump_file),
            '-vf "scale={}:{}:force_original_aspect_ratio=decrease,pad={}:{}:(ow-iw)/2:(oh-ih)/2,setsar=1"'.format(frame_w, frame_h, frame_w, frame_h),
            '{}'.format(bumpfile)
        ])
        print('command:', command, flush=True)
        failure_code = subprocess.call(command, shell=True)

        if failure_code:
            print('Failed to resize end bump')
            return

    padded_wav_filename = os.path.join(tempdir, 'output_with_end_bump.mp4')
    # ffmpeg video concatenation: https://stackoverflow.com/a/11175851
    # Structure of filters: https://stackoverflow.com/a/55463101
    # More info on filters: https://stackoverflow.com/a/22958746
    command = ' '.join([
        'ffmpeg',
        '-i {}'.format(args.outfile),
        '-i {}'.format(bumpfile),
        '-filter_complex "concat=n=2:v=1:a=1"',
        '{}'.format(padded_wav_filename)
    ])
    print('command:', command, flush=True)
    failure_code = subprocess.call(command, shell=True)

    if failure_code:
        print('Failed to concatenate end bump')
    else:
        print('Done concatenating end bump; renaming file...', flush=True)
        os.rename(padded_wav_filename, args.outfile)


def main(tempdir):
    video_faces_pickle_file = args.face + '.faces'
    print('Video faces pickle file: {}'.format(video_faces_pickle_file), flush=True)

    frame_w = 0
    frame_h = 0

    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    elif os.path.splitext(args.face)[1] in ['.jpg', '.png', '.jpeg'] \
            or args.is_image:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps
        frame_h, frame_w = full_frames[0].shape[:-1]

    else:
        video_stream = cv2.VideoCapture(args.face)
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

    print("Number of frames available for inference: "+str(len(full_frames)), flush=True)
    print("Frame dimensions: {}x{}".format(frame_w, frame_h), flush=True)

    if not args.audio.endswith('.wav'):
        print('Extracting raw audio...')
        temp_wav_filename = os.path.join(tempdir, 'temp.wav') # previously 'temp/temp.wav'
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, temp_wav_filename)

        print('command:', command, flush=True)
        subprocess.call(command, shell=True)
        args.audio = temp_wav_filename

    maybe_pad_audio_file(tempdir, args)

    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    print(mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80./fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)), flush=True)

    if len(mel_chunks) > 2000:
        # TODO: Catch that the file is too long earlier on.
        raise Exception("Too many mel chunks: {}".format(len(mel_chunks)))

    full_frames = full_frames[:len(mel_chunks)]

    # TODO: Do this offline.
    print('Face detection time...', flush=True)
    if not os.path.isfile(video_faces_pickle_file):
        print('Detecting faces...', flush=True)
        face_det_results = detect_faces_in_frames(full_frames, video_faces_pickle_file)
    else:
        with open(video_faces_pickle_file, 'rb') as f:
            face_det_results = pickle.load(f)
        if len(face_det_results) < len(full_frames):
            print("Face detect results have {} frames, but we need {}".format(len(face_det_results), len(full_frames)))
            print('Detecting faces again...', flush=True)
            face_det_results = detect_faces_in_frames(full_frames, video_faces_pickle_file)
        else:
            print("We don't need to find faces! Woo!", flush=True)

    face_det_results = face_det_results[:len(mel_chunks)]

    batch_size = args.wav2lip_batch_size
    #gen = datagen(full_frames.copy(), mel_chunks)
    gen = datagen(full_frames.copy(), face_det_results.copy(), mel_chunks)

    output_video_filename = os.path.join(tempdir, 'result.avi') # previously 'temp/result.avi'

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
                                            total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
        if i == 0:
            model = load_model(args.checkpoint_path)
            print ("Model loaded")

            #frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter(output_video_filename,
                                    cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()

    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, output_video_filename, args.outfile)
    print('command:', command, flush=True)
    subprocess.call(command, shell=True)

    maybe_concatenate_end_bump(tempdir, args, frame_w, frame_h)

if __name__ == '__main__':
    tempdir = tempfile.mkdtemp()
    print('Using Python tempdir: {}'.format(tempdir), flush=True)
    try:
        main(tempdir)
    finally:
        if not args.preserve_tempdir:
            shutil.rmtree(tempdir)

