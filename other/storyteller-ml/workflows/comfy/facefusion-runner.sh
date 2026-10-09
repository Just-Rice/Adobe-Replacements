#! /bin/bash
set -ex

# we're invoked as 
# ./facefusion-runner.sh --input-audio $input_audio --input-video --output $output
LONGOPTS=input_audio:,input_video:,output:,tmpdir:
OPTIONS=a:v:o:t

PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@") || exit 2
# read getopt’s output this way to handle the quoting right:
eval set -- "$PARSED"


while true; do
  case "$1" in
    -a|--input_audio)
      input_audio="$2"
      shift 2
      ;;
    -v|--input_video)
      input_video="$2"
      shift 2
      ;;
    -o|--output)
      output="$2"
      shift 2
      ;;
    -t|--tmpdir)
      tmpdir="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Programming error"
      exit 3
      ;;
  esac
done

if [ -z "$tmpdir" ]; then
  tmpdir=$(mktemp -d)
fi
TEMPDIR=$tmpdir




# rename /tmp/downloads_long_lived/temp_face_fusion_2.niZOOlINrFkE/audio.bin to /tmp/downloads_long_lived/temp_face_fusion_2.niZOOlINrFkE/audio.wav
cp $input_audio $TEMPDIR/audio.wav
# rename /tmp/downloads_long_lived/temp_face_fusion_2.niZOOlINrFkE/image_or_video.bin to /tmp/downloads_long_lived/temp_face_fusion_2.niZOOlINrFkE/image_or_video.mp4
cp $input_video $TEMPDIR/image_or_video.mp4

echo "input_audio: $input_audio"
echo "input_video: $input_video"
echo "output: $output"
echo "tmpdir: $tmpdir"

export input_audio
export input_video
export output
export TEMPDIR


deactivate || true

cd /facefusion
# Set up the environment
source venv/bin/activate


venv/bin/python3 run.py -s $TEMPDIR/audio.wav -t $TEMPDIR/image_or_video.mp4 --frame-processors lip_syncer --execution-provider cuda -o $TEMPDIR/1.mp4 --headless
venv/bin/python3 run.py -s $TEMPDIR/audio.wav -t $TEMPDIR/1.mp4 --frame-processors face_enhancer --execution-provider cuda -o $output --headless

# rm -rf $TEMPDIR