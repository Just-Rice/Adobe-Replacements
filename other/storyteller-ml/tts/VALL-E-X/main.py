import argparse
import sys
from typing import List
import pathlib

# https://discuss.pytorch.org/t/how-to-calculate-the-gpu-memory-that-a-model-uses/157486/6 <-- for fine tuning memory usage later.

from utils.prompt_making import make_prompt
import logging
from utils.generation import SAMPLE_RATE, generate_audio,load_models_from_pathes
import soundfile as sf
import numpy as np
import torchaudio
from typing import List
import uuid
import torchaudio.transforms as T
class AudioOps():
    
    @staticmethod
    def stitch_wav_tensors_with_crossfade(wav_tensor1:np.array,wav_tensor2:np.array,sample_rate:int):
        # Load two audio signals  as NumPy tensors
        # Compute the length of the cross-fade window in samples
        fade_len = int(0.01 * sample_rate)

        # Create a cross-fade window
        fade_window = np.hanning(2*fade_len)

        # Concatenate the signals with cross-fade
        audio_out = np.concatenate((wav_tensor1[:-fade_len], 
                                    wav_tensor1[-fade_len:] * fade_window[:fade_len] + wav_tensor2[:fade_len] * fade_window[fade_len:], 
                                    wav_tensor2[fade_len:]))
        return audio_out
    
    @staticmethod
    def cut_audio_range(file_path, start_seconds, end_seconds, output_path="cut_audio.wav"):
        """
        Cuts out a specific segment from an audio file and saves the result.

        Args:
        - file_path (str): Path to the audio file from which the segment will be cut.
        - start_seconds (float): Start time of the segment to be cut out in seconds.
        - end_seconds (float): End time of the segment to be cut out in seconds.
        - output_path (str): Path to save the resulting audio. Defaults to "cut_audio.wav".
        
        Returns:
        - None
        """
        # Load the audio file
        waveform, sr = torchaudio.load(file_path)
        # Convert the range to sample indices
        start_sample = int(start_seconds * sr)
        end_sample = int(end_seconds * sr)
        cut_tensor = waveform[:,start_sample:end_sample]
        torchaudio.save(output_path, cut_tensor, sr)
        
    @staticmethod
    def stitch_wav_files_resample(files:List[str],target_sample_rate:int=SAMPLE_RATE):
        resampler = None 
        result = None
        for file in files:  
            if type(result) == type(None):
                tensor, sr = torchaudio.load(pathlib.Path(file))
                resampler = T.Resample(orig_freq=sr,new_freq=target_sample_rate) 
                resampled_tensor = resampler(tensor)
                result = resampled_tensor.squeeze()
                result = np.array(result)
            else:  
                tensor, _ = torchaudio.load(pathlib.Path(file))
                resampled_tensor = resampler(tensor)
                resampled_tensor = resampled_tensor.squeeze()
                resampled_tensor = np.array(resampled_tensor)
                result = AudioOps.stitch_wav_tensors_with_crossfade(result,resampled_tensor,sample_rate=target_sample_rate)
                
        return result
    
class VoiceDesigner():
    def __init__(self) -> None:
        # load models from host system.
        self.logger = self.setup_logger()
        #  host path for work
        self.temp_path = None
        
    def tts_with_prompt(self, prompt_dir:pathlib.Path,
                        prompt_name:pathlib.Path,
                        audio_output_path:pathlib.Path, 
                        output_file_name:pathlib.Path,
                        text:str):
        
        audio_array = generate_audio(text,prompt_name=prompt_name,prompt_dir=prompt_dir)
        # save audio to disk
        sf.write(audio_output_path / output_file_name, audio_array, SAMPLE_RATE)
        
    def initialize_voice_designer(self,vall_e_x_file_path:pathlib.Path,vocos_folder_path:pathlib.Path,whisper_folder_path:pathlib.Path):
        load_models_from_pathes(vocos_folder_path=vocos_folder_path,vall_e_x_file_path=vall_e_x_file_path,whisper_folder_path=whisper_folder_path)
        
    def create_prompt(self,audio_file_paths:List[pathlib.Path],
                      prompt_file_name:pathlib.Path,
                      prompt_output_path:pathlib.Path,
                      whisper_folder_path:pathlib.Path):
        # multiple audio files concat
        # concat audio and trim the audio if it is longer than 14-15 seconds# if prompt is too long it will error out anyways
        if len(audio_file_paths) > 0:
            # try:# ./customs/ <-- folder it adds by default until we modified.
                    # load the audio files and then concat them. 
            tensor_result = AudioOps.stitch_wav_files_resample(files=audio_file_paths)
            # save to disk
            generated_name = f"{uuid.uuid4()}"
            
            concat_audio_file_path = self.temp_path / pathlib.Path(f"audio_concat_{generated_name}.wav")               
            sf.write(concat_audio_file_path, tensor_result, SAMPLE_RATE)
            # save result into processing...
            # trim to 15 seconds.
            audio_cut_path = self.temp_path / pathlib.Path(f"audio_cut_{generated_name}.wav")
            AudioOps.cut_audio_range(file_path=concat_audio_file_path,start_seconds=0,end_seconds=15,output_path=audio_cut_path)
            make_prompt(name=prompt_file_name,audio_prompt_output_path=prompt_output_path,audio_prompt_path=audio_cut_path,whisper_folder_path=whisper_folder_path)
                
        else:
            self.logger.debug(f"No audio files to load")
            
    def setup_logger(self):
        # Create a logger object
        logger = logging.getLogger(__name__)

        # Set the log level
        logger.setLevel(logging.DEBUG)

        # Create a file handler to log messages to a file
        file_handler = logging.FileHandler('docker_valle-x.log')
        file_handler.setLevel(logging.DEBUG)

        # Create a console handler to log messages to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Set the formatter for the handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

global voice_designer 
voice_designer = VoiceDesigner()

def main(args):
    print("Starting Vall-E-X")
    print(f"Python Version: {sys.version_info}")

    print("Starting Inference on Vall-E-X With Inputs")
    print(f"Mode: {args.mode}")
    
    print(f"Text: {args.text}")
    print(f"List of Files For Create: {args.audio_wav_files}")
    print(f"Whisper Folder Path: {args.whisper_folder_path}")
    print(f"Vocos Folder Path: {args.vocos_folder_path}")
    print(f"Vall-E-X Path: {args.vallex_path}")
    
    print(f"Prompt Path: {args.prompt_path}")
    print(f"Prompt Name: {args.prompt_name}")
    
    print(f"Audio Name: {args.audio_name}")
    print(f"Audio Path: {args.audio_path}")
    
    print(f"Temp Work Dir: {args.tmp_work_dir}")
    
    voice_designer.temp_path = pathlib.Path(args.tmp_work_dir)

    voice_designer.initialize_voice_designer(pathlib.Path(args.vallex_path),
                                             pathlib.Path(args.vocos_folder_path),
                                             pathlib.Path(args.whisper_folder_path))
    
    if args.mode == 0: # run inference
        print("Running Inference")
        voice_designer.tts_with_prompt(prompt_dir=pathlib.Path(args.prompt_path),
                                       prompt_name=pathlib.Path(args.prompt_name),
                                       audio_output_path=pathlib.Path(args.audio_path),
                                       output_file_name=pathlib.Path(args.audio_name),
                                       text=args.text)
    
    elif args.mode == 1: # run create voice
        voice_designer.create_prompt(audio_file_paths=args.audio_wav_files,
                                     prompt_output_path=pathlib.Path(args.prompt_path),
                                     prompt_file_name=pathlib.Path(args.prompt_name),
                                     whisper_folder_path=pathlib.Path(args.whisper_folder_path))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a list of speaker samples for zero shot - .wav file pathes, convert text to speech, and specify language.")
    # modes
    parser.add_argument('--mode', metavar='model mode',type=int,help='Mode 0 - (Inference) | Mode 1 - (Create)')
    parser.add_argument('--text', metavar='text to synthesize', type=str,help='Text to be converted to speech.')
    # inputs
    parser.add_argument('--audio-wav-files', metavar='wav to voice clone', type=str,nargs='+',help='Takes in a list of .wav files for an audio prompt provide the absolute path. Must be less than 15 seconds otherwise will error out. (Create Voice Only)')           
    
    # model pathes 
    parser.add_argument("--whisper-folder-path",metavar="abs whisper folder path",type=str,help="Path to the Whisper folder where the .pt file")
    parser.add_argument("--whisper-model",metavar="model large or medium",type=str,help="medium or large")
    parser.add_argument("--vocos-folder-path",metavar="abs vocos model path",type=str,help="Path to vocos codec Model requires the (folder path)")
    parser.add_argument("--vallex-path",metavar="abs vall-e model path",type=str,help="Path to the Valle Model File")
    
    # output and input pathes
    parser.add_argument("--audio-name",metavar="abs output path",type=str,help="Audio Output Saved Name (Inference)")   
    parser.add_argument("--audio-path",metavar="abs path to prompt",type=str,help="Audio Output Path (Inference)")
    parser.add_argument("--prompt-path",metavar="abs path to prompt",type=str,help="Prompt Embedding Path (Inference and Create Voice)")
    parser.add_argument("--prompt-name",metavar="abs path to prompt",type=str,help="Prompt Name (Inference and Create Voice)")
    
    parser.add_argument("--tmp-work-dir",metavar="abs tmp dir",type=str,help="Temporary work directory in host system.")
    
    args = parser.parse_args()

    main(args)


# inference example
#/home/tensor/code/storyteller/storyteller-ml/tts/VALL-E-X/.venv/bin/python /home/tensor/code/storyteller/storyteller-ml/tts/VALL-E-X/main.py --text "hello world" --mode 0 --whisper-folder-path "/home/tensor/code/storyteller/storyteller-ml/tts/VALL-E-X/whisper" --whisper-model medium --vocos-folder-path "/home/tensor/code/TTSDockerContainer/Vall-E-mount/models/vocos-encodec-24khz" --vallex-path "/home/tensor/code/TTSDockerContainer/Vall-E-mount/models/vallex-checkpoint.pt" --prompt-path "/home/tensor/code/storyteller/VALL-E-X-TTS-Container/Vall-E-mount/prompts" --prompt-name "goku.npz" --audio-path "/home/tensor/code/storyteller/VALL-E-X-TTS-Container/Vall-E-mount/output/" --audio-name "hello_world_test.wav" --tmp-work-dir "/tmp"

#/home/tensor/code/storyteller/storyteller-ml/tts/VALL-E-X/.venv/bin/python
#/home/tensor/code/storyteller/storyteller-ml/tts/VALL-E-X/main.py 

# --text "hello world" 
# --mode 0 
# --whisper-folder-path "/home/tensor/code/storyteller/storyteller-ml/tts/VALL-E-X/whisper" 
# --whisper-model medium 
# --vocos-folder-path "/home/tensor/code/TTSDockerContainer/Vall-E-mount/models/vocos-encodec-24khz" 
# --vallex-path "/home/tensor/code/TTSDockerContainer/Vall-E-mount/models/vallex-checkpoint.pt" 
# --prompt-path "/home/tensor/code/storyteller/VALL-E-X-TTS-Container/Vall-E-mount/prompts" 
# --prompt-name "goku" 
# --audio-path "/home/tensor/code/storyteller/VALL-E-X-TTS-Container/Vall-E-mount/output/" 
# --audio-name "hello_world_test.wav" 
# --tmp-work-dir "/tmp"


# create voice example
#/home/tensor/code/storyteller/storyteller-ml/tts/VALL-E-X/.venv/bin/python /home/tensor/code/storyteller/storyteller-ml/tts/VALL-E-X/main.py --mode 1 --audio-wav-files "/home/tensor/code/TTSDockerContainer/Vall-E-mount/input/20.wav" "/home/tensor/code/TTSDockerContainer/Vall-E-mount/input/21.wav" --whisper-folder-path "/home/tensor/code/storyteller/storyteller-ml/tts/VALL-E-X/whisper" --whisper-model medium --vocos-folder-path "/home/tensor/code/TTSDockerContainer/Vall-E-mount/models/vocos-encodec-24khz" --vallex-path "/home/tensor/code/TTSDockerContainer/Vall-E-mount/models/vallex-checkpoint.pt" --prompt-path "/home/tensor/code/storyteller/VALL-E-X-TTS-Container/Vall-E-mount/prompts" --prompt-name "test_prompt" --tmp-work-dir "/tmp"

#/home/tensor/code/storyteller/storyteller-ml/tts/VALL-E-X/.venv/bin/python 
#/home/tensor/code/storyteller/storyteller-ml/tts/VALL-E-X/main.py 
# --mode 1 
# --audio-wav-files 
# "/home/tensor/code/TTSDockerContainer/Vall-E-mount/input/20.wav" 
# "/home/tensor/code/TTSDockerContainer/Vall-E-mount/input/21.wav" 
# --whisper-folder-path "/home/tensor/code/storyteller/storyteller-ml/tts/VALL-E-X/whisper" 
# --whisper-model medium 
# --vocos-folder-path "/home/tensor/code/TTSDockerContainer/Vall-E-mount/models/vocos-encodec-24khz" 
# --vallex-path "/home/tensor/code/TTSDockerContainer/Vall-E-mount/models/vallex-checkpoint.pt" 
# --prompt-path "/home/tensor/code/storyteller/VALL-E-X-TTS-Container/Vall-E-mount/prompts" 
# --prompt-name "test_prompt" 
# --tmp-work-dir "/tmp"
